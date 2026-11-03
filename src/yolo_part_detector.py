import os
import re
import ssl
import shutil
import cv2
import numpy as np
import pymupdf
import torch

ssl._create_default_https_context = ssl._create_unverified_context


def refine_tight_bbox_cv(rgb_crop: np.ndarray, bg_tol: float = 14.0, pad: int = 2) -> tuple[int, int, int, int]:
    """
    Given an RGB image region (from an inventory cell or callout cell),
    computes the tight rectangular bounding box (x0, y0, x1, y1) around the LEGO part pixels
    by estimating background color from the border in CIE LAB space.
    """
    h, w = rgb_crop.shape[:2]
    if h < 4 or w < 4:
        return (0, 0, w, h)

    lab = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_pixels = np.concatenate([
        lab[0, :, :],
        lab[-1, :, :],
        lab[:, 0, :],
        lab[:, -1, :]
    ], axis=0)
    bg_lab = np.median(border_pixels, axis=0)

    dist = np.linalg.norm(lab - bg_lab[None, None, :], axis=2)
    fg_mask = (dist > bg_tol).astype(np.uint8)

    # Clean isolated 1px noise and close small internal gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    ys, xs = np.where(fg_mask > 0)
    if len(xs) < 6:
        return (0, 0, w, h)

    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return (x0, y0, x1, y1)


class YoloManualPartDetector:
    """
    Hybrid CV + YOLOv8 rectangular detector for LEGO instruction manuals.
    Supports:
      1. Automatic generation of YOLO-format datasets from PDF manuals.
      2. Training YOLOv8n on Apple Silicon GPU (MPS).
      3. Running rectangular inference + tight CV pixel refinement on manual pages.
    """
    CLASSES = {
        0: "inventory_part",
        1: "step_part",
        2: "callout_box",
        3: "qty_label",
        4: "element_code"
    }

    def __init__(self, weights_path: str = "models/yolo_manual_parts_mps.pt"):
        self.weights_path = weights_path
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.max_workers = max(1, (os.cpu_count() or 8) - 2)
        self.model = None
        if os.path.exists(self.weights_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights_path)
            except Exception as e:
                print(f"Aviso cargando pesos YOLO ({self.weights_path}): {e}")

    @staticmethod
    def _to_yolo_norm(bbox: tuple[float, float, float, float], page_w: float, page_h: float) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        x0 = max(0.0, min(page_w, x0))
        x1 = max(0.0, min(page_w, x1))
        y0 = max(0.0, min(page_h, y0))
        y1 = max(0.0, min(page_h, y1))
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        xc = (x0 + x1) / 2.0 / page_w
        yc = (y0 + y1) / 2.0 / page_h
        return (
            min(1.0, max(0.0, xc)),
            min(1.0, max(0.0, yc)),
            min(1.0, max(0.001, bw / page_w)),
            min(1.0, max(0.001, bh / page_h))
        )

    def extract_page_annotations(self, doc: pymupdf.Document, page_idx: int, is_inventory_page: bool) -> list[tuple[int, tuple[float, float, float, float]]]:
        """Extracts ground-truth bounding boxes in PDF point coordinates for a single page."""
        page = doc[page_idx]
        words = page.get_text("words")
        imgs = page.get_image_info(xrefs=True)
        drawings = page.get_drawings()

        annotations = []
        qtys = [w for w in words if re.match(r"^\d+x$", w[4], re.IGNORECASE)]
        codes = [w for w in words if re.match(r"^\d{6,7}$", w[4])]

        if is_inventory_page:
            for cw in codes:
                cx0, cy0, cx1, cy1, _ = cw[:5]
                annotations.append((4, (cx0, cy0, cx1, cy1)))
                qw_candidates = [qw for qw in qtys if abs(qw[0] - cx0) < 18 and 0 <= (cy0 - qw[1]) < 22]
                if not qw_candidates:
                    continue
                qw = min(qw_candidates, key=lambda q: abs(cy0 - q[3]) + abs(q[0] - cx0))
                qx0, qy0, qx1, qy1 = qw[:4]
                annotations.append((3, (qx0, qy0, qx1, qy1)))

                im_candidates = [
                    im for im in imgs
                    if (im["bbox"][2] - im["bbox"][0]) < 130
                    and (im["bbox"][3] - im["bbox"][1]) < 130
                    and abs(im["bbox"][0] - qx0) < 18
                    and -5 <= (qy0 - im["bbox"][3]) < 28
                ]
                if im_candidates:
                    im_best = min(im_candidates, key=lambda im: abs(qy0 - im["bbox"][3]) + abs(im["bbox"][0] - qx0))
                    annotations.append((0, tuple(im_best["bbox"])))
        else:
            # Step callout boxes
            step_qtys = [qw for qw in qtys if qw[1] < page.rect.height * 0.5]
            for draw in drawings:
                rect = draw.get("rect")
                fill = draw.get("fill")
                if rect and 20 < rect.width < page.rect.width * 0.85 and 20 < rect.height < page.rect.height * 0.65:
                    if fill and len(fill) >= 3 and 0.65 < fill[0] < 0.96 and abs(fill[0] - fill[1]) < 0.15:
                        # Check if it encloses at least one qty
                        if any(rect.x0 - 5 <= qw[0] <= rect.x1 + 5 and rect.y0 - 5 <= qw[1] <= rect.y1 + 5 for qw in step_qtys):
                            annotations.append((2, (rect.x0, rect.y0, rect.x1, rect.y1)))

            for qw in step_qtys:
                qx0, qy0, qx1, qy1 = qw[:4]
                annotations.append((3, (qx0, qy0, qx1, qy1)))
                im_candidates = [
                    im for im in imgs
                    if (im["bbox"][2] - im["bbox"][0]) < 150
                    and (im["bbox"][3] - im["bbox"][1]) < 150
                    and abs(im["bbox"][0] - qx0) < 22
                    and -5 <= (qy0 - im["bbox"][3]) < 28
                ]
                if im_candidates:
                    im_best = min(im_candidates, key=lambda im: abs(qy0 - im["bbox"][3]) + abs(im["bbox"][0] - qx0))
                    annotations.append((1, tuple(im_best["bbox"])))

        return annotations

    def build_yolo_dataset(self, pdf_paths: list[str], output_dir: str = "data/yolo_manual_dataset", dpi: int = 150) -> str:
        """Generates a complete YOLOv8 dataset (train/val images & labels + dataset.yaml) from manual PDFs."""
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        for split in ("train", "val"):
            os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

        total_pages_exported = 0
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                continue
            prefix = os.path.splitext(os.path.basename(pdf_path))[0].split("_")[0]
            doc = pymupdf.open(pdf_path)
            num_pages = len(doc)

            # Identify inventory pages
            inv_pages = set()
            for idx in range(max(0, num_pages - 10), num_pages):
                codes = re.findall(r"\b\d{6,7}\b", doc[idx].get_text())
                if len(codes) >= 5:
                    inv_pages.add(idx)

            for idx in range(num_pages):
                is_inv = idx in inv_pages
                anns = self.extract_page_annotations(doc, idx, is_inv)
                if not anns:
                    continue

                split = "val" if (total_pages_exported % 5 == 0) else "train"
                page = doc[idx]
                pix = page.get_pixmap(dpi=dpi)
                img_name = f"{prefix}_p{idx + 1:03d}.png"
                lbl_name = f"{prefix}_p{idx + 1:03d}.txt"

                img_out = os.path.join(output_dir, "images", split, img_name)
                lbl_out = os.path.join(output_dir, "labels", split, lbl_name)
                pix.save(img_out)

                pw, ph = page.rect.width, page.rect.height
                with open(lbl_out, "w", encoding="utf-8") as lf:
                    for cls_id, bbox in anns:
                        xc, yc, bw, bh = self._to_yolo_norm(bbox, pw, ph)
                        lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                total_pages_exported += 1
            doc.close()

        yaml_path = os.path.join(output_dir, "dataset.yaml")
        abs_out = os.path.abspath(output_dir)
        yaml_content = (
            f"path: {abs_out}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n"
            f"  0: inventory_part\n"
            f"  1: step_part\n"
            f"  2: callout_box\n"
            f"  3: qty_label\n"
            f"  4: element_code\n"
        )
        with open(yaml_path, "w", encoding="utf-8") as yf:
            yf.write(yaml_content)

        print(f"Dataset YOLO generado en {output_dir} con {total_pages_exported} páginas etiquetadas.")
        return yaml_path

    def train(self, dataset_yaml: str, epochs: int = 8, imgsz: int = 640) -> str:
        """Trains YOLOv8n on Apple Silicon MPS and exports weights to self.weights_path."""
        from ultralytics import YOLO
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        base_model = YOLO("yolov8n.pt")
        results = base_model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            device=self.device,
            workers=min(4, self.max_workers),
            batch=16,
            project="scratch/yolo_runs",
            name="manual_parts_detector",
            exist_ok=True,
            verbose=False
        )
        save_dir = str(getattr(results, "save_dir", os.path.join("scratch/yolo_runs", "manual_parts_detector")))
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(best_pt):
            shutil.copy(best_pt, self.weights_path)
            self.model = YOLO(self.weights_path)
            print(f"Modelo YOLO entrenado en {self.device.upper()} y guardado en {self.weights_path}")
        return self.weights_path

    def detect_on_image(self, image_path: str, conf: float = 0.25) -> list[dict]:
        """Runs YOLO rectangular inference on a rendered page image."""
        if self.model is None:
            return []
        preds = self.model.predict(source=image_path, conf=conf, device=self.device, verbose=False)
        detections = []
        if not preds:
            return detections
        res = preds[0]
        boxes = res.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            x0, y0, x1, y1 = [float(v) for v in boxes.xyxy[i].tolist()]
            detections.append({
                "class_id": cls_id,
                "class_name": self.CLASSES.get(cls_id, "unknown"),
                "confidence": score,
                "bbox_px": (x0, y0, x1, y1)
            })
        return detections
