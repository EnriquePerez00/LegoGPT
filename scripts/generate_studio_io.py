import os
import sys
import json
import zipfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inventory_matcher import LegoInventoryMatcher
from src.manual_connection_graph import ManualConnectionGraphBuilder


def create_studio_io(out_path: str, nodes: list, model_title: str) -> str:
    """
    Builds a compliant BrickLink Studio .io container with model.ldr, model2.ldr, .info, and errorPartList.err.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    filename = os.path.basename(out_path)
    ldr_lines = [
        f"0 FILE {filename}",
        f"0 LEGO {model_title}",
        f"0 Name:  {filename}",
        "0 Author: LegoGPT Generator",
        "0 CustomBrick",
        "0 FlexibleBrickControlPointUnitLength -1",
        "0 FlexibleBrickLockedControlPoint",
        f"0 TotalBricks: {len(nodes)}"
    ]

    current_step = None
    for n in nodes:
        step_lbl = n.get("step_label", f"Page {n['page_num']}")
        if current_step is not None and current_step != step_lbl:
            ldr_lines.append("0 STEP")
        current_step = step_lbl

        t = n["absolute_transform"]
        col = n["color_code"]
        pid = n["part_id"]
        # LDraw format: 1 <color> x y z a b c d e f g h i <part>
        line = (
            f"1 {col} {t[3]:.4f} {t[7]:.4f} {t[11]:.4f} "
            f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{t[4]:.6f} {t[5]:.6f} {t[6]:.6f} "
            f"{t[8]:.6f} {t[9]:.6f} {t[10]:.6f} "
            f"{pid}"
        )
        ldr_lines.append(line)

    ldr_lines.append("0 NOFILE")
    model_ldr_content = "\n".join(ldr_lines) + "\n"

    info_json = json.dumps({
        "version": "2.25.11_3",
        "total_parts": len(nodes),
        "parts_db_version": 200
    })

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.ldr", model_ldr_content.encode("utf-8"))
        zf.writestr("model2.ldr", model_ldr_content.encode("utf-8"))
        zf.writestr(".info", info_json.encode("utf-8"))
        zf.writestr("errorPartList.err", b"[]")

    return out_path


def generate_io_and_graph(
    pdf_path: str = "data/manual/official/75295-1_Millennium_Falcon_Microfighter.pdf",
    set_num: str = "75295-1",
    db_path: str = "data/catalog/models_catalog.db",
    downloads_dir: str = "/Users/I764690/Downloads"
):
    print(f"=== Extrayendo Grafo de Construcción para Set {set_num} ===")
    matcher = LegoInventoryMatcher(pdf_path, temp_dir=f"scratch/inventory_templates_{set_num}")
    res_steps = matcher.extract_and_match_all_steps(db_path=db_path, set_num=set_num)

    builder = ManualConnectionGraphBuilder(pdf_path, set_num=set_num, db_path=db_path)
    graph = builder.build_connection_graph(res_steps["step_pages"])

    nodes = graph["nodes"]
    print(f"Grafo generado exitosamente: {len(nodes)} nodos de piezas, {graph['ground_parts_touching']} en base.")

    # Save JSON graph locally in outputs/ and data/
    os.makedirs("outputs", exist_ok=True)
    json_path = f"outputs/{set_num}_construction_graph.json"

    def default_converter(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, default=default_converter)
    print(f"Grafo de construcción guardado en: {json_path}")

    # Reference without version suffix (e.g. '75295')
    ref = set_num.split("-")[0]

    # Separate nodes into minifigures and main assembly
    main_nodes = [n for n in nodes if not n.get("is_minifig", False)]
    
    # Group minifigures by minifig_index
    minifig_groups: dict[int, list] = {}
    for n in nodes:
        if n.get("is_minifig", False):
            idx = n.get("minifig_index", 1) or 1
            minifig_groups.setdefault(idx, []).append(n)

    generated_files = []

    # 1. Generate minifigX.io for each minifigure in downloads_dir and outputs/
    for mf_idx, mf_nodes in sorted(minifig_groups.items()):
        mf_name = f"minifig{mf_idx}.io"
        mf_dl_path = os.path.join(downloads_dir, mf_name)
        mf_out_path = os.path.join("outputs", mf_name)
        title = f"Minifigure {mf_idx} (Set {set_num})"

        create_studio_io(mf_dl_path, mf_nodes, title)
        create_studio_io(mf_out_path, mf_nodes, title)
        generated_files.append(mf_dl_path)
        print(f"📦 Minifigura {mf_idx} generada ({len(mf_nodes)} piezas) -> {mf_dl_path}")

    # 2. Generate ref.io for the main model (without minifigures) in downloads_dir and outputs/
    ref_name = f"{ref}.io"
    ref_dl_path = os.path.join(downloads_dir, ref_name)
    ref_out_path = os.path.join("outputs", ref_name)
    main_title = f"Set {ref} (Sin minifiguras)"

    create_studio_io(ref_dl_path, main_nodes, main_title)
    create_studio_io(ref_out_path, main_nodes, main_title)
    generated_files.append(ref_dl_path)
    print(f"🚀 Set principal '{ref}.io' generado ({len(main_nodes)} piezas) -> {ref_dl_path}")

    # Also keep complete model as legacy reference in outputs
    complete_io_name = f"{set_num}_complete.io"
    complete_out_path = os.path.join("outputs", complete_io_name)
    create_studio_io(complete_out_path, nodes, f"Set {set_num} (Completo)")

    return {
        "json_path": json_path,
        "ref_io": ref_dl_path,
        "minifig_ios": [os.path.join(downloads_dir, f"minifig{idx}.io") for idx in sorted(minifig_groups.keys())],
        "total_nodes": len(nodes),
        "main_nodes_count": len(main_nodes),
        "minifigs_count": len(minifig_groups)
    }


if __name__ == "__main__":
    generate_io_and_graph()
