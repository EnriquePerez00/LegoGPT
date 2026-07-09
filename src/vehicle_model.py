"""src/vehicle_model.py - Phase 2: VehicleHierarchicalSoftmax + GraphFingerprintConditioner + VehicleLegoGNN"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from typing import Optional
from torch_geometric.nn import SAGEConv, global_mean_pool
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS

CATEGORY_CHASSIS = ["2926.dat","4150.dat","3024.dat","3023.dat","3022.dat","3021.dat","3020.dat","3710.dat","3666.dat","3460.dat","3034.dat","3832.dat","3032.dat","3031.dat","3030.dat","3035.dat","3036.dat","3958.dat","3795.dat"]
CATEGORY_BRICK_BODY = ["3005.dat","3004.dat","3003.dat","3002.dat","3001.dat","3010.dat","3009.dat","3008.dat","3007.dat","3006.dat","3622.dat","30029.dat"]
CATEGORY_WHEEL = ["3139.dat","42610.dat","56902.dat","30027.dat","18976.dat","18977.dat","55981.dat","55982.dat","30285.dat","30285b.dat","6014.dat","6015.dat","56890.dat","56891.dat","6014b.dat"]
CATEGORY_CABIN = ["3823.dat","3829.dat","3829c01.dat","4274.dat"]
CATEGORY_SLOPE = ["50950.dat","61678.dat","85970.dat","54200.dat","85984.dat","6091.dat","11477.dat"]
CATEGORY_SPECIAL = ["50745.dat","98282.dat","3069b.dat","2431.dat","87079.dat","3070b.dat","63864.dat","6636.dat","3794a.dat","86996.dat","3623.dat"]
FINGERPRINT_DIM = len(VEHICLE_ALLOWED_PARTS) + 6


class VehicleHierarchicalSoftmax(nn.Module):
    def __init__(self, hidden_dim, allowed_parts):
        super().__init__()
        self.allowed_parts = allowed_parts
        self.cat_chassis = [i for i,p in enumerate(allowed_parts) if p in CATEGORY_CHASSIS]
        self.cat_brick   = [i for i,p in enumerate(allowed_parts) if p in CATEGORY_BRICK_BODY]
        self.cat_wheel   = [i for i,p in enumerate(allowed_parts) if p in CATEGORY_WHEEL]
        self.cat_cabin   = [i for i,p in enumerate(allowed_parts) if p in CATEGORY_CABIN]
        self.cat_slope   = [i for i,p in enumerate(allowed_parts) if p in CATEGORY_SLOPE]
        self.cat_special = [i for i,p in enumerate(allowed_parts) if p in CATEGORY_SPECIAL]
        self.l1            = nn.Linear(hidden_dim, 2)
        self.l2_structural = nn.Linear(hidden_dim, 2)
        self.l2_functional = nn.Linear(hidden_dim, 4)
        self.l3_chassis    = nn.Linear(hidden_dim, max(len(self.cat_chassis), 1))
        self.l3_brick      = nn.Linear(hidden_dim, max(len(self.cat_brick),   1))
        self.l3_wheel      = nn.Linear(hidden_dim, max(len(self.cat_wheel),   1))
        self.l3_cabin      = nn.Linear(hidden_dim, max(len(self.cat_cabin),   1))
        self.l3_slope      = nn.Linear(hidden_dim, max(len(self.cat_slope),   1))
        self.l3_special    = nn.Linear(hidden_dim, max(len(self.cat_special), 1))
        n = len(allowed_parts)
        l1_lbl = torch.zeros(n, dtype=torch.long)
        l2_lbl = torch.zeros(n, dtype=torch.long)
        lf_lbl = torch.zeros(n, dtype=torch.long)
        for idx, part in enumerate(allowed_parts):
            if part in CATEGORY_CHASSIS:
                l1_lbl[idx], l2_lbl[idx] = 0, 0
                lf_lbl[idx] = self.cat_chassis.index(idx) if idx in self.cat_chassis else 0
            elif part in CATEGORY_BRICK_BODY:
                l1_lbl[idx], l2_lbl[idx] = 0, 1
                lf_lbl[idx] = self.cat_brick.index(idx) if idx in self.cat_brick else 0
            elif part in CATEGORY_WHEEL:
                l1_lbl[idx], l2_lbl[idx] = 1, 0
                lf_lbl[idx] = self.cat_wheel.index(idx) if idx in self.cat_wheel else 0
            elif part in CATEGORY_CABIN:
                l1_lbl[idx], l2_lbl[idx] = 1, 1
                lf_lbl[idx] = self.cat_cabin.index(idx) if idx in self.cat_cabin else 0
            elif part in CATEGORY_SLOPE:
                l1_lbl[idx], l2_lbl[idx] = 1, 2
                lf_lbl[idx] = self.cat_slope.index(idx) if idx in self.cat_slope else 0
            else:
                l1_lbl[idx], l2_lbl[idx] = 1, 3
                lf_lbl[idx] = self.cat_special.index(idx) if idx in self.cat_special else 0
        self.register_buffer("l1_labels",   l1_lbl)
        self.register_buffer("l2_labels",   l2_lbl)
        self.register_buffer("leaf_labels", lf_lbl)

    def forward(self, h):
        B, device = h.size(0), h.device
        p_l1  = F.softmax(self.l1(h),            dim=-1)
        p_l2s = F.softmax(self.l2_structural(h), dim=-1)
        p_l2f = F.softmax(self.l2_functional(h), dim=-1)
        p_ch  = F.softmax(self.l3_chassis(h), dim=-1)
        p_br  = F.softmax(self.l3_brick(h),   dim=-1)
        p_wh  = F.softmax(self.l3_wheel(h),   dim=-1)
        p_ca  = F.softmax(self.l3_cabin(h),   dim=-1)
        p_sl  = F.softmax(self.l3_slope(h),   dim=-1)
        p_sp  = F.softmax(self.l3_special(h), dim=-1)
        full = torch.zeros(B, len(self.allowed_parts), device=device)
        if self.cat_chassis: full[:, self.cat_chassis] = p_l1[:,0:1]*p_l2s[:,0:1]*p_ch
        if self.cat_brick:   full[:, self.cat_brick]   = p_l1[:,0:1]*p_l2s[:,1:2]*p_br
        if self.cat_wheel:   full[:, self.cat_wheel]   = p_l1[:,1:2]*p_l2f[:,0:1]*p_wh
        if self.cat_cabin:   full[:, self.cat_cabin]   = p_l1[:,1:2]*p_l2f[:,1:2]*p_ca
        if self.cat_slope:   full[:, self.cat_slope]   = p_l1[:,1:2]*p_l2f[:,2:3]*p_sl
        if self.cat_special: full[:, self.cat_special] = p_l1[:,1:2]*p_l2f[:,3:4]*p_sp
        return torch.log(full + 1e-12)

    def compute_loss(self, h, targets):
        tgt_l1 = self.l1_labels[targets]
        tgt_l2 = self.l2_labels[targets]
        tgt_lf = self.leaf_labels[targets]
        loss = F.cross_entropy(self.l1(h), tgt_l1)
        is_s, is_f = (tgt_l1==0), (tgt_l1==1)
        if is_s.any(): loss += F.cross_entropy(self.l2_structural(h[is_s]), tgt_l2[is_s])
        if is_f.any(): loss += F.cross_entropy(self.l2_functional(h[is_f]), tgt_l2[is_f])
        for lin, mask in [
            (self.l3_chassis, (tgt_l1==0)&(tgt_l2==0)),
            (self.l3_brick,   (tgt_l1==0)&(tgt_l2==1)),
            (self.l3_wheel,   (tgt_l1==1)&(tgt_l2==0)),
            (self.l3_cabin,   (tgt_l1==1)&(tgt_l2==1)),
            (self.l3_slope,   (tgt_l1==1)&(tgt_l2==2)),
            (self.l3_special, (tgt_l1==1)&(tgt_l2==3)),
        ]:
            if mask.any(): loss += F.cross_entropy(lin(h[mask]), tgt_lf[mask])
        return loss


class GraphFingerprintConditioner(nn.Module):
    def __init__(self, fingerprint_dim=FINGERPRINT_DIM, cond_dim=32):
        super().__init__()
        self.fingerprint_dim = fingerprint_dim
        self.cond_dim = cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(fingerprint_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, cond_dim), nn.ReLU(),
        )

    def forward(self, fingerprint):
        return self.encoder(fingerprint)

    @staticmethod
    def build_fingerprint(target_inventory, current_parts, target_parts_count=100, allowed_parts=None):
        from src.vehicle_rules import evaluate_vehicle_topology
        vocab = allowed_parts if allowed_parts is not None else VEHICLE_ALLOWED_PARTS
        vsize = len(vocab)
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        total_tgt = max(sum(target_inventory.values()), 1) if target_inventory else 1
        for i, pid in enumerate(vocab):
            fp[i] = min(target_inventory.get(pid, 0) / total_tgt, 1.0)
        if current_parts:
            m = evaluate_vehicle_topology(current_parts)
            fp[vsize+0] = min(m.get("wheel_count", 0) / 10.0, 1.0)
            fp[vsize+1] = float(m.get("symmetry_score", 0.0))
            fp[vsize+2] = 1.0 if m.get("has_cabin_parts", False) else 0.0
            fp[vsize+3] = 1.0 if m.get("wheels_at_bottom", False) else 0.0
        fp[vsize+4] = min(len(current_parts) / max(target_parts_count, 1), 1.0)
        unique = len(set(p.part_id for p in current_parts))
        fp[vsize+5] = min(unique / max(vsize, 1), 1.0)
        return torch.tensor(fp, dtype=torch.float32).unsqueeze(0)


class VehicleLegoGNN(nn.Module):
    def __init__(self, num_color_classes=8, hidden_dim=64, cond_dim=32, use_fingerprint=True):
        super().__init__()
        n_parts = len(VEHICLE_ALLOWED_PARTS)
        in_channels = n_parts + num_color_classes + 12
        self.use_fingerprint = use_fingerprint
        self.cond_dim = cond_dim if use_fingerprint else 0

        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        fc_in = hidden_dim + self.cond_dim
        self.fc = nn.Sequential(nn.Linear(fc_in, hidden_dim), nn.ReLU(), nn.Dropout(0.1))

        self.head_part      = VehicleHierarchicalSoftmax(hidden_dim, VEHICLE_ALLOWED_PARTS)
        self.head_color     = nn.Linear(hidden_dim, num_color_classes)
        self.head_transform = nn.Linear(hidden_dim, 12)

        if use_fingerprint:
            self.conditioner = GraphFingerprintConditioner(
                fingerprint_dim=FINGERPRINT_DIM, cond_dim=cond_dim
            )

    def forward(self, x, edge_index, batch, fingerprint=None):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        g = global_mean_pool(h, batch)

        if self.use_fingerprint and fingerprint is not None:
            if fingerprint.size(0) != g.size(0):
                fingerprint = fingerprint.expand(g.size(0), -1)
            cond = self.conditioner(fingerprint.to(g.device))
            g = torch.cat([g, cond], dim=-1)
        elif self.use_fingerprint:
            pad = torch.zeros(g.size(0), self.cond_dim, device=g.device)
            g = torch.cat([g, pad], dim=-1)

        g = self.fc(g)
        return self.head_part(g), self.head_color(g), self.head_transform(g)
