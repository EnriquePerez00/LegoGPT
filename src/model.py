import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

def get_device() -> str:
    """
    Checks for hardware acceleration options, prioritizing Apple Silicon MPS
    (Metal Performance Shaders) over standard CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class HierarchicalSoftmax(nn.Module):
    def __init__(self, hidden_dim: int, allowed_parts: list[str]):
        super().__init__()
        self.allowed_parts = allowed_parts
        
        # Hardcoded hierarchy based on geometric parsing
        # L1: Brick (Height=24) vs Plate (Height=8)
        # L2: Small/Medium vs Large
        
        # Define lists of parts in each subcategory
        self.bricks_small = [
            "3005.dat", "3004.dat", "3003.dat", "3062b.dat", "3941.dat", 
            "3005p01.dat", "3004p01.dat", "3003p01.dat"
        ]
        self.bricks_large = [
            "3002.dat", "3001.dat", "3010.dat", "3009.dat", "3008.dat", 
            "3007.dat", "3006.dat", "3010p01.dat"
        ]
        self.plates_small = [
            "3024.dat", "3023.dat", "3022.dat", "3021.dat", "3020.dat", 
            "3710.dat", "3666.dat", "3460.dat", "3034.dat", "3832.dat"
        ]
        self.plates_large = [
            "3032.dat", "3031.dat", "3030.dat", "3035.dat", "3036.dat", "3958.dat"
        ]
        
        # Group mappings to indices
        self.part_to_idx = {p: i for i, p in enumerate(allowed_parts)}
        
        # Predict L1 category (2 classes: Brick=0, Plate=1)
        self.l1_classifier = nn.Linear(hidden_dim, 2)
        
        # Predict L2 category within Brick (2 classes: Small=0, Large=1)
        self.l2_brick_classifier = nn.Linear(hidden_dim, 2)
        # Predict L2 category within Plate (2 classes: Small=0, Large=1)
        self.l2_plate_classifier = nn.Linear(hidden_dim, 2)
        
        # Predict individual leaf part
        self.leaf_brick_small = nn.Linear(hidden_dim, len(self.bricks_small))
        self.leaf_brick_large = nn.Linear(hidden_dim, len(self.bricks_large))
        self.leaf_plates_small = nn.Linear(hidden_dim, len(self.plates_small))
        self.leaf_plates_large = nn.Linear(hidden_dim, len(self.plates_large))
        
        # Build tensor index mappings for fast loss computation
        l1_labels = torch.zeros(len(allowed_parts), dtype=torch.long)
        l2_labels = torch.zeros(len(allowed_parts), dtype=torch.long)
        leaf_labels = torch.zeros(len(allowed_parts), dtype=torch.long)
        
        # L1: 0=Brick, 1=Plate
        # L2: 0=Small, 1=Large
        for idx, part in enumerate(allowed_parts):
            if part in self.bricks_small:
                l1_labels[idx] = 0
                l2_labels[idx] = 0
                leaf_labels[idx] = self.bricks_small.index(part)
            elif part in self.bricks_large:
                l1_labels[idx] = 0
                l2_labels[idx] = 1
                leaf_labels[idx] = self.bricks_large.index(part)
            elif part in self.plates_small:
                l1_labels[idx] = 1
                l2_labels[idx] = 0
                leaf_labels[idx] = self.plates_small.index(part)
            elif part in self.plates_large:
                l1_labels[idx] = 1
                l2_labels[idx] = 1
                leaf_labels[idx] = self.plates_large.index(part)

        self.register_buffer('l1_labels', l1_labels)
        self.register_buffer('l2_labels', l2_labels)
        self.register_buffer('leaf_labels', leaf_labels)


    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes full part probabilities over the vocabulary.
        For generation inference, we can compute the path probabilities:
        P(part) = P(L1) * P(L2|L1) * P(leaf|L2, L1)
        """
        l1_logits = self.l1_classifier(h)
        l1_probs = F.softmax(l1_logits, dim=-1) # [batch, 2]
        
        l2_brick_logits = self.l2_brick_classifier(h)
        l2_brick_probs = F.softmax(l2_brick_logits, dim=-1) # [batch, 2]
        
        l2_plate_logits = self.l2_plate_classifier(h)
        l2_plate_probs = F.softmax(l2_plate_logits, dim=-1) # [batch, 2]
        
        leaf_bs_logits = self.leaf_brick_small(h)
        leaf_bs_probs = F.softmax(leaf_bs_logits, dim=-1) # [batch, 8]
        
        leaf_bl_logits = self.leaf_brick_large(h)
        leaf_bl_probs = F.softmax(leaf_bl_logits, dim=-1) # [batch, 8]
        
        leaf_ps_logits = self.leaf_plates_small(h)
        leaf_ps_probs = F.softmax(leaf_ps_logits, dim=-1) # [batch, 10]
        
        leaf_pl_logits = self.leaf_plates_large(h)
        leaf_pl_probs = F.softmax(leaf_pl_logits, dim=-1) # [batch, 6]
        
        # Reconstruct full vocab probabilities
        batch_size = h.size(0)
        full_probs = torch.zeros(batch_size, len(self.allowed_parts), device=h.device)
        
        # Brick -> Small
        p_l1_brick = l1_probs[:, 0:1]
        p_l2_bs = l2_brick_probs[:, 0:1]
        full_probs[:, [self.part_to_idx[p] for p in self.bricks_small]] = p_l1_brick * p_l2_bs * leaf_bs_probs
        
        # Brick -> Large
        p_l2_bl = l2_brick_probs[:, 1:2]
        full_probs[:, [self.part_to_idx[p] for p in self.bricks_large]] = p_l1_brick * p_l2_bl * leaf_bl_probs
        
        # Plate -> Small
        p_l1_plate = l1_probs[:, 1:2]
        p_l2_ps = l2_plate_probs[:, 0:1]
        full_probs[:, [self.part_to_idx[p] for p in self.plates_small]] = p_l1_plate * p_l2_ps * leaf_ps_probs
        
        # Plate -> Large
        p_l2_pl = l2_plate_probs[:, 1:2]
        full_probs[:, [self.part_to_idx[p] for p in self.plates_large]] = p_l1_plate * p_l2_pl * leaf_pl_probs
        
        # Return log probabilities so it functions similarly to logits for Loss/Inference
        return torch.log(full_probs + 1e-12)

    def compute_loss(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the sum of CrossEntropy loss at each level of the tree.
        """
        # Map targets to L1, L2, and leaf categories
        target_l1 = self.l1_labels[targets]
        target_l2 = self.l2_labels[targets]
        target_leaf = self.leaf_labels[targets]
        
        # Level 1 Loss
        l1_logits = self.l1_classifier(h)
        loss = F.cross_entropy(l1_logits, target_l1)
        
        # Level 2 Loss (conditional on target class)
        l2_brick_logits = self.l2_brick_classifier(h)
        l2_plate_logits = self.l2_plate_classifier(h)
        
        # Mask for bricks vs plates in targets
        is_brick = (target_l1 == 0)
        is_plate = (target_l1 == 1)
        
        if is_brick.any():
            loss += F.cross_entropy(l2_brick_logits[is_brick], target_l2[is_brick])
        if is_plate.any():
            loss += F.cross_entropy(l2_plate_logits[is_plate], target_l2[is_plate])
            
        # Leaf Level Loss
        leaf_bs_logits = self.leaf_brick_small(h)
        leaf_bl_logits = self.leaf_brick_large(h)
        leaf_ps_logits = self.leaf_plates_small(h)
        leaf_pl_logits = self.leaf_plates_large(h)
        
        # Further split by L2
        is_bs = is_brick & (target_l2 == 0)
        is_bl = is_brick & (target_l2 == 1)
        is_ps = is_plate & (target_l2 == 0)
        is_pl = is_plate & (target_l2 == 1)
        
        if is_bs.any():
            loss += F.cross_entropy(leaf_bs_logits[is_bs], target_leaf[is_bs])
        if is_bl.any():
            loss += F.cross_entropy(leaf_bl_logits[is_bl], target_leaf[is_bl])
        if is_ps.any():
            loss += F.cross_entropy(leaf_ps_logits[is_ps], target_leaf[is_ps])
        if is_pl.any():
            loss += F.cross_entropy(leaf_pl_logits[is_pl], target_leaf[is_pl])
            
        return loss

class LegoGNN(nn.Module):
    def __init__(self, num_part_classes: int = 32, num_color_classes: int = 16, hidden_dim: int = 64, allowed_parts: list[str] = None):
        """
        A Multi-Task Graph Neural Network that inputs the current Lego assembly graph
        and outputs predictions for the next part ID, color, and translation/rotation.
        """
        super().__init__()
        in_channels = num_part_classes + num_color_classes + 12
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Use HierarchicalSoftmax if allowed_parts is specified
        if allowed_parts is not None:
            self.head_part = HierarchicalSoftmax(hidden_dim, allowed_parts)
        else:
            self.head_part = nn.Linear(hidden_dim, num_part_classes)
            
        self.head_color = nn.Linear(hidden_dim, num_color_classes)
        self.head_transform = nn.Linear(hidden_dim, 12)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        
        g = global_mean_pool(h, batch)
        g = self.fc(g)
        
        # Parallel multi-task prediction heads
        if isinstance(self.head_part, HierarchicalSoftmax):
            part_log_probs = self.head_part(g)
            part_out = part_log_probs # returns log probabilities
        else:
            part_out = self.head_part(g)
            
        color_logits = self.head_color(g)
        transform_preds = self.head_transform(g)
        
        return part_out, color_logits, transform_preds

class LegoGraphTransformer(nn.Module):
    def __init__(self, num_part_classes: int = 32, num_color_classes: int = 16, hidden_dim: int = 64, allowed_parts: list[str] = None):
        """
        A Hybrid Graph-Transformer model that combines GNN spatial message passing
        with sequence self-attention layers to model global, long-range structural dependencies.
        """
        super().__init__()
        in_channels = num_part_classes + num_color_classes + 12
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        # Transformer layer (batch_first=True makes it compatible with dense batch output)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 2, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        if allowed_parts is not None:
            self.head_part = HierarchicalSoftmax(hidden_dim, allowed_parts)
        else:
            self.head_part = nn.Linear(hidden_dim, num_part_classes)
            
        self.head_color = nn.Linear(hidden_dim, num_color_classes)
        self.head_transform = nn.Linear(hidden_dim, 12)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        
        # Convert sparse node features to dense batch format for Transformer self-attention
        h_dense, mask = to_dense_batch(h, batch)
        
        # Apply Transformer self-attention (batch_first is True)
        h_attn = self.transformer(h_dense, src_key_padding_mask=~mask)
        
        # Mask out padding elements and pool to get graph embedding
        h_attn = h_attn * mask.unsqueeze(-1)
        g = h_attn.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        g = self.fc(g)
        
        # Prediction heads
        if isinstance(self.head_part, HierarchicalSoftmax):
            part_log_probs = self.head_part(g)
            part_out = part_log_probs
        else:
            part_out = self.head_part(g)
            
        color_logits = self.head_color(g)
        transform_preds = self.head_transform(g)
        
        return part_out, color_logits, transform_preds

