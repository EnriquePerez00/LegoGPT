import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.llm_pipeline.tokenizer_ldr import LdrTokenizer

class BrickLLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, hidden_dim: int = 128, num_layers: int = 2):
        """
        A lightweight autoregressive Transformer Decoder (BrickLLM) 
        trained to generate LDraw syntax.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encodings
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len]
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_emb[:, :seq_len, :]
        
        # Casual/autoregressive mask to prevent looking ahead
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Transformer forward pass (in PyTorch, TransformerEncoder can use is_causal mask)
        h = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.head(h)
        return logits

def train_llm_step(model, optimizer, token_ids: list[int], device="cpu") -> float:
    """
    Performs a single autoregressive training step over a sequence of token IDs.
    """
    device = torch.device(device)
    model.train()
    
    if len(token_ids) < 2:
        return 0.0
        
    x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)
    
    optimizer.zero_grad()
    logits = model(x) # [batch, seq_len, vocab_size]
    
    # Calculate cross entropy loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    
    return loss.item()
