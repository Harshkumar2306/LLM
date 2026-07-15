import torch
import torch.nn as nn
from config.gpt_config import GPTConfig


class GPTEmbeddings(nn.Module):
    """
    Handles Token Embeddings.
    With RoPE, we no longer need absolute positional embeddings (wpe).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # Word Token Embeddings (wte)
        # Maps token integers [0, vocab_size-1] to dense vectors of size d_model
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        
        # Regularization applied immediately after combining embeddings
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """
        Forward pass.
        Args:
            idx: Tensor of shape (Batch, Time) containing integer token IDs.
            pos_offset: (Unused with RoPE, kept for compatibility if needed).
        Returns:
            Tensor of shape (Batch, Time, Channels)
        """
        # Extract the dense vectors
        tok_emb = self.wte(idx) # shape: (Batch, Time, Channels)
        
        return self.dropout(tok_emb)
