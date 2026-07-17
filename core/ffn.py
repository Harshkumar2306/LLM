import torch
import torch.nn as nn
import sys
import os

# Ensure config can be imported if run directly (though usually run via train.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpt_config import GPTConfig


class SwiGLUFFN(nn.Module):
    """
    The SwiGLU Feed-Forward Network (FFN) used in modern LLMs like LLaMA.
    Uses Swish (SiLU) gating mechanism.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # To match the parameter count of standard 4x expansion, 
        # hidden dimension is typically 8/3 of d_model.
        hidden_dim = int(8 * config.d_model / 3)
        
        # We need two separate weight matrices for the up-projection and the gate
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        
        # The activation function is Swish (SiLU in PyTorch)
        self.act = nn.SiLU()
        
        # Project back down to the model dimension
        self.w3 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        
        # Regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # --- Initialization Flag ---
        # We flag this layer so that our custom initialization logic (in the main GPT model)
        # knows to scale its weights by 1/sqrt(2 * n_layers). This prevents variance explosion
        # because this layer writes directly into the residual stream.
        self.w3.RESIDUAL_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Time, Channels)
        Returns:
            Tensor of shape (Batch, Time, Channels)
        """
        # SwiGLU(x) = Swish(xW_1) * (xW_2)
        swish_gate = self.act(self.w1(x))
        linear_val = self.w2(x)
        
        x = swish_gate * linear_val
        
        x = self.w3(x)
        x = self.dropout(x)
        return x
