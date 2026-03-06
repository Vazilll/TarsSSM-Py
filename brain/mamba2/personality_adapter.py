# personality_adapter.py - stub (module referenced but not yet implemented)
import torch
import torch.nn as nn


class PersonalityAdapter(nn.Module):
    """Stub: PersonalityAdapter — identity transform until fully implemented."""
    def __init__(self, d_model: int = 768, **kwargs):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def adapt(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x
