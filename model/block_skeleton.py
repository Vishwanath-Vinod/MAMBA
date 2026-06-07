import torch
from torch import nn, Tensor

class Block_Skeleton(nn.Module):
    """
    Since we are implementing a Barebones MAMBA model, we do away with the fused kernels introduced in the official implementation. 
    Since fused kernels are no longer used we can stick to the transformer Block architecture itself.
    So x -> Norm -> Layer (MAMBA/Attention) -> Add
    """
    def __init__(self, model, mlp, norm1= nn.LayerNorm, norm2 = nn.LayerNorm,dropout = 0.1):
        super().__init__()
        self.norm = norm1
        self.layer = model
        self.norm2 = norm2
        self.mlp = mlp
        self.drop = dropout
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x: Tensor, inference_params=None, **layer_kwargs):
        residual = x
        x = self.norm(x.to(dtype=self.norm.weight.dtype))
        x = self.layer(x, inference_params=inference_params, **layer_kwargs)
        x = self.dropout(x)
        x = x + residual
        if self.mlp is not None:
            residual = x
            x = self.norm2(x.to(dtype=self.norm2.weight.dtype))
            x = self.mlp(x,dropout = self.drop)
            x = x + residual 
        return x
    
