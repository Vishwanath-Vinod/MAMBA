import torch
import math
import copy
from torch import nn, Tensor
from model.mamba_layer import Mamba
from model.ffn_mlp import MLP
from model.block_skeleton import Block_Skeleton

class Model(nn.Module):
    """
    A minimal Mamba language-model backbone composed of an embedding layer,a stack of pre-norm Mamba blocks, and a final LayerNorm.

    Each block follows the standard Transformer-style pre-norm residual architecture:
    x -> LayerNorm -> Mamba -> Add  and x -> LayerNorm -> MLP   -> Add  (optional)
    """
    def __init__(self,d_model: int,n_layer: int,mlp_intermediate: int,vocab_size: int,
                 dropout : float = 0.1,ssm_cfg=None,norm_epsilon: float = 1e-5,device=None,dtype=None,):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.config = ssm_cfg
        self.n_layer = n_layer
        self.mlp_intermediate = mlp_intermediate
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, d_model, **self.factory_kwargs)
        self.layers = nn.ModuleList([self.create_block(i) for i in range(n_layer)])
        self.final_norm = nn.LayerNorm(d_model, eps=norm_epsilon, **self.factory_kwargs)

        # Equivalent to "for module in self.modules(): self._init_weights(module)." But works recursively.
        self.apply(self._init_weights)

    def forward(self, input_ids, inference_params=None, **kwargs):
        """
        Embed the input tokens and propagate them through the stack of Mamba blocks, followed by a final LayerNorm.
        Returns the final hidden representations for each token.
        """
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params, **kwargs)
        hidden_states = self.final_norm(hidden_states.to(dtype=self.final_norm.weight.dtype))
        return hidden_states
    
    def create_block(self,layer_idx=None):
        """
        Construct a single Mamba block consisting of a Mamba layer, LayerNorm, and an optional MLP branch.
        The block follows the standard pre-norm residual structure and is parameterized by its layer index, which is used during inference cache management.
        """
        config = copy.deepcopy(self.config or {})  # Create a copy of the config to modify
        mamba = Mamba(**config.mamba_kwargs,**self.factory_kwargs,layer_idx=layer_idx,)
        norm = nn.LayerNorm(self.d_model,**self.factory_kwargs)

        if self.mlp_intermediate == 0:
            mlp = None
            norm2 = None
        else:
            mlp = MLP(in_features=self.d_model,hidden_features=self.mlp_intermediate,out_features=self.d_model,**self.factory_kwargs,)
            norm2 = nn.LayerNorm(self.d_model,**self.factory_kwargs,)
            
        block = Block_Skeleton(model= mamba,norm1=norm,mlp=mlp,norm2=norm2,dropout=self.dropout)
        block.layer_idx = layer_idx
        return block
    
    def _init_weights(self,module,initializer_range=0.02):
        """
        Apply GPT-2 style residual scaling initialization. (adopted by Megatron-LM for improved stability when training deep residual networks)
        To stabilize deep pre-norm residual networks, the final projection of each residual branch (e.g. Mamba's out_proj and MLP's fc2) is reinitialized and
        scaled by 1 / sqrt(n_layer * n_residuals_per_layer), preventing residual updates from growing with model depth.
        """

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        #For a Mamba-only block: n_residuals_per_layer = 1. For a Mamba + MLP block: n_residuals_per_layer = 2
        n_residuals_per_layer = 1 if self.mlp_intermediate==0 else 2 
        
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * self.n_layer)
