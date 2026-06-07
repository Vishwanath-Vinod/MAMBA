import torch
import torch.nn as nn
from configs.config import MambaConfig
from model.mamba_model import Model

class MambaLM(nn.Module):
    """
    The simple Language Model obtained by using the MAMBA model defined in create_model.
    """
    def __init__(self,config: MambaConfig,vocab_size = None,device=None,dtype=None,) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        mlp_intermediate = config.mlp_intermediate
        vocab_size = (config.vocab_size if vocab_size is None else vocab_size)
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.backbone = Model(d_model,n_layer,mlp_intermediate,vocab_size,config.dropout,self.config,**factory_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.tie_weights()

    def tie_weights(self):
        """
        Ties the weights of Embedding and Final Logit layer which is conventional in LLMs.
        """
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate and return inference caches for all Mamba layers in all blocks of the Model. Used only in generation (auto-regressive).
        """
        return {i: block.layer.allocate_inference_cache(batch_size,dtype=dtype, **kwargs)for i, block in enumerate(self.backbone.layers)}

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens. Useful in autoregressive generation.
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
