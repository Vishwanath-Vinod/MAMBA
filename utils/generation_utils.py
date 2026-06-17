from dataclasses import dataclass, field
from typing import Optional
import torch
from torch import Tensor
from transformers.generation import TextStreamer
from utils.sampling_utils import *

@dataclass
class InferenceParams:
    """
    Maintains internal cached states (Mamba SSM states, convolution states) so that previously processed tokens do not need to be recomputed.

    max_seqlen : Maximum sequence length that can be processed during the current inference session.
    max_batch_size : Maximum batch size supported by the currently allocated cache.

    seqlen_offset : Number of tokens that have already been processed and stored in the inference cache.
                  : =0   Prompt processing mode. The entire prompt is passed through the model.
                  : >0   Decoding mode. Only newly generated tokens are passed through the model, while previous context is retrieved from the cache.

    key_value_memory_dict : Stores convolution states and SSM states for each layer.
    """

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)

    def reset(self, max_seqlen, max_batch_size):
        """
        Reset inference state for a new generation session.Does not reallocate caches; only resets bookkeeping variables.
        """
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        # Start from prompt-processing mode.
        self.seqlen_offset = 0

class Generation_Wrapper:
    """ 
    Just a convenience wrapper that adds the additional functionality that model.generate() is enabled.
    Internally it just calls decode(...).
    """
    def generate(self,prompt,max_length,top_k=1,top_p=0.0,min_p=0.0,temperature=1.0,eos_token_id=None,teacher_outputs = None,**kwargs,):
        return decode(prompt, self, max_length, top_k=top_k, top_p=top_p, min_p = min_p, temperature=temperature,
                      eos_token_id=eos_token_id, teacher_outputs= teacher_outputs, **kwargs)

@torch.inference_mode()
def decode(prompt,model,max_length,eos_token_id=None, teacher_outputs=None,vocab_size=None,streamer: Optional[TextStreamer] = None,
           top_k=1,top_p=0.0,min_p=0.0,temperature=1.0,penalty=1.0):
    
    """
    The auto-regressive generation of the model is implemented by this function.

    prompt: (B,prompt_length)
          : Prompt tokens that generation starts from.
    max_length: Maximum possible length of output.
    model : MAMBA Language Model
    top_k, min_p, temperature, penalty : Sampling techniques hyperparameters (see sampling_utils.py)
    eos_token_id: Token indicating end-of-sequence.
                : Generation stops when every sequence in the batch produces token.
    teacher_outputs : (B, length)
                    : Teacher Forcing. If provided, instead of sampling from the logits, the next token is taken from the teacher_outputs. Useful for testing.
    vocab_size: Optional truncation of output logit
    streamer: Real-time token streaming (displays output one by one doesnt wait for whole thing to end).

    RETURN
    sequences: (B, max_length)
    """
    batch_size,_= prompt.shape
    # Allocate memory for inference cache
    inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
    inference_params.key_value_memory_dict = model.allocate_inference_cache(batch_size=batch_size,max_seqlen=max_length,)
    sequences = [prompt]
    sequences_cat = prompt

    if streamer is not None:
        streamer.put(prompt.cpu())

    # Stopping Condition for Generation: if EOS token is encountered or Maxlength of Generation is exceeded.
    def stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all() or inference_params.seqlen_offset>= max_length - 1:
            return True
        return False

    while not stop(sequences[-1], inference_params):
        logits = model(sequences[-1],inference_params=inference_params,num_last_tokens=1,).squeeze(dim=1)
        if vocab_size is not None:
            logits = logits[..., :vocab_size]
        inference_params.seqlen_offset += sequences[-1].shape[1]
        logits = repetition_penalty(logits, sequences_cat, penalty)
        # Checks for teacher Forcing
        if teacher_outputs is None or teacher_outputs.shape[1] <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]

        #Rearranges token from "b -> b 1"
        sampled_tokens = token.unsqueeze(1)
        sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)
        if streamer is not None:
            streamer.put(sampled_tokens.cpu())

    if streamer is not None:
        streamer.end()

    return torch.cat(sequences, dim=1)
