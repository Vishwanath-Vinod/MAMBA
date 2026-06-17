import torch 
"""
This file contains various sampling techniques along with a sample(.) function that utilizes them to extract logits for autoregressive generation.
"""

def topk_filtering(logits, top_k):
    """
    Keeps only the K most likely tokens by setting rest to -inf.
    logits: (B,vocab_size)
    """
    topk_values, _ = torch.topk(logits, top_k) # returns (values, indices)
    kth_largest = topk_values[..., -1:] #(B,vocab_size)
    indices_to_remove = logits < kth_largest
    logits.masked_fill_(indices_to_remove, float("-Inf")) #(B,vocab_size)


def min_prob_filtering(logits, min_p):
    """
    Keeps only logits with atleast min_p probability.
    logits: (B,vocab_size)
    """
    if min_p <= 0.0 or min_p >= 1.0:
        return
    indices_to_remove = logits < min_p
    logits.masked_fill_(indices_to_remove, float("-Inf"))


def cummulative_prob_filtering(logits, top_p):
    """
    Keeps the smallest set (in sorted order) whose cumulative probability exceeds top_p, else sets -inf. Removes irrelevant options.
    logits: (B,vocab_size)
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return
    sorted_logits, sorted_indices = torch.sort(logits, descending=False) # returns (values, indices)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

    # Revert to original indexing from sorted indices.
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits.masked_fill_(indices_to_remove, float("-inf"))


def repetition_penalty(logits, history, penalty=1.0):
    """
    Apply repetition penalty to reduce probability of already-generated tokens. Discourages loops like: the cat sat sat sat.
    logits: (B, vocab_size)  
    history: (B, seq_len)  previously output tokens
    """
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, history)

    # if logit > 0: divide by penalty else: multiply by penalty to reduce probability.
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1,history, score)
    return logits

def sample_from_logits(logits,num_samples = 1):
    """
    Returns sampled index given the logits after all modifications.
    """
    probs = torch.softmax(logits, dim=-1)
    sampled_index = torch.multinomial(probs,num_samples=num_samples).squeeze(-1)
    return sampled_index
        
def sample(logits, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0):
    """
    logits: (B,vocab_size)
    min_p: Minimum probability filtering (Keeps only logits with atleast (min_p*max_logit) probability)
    top_p: Keeps the smallest set (in sorted order) whose cumulative probability exceeds top_p
    top_k:  Keeps only the K most likely tokens.
    temperature: Smooths the sampling process.
    """
    _, vocab_size = logits.shape

    # Greedy Decoding
    if top_k == 1:
        return logits.argmax(dim=-1)
    
    #Safety check
    if top_k >= vocab_size:
        top_k = vocab_size
    
    if top_k > 0:
        logits_top, indices = torch.topk(logits, top_k, dim=-1)
        logits_top /= temperature
        cummulative_prob_filtering(logits_top, top_p)

        # Because logits_top has only top_k logits there is index mismatch, which is corrected by this.
        return indices[torch.arange(indices.shape[0], device=indices.device),sample_from_logits(logits_top)]
    else:
        if min_p > 0.0:
            logits_top = logits.clone()
            max_prob = logits_top[..., 0].item()
            threshold = max_prob * min_p
            min_prob_filtering(logits_top, threshold)
            logits_top /= temperature
            return sample_from_logits(logits_top)
        
        # Clone so that when we modify for top_p we don't change the original logits
        logits_top = logits / temperature
        cummulative_prob_filtering(logits_top, top_p)
        return sample_from_logits(logits_top)
        