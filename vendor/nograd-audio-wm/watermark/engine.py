# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from models.moshi.utils.sampling import sample_token as sampling_sample_token
from watermark.wm_window_hash import get_wm_window_hash
from watermark.wm_sampling_methods import gumbel_sample, maryland_sample, alignedis_sample

GENERATOR = torch.Generator(device="cpu")

def get_wm_window_hash(
    ngrams: torch.Tensor = None, 
    seed: int = 0,
    clustering_map: torch.Tensor = None
) -> torch.Tensor:
    """Get watermarking window hash."""
    # Get the hash of the ngrams
    batch_size, wm_ngram = ngrams.shape
    if wm_ngram == 0:
        return torch.full((batch_size,), seed, dtype=torch.int64)
    else:
        window_hash = torch.zeros(batch_size, dtype=torch.int64)
        
        # Apply clustering map to context if provided (synonyms)
        # CRITICAL: This ensures we hash Cluster IDs, not Token IDs.
        if clustering_map is not None:
            # # Ensure ngrams are within map bounds
            # if ngrams.max() >= clustering_map.size(0):
            #     # This can happen with padding tokens or special tokens. 
            #     # We clamp or handle them to avoid index errors, though usually ngrams should be valid.
            #     # For safety in this context, we can let it crash or clamp. 
            #     # Assuming valid tokens, we proceed.
            #     pass
            # ngrams = clustering_map[ngrams.long()]
            # Move indices to clustering_map device and clamp to valid range to avoid indexing asserts
            safe_idx = ngrams.to(clustering_map.device).long()
            max_idx = max(0, clustering_map.size(0) - 1)
            safe_idx = torch.clamp(safe_idx, 0, max_idx)
            ngrams = clustering_map[safe_idx]
        else:
            ngrams = ngrams

        for bsz in range(batch_size):
            GENERATOR.manual_seed(seed)
            window_hash[bsz] = torch.randint(0, 2**31 - 1, (1,), generator=GENERATOR).item()
            for ii in range(wm_ngram):
                window_hash[bsz] ^= ngrams[bsz, ii].item()
        return window_hash


def gumbel_sample(
    logits: torch.Tensor, 
    window_hash: torch.Tensor,  # b
    use_sampling: bool = False,
    temp: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """Aaronson-style watermarking sampling method."""
    if not (use_sampling and temp > 0.0):
        return torch.argmax(logits, dim=-1)
        
    probs = torch.softmax(logits / temp, dim=-1)
    if top_p > 0.0:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = probs_sort
        need_remap = True
    elif top_k > 0:
        topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
        probs = torch.full_like(probs, 1e-6)
        probs.scatter_(-1, topk_idx, topk_probs)
        probs.div_(probs.sum(dim=-1, keepdim=True))
        need_remap = False
        probs_idx = None
    else:
        need_remap = False
        probs_idx = None
    
    # Create batched random values using different seeds
    batch_size = logits.shape[0]
    rps = torch.empty_like(probs)  # b v
    for bsz in range(batch_size):
        GENERATOR.manual_seed(window_hash[bsz].item())
        rs = torch.rand(probs[bsz].shape, generator=GENERATOR).to(probs.device)
        if need_remap:
            rs = torch.gather(rs, -1, probs_idx[bsz])
        rps[bsz] = torch.pow(rs, 1/probs[bsz])  # v
    
    # Select token per batch
    next_token = torch.argmax(rps, dim=-1)
    if need_remap:
        next_token = torch.stack([probs_idx[b, next_token[b]] for b in range(batch_size)])
    return next_token


def maryland_sample(
    logits: torch.Tensor, 
    window_hash: torch.Tensor,  # shape: (b,)
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    gamma: float = 0.5, 
    delta: float = 1.0,
    clustering_map: torch.Tensor = None
) -> torch.Tensor:
    """Maryland-style watermarking sampling method."""
    vocab_size = logits.shape[-1]
    batch_size = logits.shape[0]

    # Determine effective vocab size
    if clustering_map is not None:
        effective_vocab_size = int(clustering_map.max().item()) + 1
    else:
        effective_vocab_size = vocab_size

    # print("Sampling effective vocab:", effective_vocab_size)

    # Create batch-specific greenlist and bias
    bias = torch.zeros_like(logits)  # b 1 1 v
    for bsz in range(batch_size):
        GENERATOR.manual_seed(window_hash[bsz].item())
        vocab_perm = torch.randperm(effective_vocab_size, generator=GENERATOR).to(logits.device)
        greenlist = vocab_perm[:int(gamma * effective_vocab_size)]
        
        deltas = torch.zeros(vocab_size, device=logits.device)  # v
        
        if clustering_map is not None:
            # Map greenlist (clusters) to tokens
            is_green_cluster = torch.zeros(effective_vocab_size, device=logits.device, dtype=torch.bool)
            is_green_cluster[greenlist] = True
            is_green_token = is_green_cluster[clustering_map]
            
            # --- DEBUG / SAFETY FIX ---
            # Slice the mask if the map (e.g. 2048) is larger than the logits (e.g. 2026).
            # This happens if the codebook size > actual model output card.
            if is_green_token.shape[0] > vocab_size:
                is_green_token = is_green_token[:vocab_size]
            
            # --- ASSERTION / PRINTING ---
            # Calculate the actual ratio of green tokens
            # We want this to be close to gamma (0.5). If clustering is very unbalanced, this might skew.
            if bsz == 0: # Print only for the first element in batch to avoid spam
                actual_green_ratio = is_green_token.float().mean().item()
                # print(f"DEBUG [Stream Cluster]: Green Ratio: {actual_green_ratio:.4f} (Target: {gamma}) | Green Clusters: {len(greenlist)}/{effective_vocab_size}")
                
                # TODO: examine
                # if abs(actual_green_ratio - gamma) > 0.15:
                #     print(f"WARNING: Green skew detected: {actual_green_ratio:.3f} VS expected {gamma}.")

            deltas[is_green_token] = delta
        else:
            deltas[greenlist] = delta
            
        bias[bsz] = deltas  # v --> 1 1 v
    
    # Sample using modified logits
    modified_logits = logits + bias
    # from models.moshi.utils.sampling import sample_token
    return sampling_sample_token(modified_logits, use_sampling, temp, top_k, top_p)


def maryland_score_tok(
    tokens: torch.Tensor, 
    window_hash: torch.Tensor,  # shape: (b,)
    vocab_size: int,
    gamma: float = 0.5,
    clustering_map: torch.Tensor = None
) -> torch.Tensor:
    """Maryland-style watermarking detection method."""
    scores = torch.zeros_like(tokens, dtype=torch.float)  # b
    
    if clustering_map is not None:
        effective_vocab_size = int(clustering_map.max().item()) + 1
    else:
        effective_vocab_size = vocab_size

    for bsz in range(tokens.shape[0]):
        GENERATOR.manual_seed(window_hash[bsz].item())
        vocab_perm = torch.randperm(effective_vocab_size, generator=GENERATOR).to(tokens.device)
        greenlist = vocab_perm[:int(gamma * effective_vocab_size)]
        
        if clustering_map is not None:
            cluster_id = clustering_map[tokens[bsz].long()]
            scores[bsz] = 1.0 if cluster_id in greenlist else 0.0
        else:
            scores[bsz] = 1.0 if tokens[bsz] in greenlist else 0.0
            
    return scores


def alignedis_sample(
    logits: torch.Tensor,                     # shape: (batch, V)
    use_sampling: bool,
    temp: float,
    top_k: int,
    top_p: float,
    window_hash: torch.Tensor,                # kept for parity; not used here
    aux_params: dict,
) -> torch.Tensor:
    """
    Reweight `logits` using an AlignedIS WatermarkLogitsProcessor (required in aux_params),
    then sample using sample_token.

    REQUIRED keys in aux_params:
      - "aligned_wp": an instance of watermarks.transformers.WatermarkLogitsProcessor (callable)
      - "input_ids":  torch.LongTensor shape (batch, seq_len) used by the aligned processor
    """
    if aux_params is None:
        raise RuntimeError("alignedis_sample requires aux_params dict with 'aligned_wp' and 'input_ids'")

    aligned_wp = aux_params.get("aligned_wp", None)
    if aligned_wp is None:
        raise RuntimeError("alignedis_sample requires aux_params['aligned_wp'] to be set to a WatermarkLogitsProcessor instance")

    input_ids = aux_params.get("input_ids", None)
    if input_ids is None:
        raise RuntimeError("alignedis_sample requires aux_params['input_ids'] (torch.LongTensor) to be provided")

    if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long:
        raise TypeError("alignedis_sample expects aux_params['input_ids'] to be a torch.LongTensor")

    input_ids = input_ids.to(logits.device)

    # AlignedIS processor is expected to return reweighted logits of same shape as `logits`.
    reweighted_logits = aligned_wp(input_ids, logits)

    # Use existing sampling utility to pick tokens from reweighted logits
    # from models.moshi.utils.sampling import sample_token
    return sampling_sample_token(reweighted_logits, use_sampling, temp, top_k, top_p)


def alignedis_score_tok(
    tokens: torch.Tensor,               # shape: (batch,)
    input_ids: torch.LongTensor,        # shape: (batch, seq_len)
    aligned_wp,                         # WatermarkLogitsProcessor instance (must support get_cluster_n_res)
    vocab_size: int,
    cur_n: int,
    clustering_map: torch.Tensor,       # 1D LongTensor mapping token_id -> cluster_id
) -> torch.Tensor:
    """
    Score a single-time-step batch of tokens using AlignedIS.

    Returns:
      LongTensor of shape (batch,) with values 0/1 indicating whether each token is 'green'.

    REQUIREMENTS (will raise if not met):
      - aligned_wp must be provided and implement get_cluster_n_res
      - input_ids must be provided (used to derive per-example RNG)
      - clustering_map must be provided (torch.LongTensor)
    """
    if aligned_wp is None:
        raise RuntimeError("alignedis_score_tok requires aligned_wp (WatermarkLogitsProcessor instance)")

    if input_ids is None:
        raise RuntimeError("alignedis_score_tok requires input_ids (torch.LongTensor)")

    if clustering_map is None:
        raise RuntimeError("alignedis_score_tok requires clustering_map (torch.LongTensor)")

    if not hasattr(aligned_wp, "get_cluster_n_res"):
        raise RuntimeError("aligned_wp does not implement get_cluster_n_res required for aligned scoring")

    # Ensure devices and dtypes
    input_ids = input_ids.to(tokens.device)
    clustering_map = clustering_map.to(tokens.device).long()

    # Build cluster dict mapping token_id -> cluster_id as plain Python dict (expected by AlignedIS API)
    cluster_dict = {int(i): int(clustering_map[i].item()) for i in range(clustering_map.numel())}

    # aligned_wp.get_cluster_n_res returns a Python list (0/1) per batch element
    res_list = aligned_wp.get_cluster_n_res(input_ids, vocab_size, tokens, cur_n, cluster_dict)
    res_tensor = torch.tensor(res_list, dtype=torch.long, device=tokens.device)
    return res_tensor


def alignedis_stream_scores(
    wm_stream: torch.Tensor,                 # shape: (T,) or (1, T)
    input_ids: torch.LongTensor,             # shape: (1, seq_len) required
    aligned_wp,                              # WatermarkLogitsProcessor instance (required)
    vocab_size: int,
    cur_n: int,
    clustering_map: torch.Tensor,            # 1D LongTensor mapping token_id -> cluster_id (required)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (green_mask, to_score_mask) for a 1D stream of tokens using AlignedIS.

    Returns:
      - green_mask:  torch.BoolTensor length T marking per-position 'green' membership
      - to_score_mask: torch.BoolTensor length T marking first-occurrence tokens (first-seen)

    REQUIREMENTS (no fallbacks; will raise):
      - aligned_wp present and input_ids provided
      - clustering_map provided
      - wm_stream must be 1D or shape (1, T)
    """
    if aligned_wp is None:
        raise RuntimeError("alignedis_stream_scores requires aligned_wp (WatermarkLogitsProcessor instance)")

    if input_ids is None:
        raise RuntimeError("alignedis_stream_scores requires input_ids (torch.LongTensor)")

    if clustering_map is None:
        raise RuntimeError("alignedis_stream_scores requires clustering_map (torch.LongTensor)")

    # Normalize stream to 1D tensor
    if wm_stream.dim() == 2 and wm_stream.shape[0] == 1:
        s = wm_stream.squeeze(0)
    elif wm_stream.dim() == 1:
        s = wm_stream
    else:
        raise ValueError("wm_stream must be 1D or shape (1, T)")

    T = s.shape[-1]
    device = s.device

    # Ensure input_ids on same device
    input_ids = input_ids.to(device)
    clustering_map = clustering_map.to(device).long()

    seen_tokens = set()
    green_mask = torch.zeros((T,), dtype=torch.bool, device=device)
    to_score_mask = torch.zeros((T,), dtype=torch.bool, device=device)

    # Iterate through positions and compute green via alignedis_score_tok
    for ii in range(T):
        # tokens for batch dimension (batch=1)
        cur_token = torch.tensor([int(s[ii].item())], dtype=torch.long, device=device)
        # score using aligned_wp
        res = alignedis_score_tok(cur_token, input_ids, aligned_wp, vocab_size, cur_n, clustering_map)
        # res is tensor shape (batch,) -> take first element
        green_mask[ii] = bool(int(res[0].item()))

        token_val = int(cur_token[0].item())
        if token_val not in seen_tokens:
            to_score_mask[ii] = True
            seen_tokens.add(token_val)

    return green_mask, to_score_mask


def gumbel_score_tok(
    tokens: torch.Tensor, 
    window_hash: torch.Tensor,  # shape: (b,)
    vocab_size: int,
) -> torch.Tensor:
    """gumbel-style watermarking detection method."""
    scores = torch.zeros_like(tokens)  # b
    for bsz in range(tokens.shape[0]):
        GENERATOR.manual_seed(window_hash[bsz].item())
        rs = torch.rand(vocab_size, generator=GENERATOR) # n
        scores[bsz] = -(1 - rs).log()[tokens[bsz]]
    return scores



