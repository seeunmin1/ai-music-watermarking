# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from models.moshi.utils.sampling import sample_token as sampling_sample_token

GENERATOR = torch.Generator(device="cpu")

def gumbel_sample(
    logits: torch.Tensor, 
    window_hash: torch.Tensor,  # b
    use_sampling: bool = False,
    temp: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
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
    batch_size = logits.shape[0]
    rps = torch.empty_like(probs)
    for bsz in range(batch_size):
        GENERATOR.manual_seed(window_hash[bsz].item())
        rs = torch.rand(probs[bsz].shape, generator=GENERATOR).to(probs.device)
        if need_remap:
            rs = torch.gather(rs, -1, probs_idx[bsz])
        rps[bsz] = torch.pow(rs, 1/probs[bsz])
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
    vocab_size = logits.shape[-1]
    batch_size = logits.shape[0]
    if clustering_map is not None:
        effective_vocab_size = int(clustering_map.max().item()) + 1
    else:
        effective_vocab_size = vocab_size
    bias = torch.zeros_like(logits)
    for bsz in range(batch_size):
        GENERATOR.manual_seed(window_hash[bsz].item())
        vocab_perm = torch.randperm(effective_vocab_size, generator=GENERATOR).to(logits.device)
        greenlist = vocab_perm[:int(gamma * effective_vocab_size)]
        deltas = torch.zeros(vocab_size, device=logits.device)
        if clustering_map is not None:
            is_green_cluster = torch.zeros(effective_vocab_size, device=logits.device, dtype=torch.bool)
            is_green_cluster[greenlist] = True
            is_green_token = is_green_cluster[clustering_map]
            if is_green_token.shape[0] > vocab_size:
                is_green_token = is_green_token[:vocab_size]
            deltas[is_green_token] = delta
        else:
            deltas[greenlist] = delta
        bias[bsz] = deltas
    modified_logits = logits + bias
    return sampling_sample_token(modified_logits, use_sampling, temp, top_k, top_p)

def alignedis_sample(
    logits: torch.Tensor,                     # shape: (batch, V)
    use_sampling: bool,
    temp: float,
    top_k: int,
    top_p: float,
    window_hash: torch.Tensor,                # kept for parity; not used here
    aux_params: dict,
) -> torch.Tensor:
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
    reweighted_logits = aligned_wp(input_ids, logits)
    return sampling_sample_token(reweighted_logits, use_sampling, temp, top_k, top_p)
