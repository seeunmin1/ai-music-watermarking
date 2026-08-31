# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from models.moshi.utils.sampling import sample_token as sampling_sample_token
from watermark.wm_sampling_methods import gumbel_sample, maryland_sample, alignedis_sample

def wm_sample_token(
    logits: torch.Tensor,  # b 1 1 v
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    method: str = "gumbel",
    window_hash: torch.Tensor = None,  # b
    aux_params: dict = None,
) -> torch.Tensor:
    """Given logits of shape [*, Card], returns a LongTensor of shape [*]."""
    if window_hash is None:
        assert method == "none", f"window_hash is required for {method} sampling"
        
    # Extract map if available
    clustering_map = None
    if aux_params:
        if "clustering_map" in aux_params:
            clustering_map = aux_params["clustering_map"]
        elif "clustering_maps" in aux_params and "stream_id" in aux_params:
            clustering_map = aux_params["clustering_maps"].get(aux_params["stream_id"])

    if method == "gumbel":
        return gumbel_sample(logits, window_hash, use_sampling, temp, top_p, top_k)
    elif method == "maryland":
        gamma = aux_params.get("gamma", 0.5)
        delta = aux_params.get("delta", 1.0)
        return maryland_sample(logits, window_hash, use_sampling, temp, top_k, top_p, gamma, delta, clustering_map)
    elif method == "alignedis":
        return alignedis_sample(logits, window_hash, use_sampling, temp, top_p, top_k, aux_params)
    else:
        return sampling_sample_token(logits, use_sampling, temp, top_k, top_p)
