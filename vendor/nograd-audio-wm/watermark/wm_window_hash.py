# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def get_wm_window_hash(
    ngrams: torch.Tensor = None, 
    seed: int = 0,
    clustering_map: torch.Tensor = None
) -> torch.Tensor:
    """Get watermarking window hash."""
    batch_size, wm_ngram = ngrams.shape
    if wm_ngram == 0:
        return torch.full((batch_size,), seed, dtype=torch.int64)
    else:
        window_hash = torch.zeros(batch_size, dtype=torch.int64)
        if clustering_map is not None:
            safe_idx = ngrams.to(clustering_map.device).long()
            max_idx = max(0, clustering_map.size(0) - 1)
            safe_idx = torch.clamp(safe_idx, 0, max_idx)
            ngrams = clustering_map[safe_idx]
        else:
            ngrams = ngrams
        for bsz in range(batch_size):
            window_hash[bsz] = torch.randint(0, 2**31 - 1, (1,)).item()
            for ii in range(wm_ngram):
                window_hash[bsz] ^= ngrams[bsz, ii].item()
        return window_hash
