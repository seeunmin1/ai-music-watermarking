#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor,BoolTensor
import time
from typing import Union

from .base import AbstractWatermarkCode, AbstractReweight
import pickle


#TODO: fix the highest number for split_k!
class Aligned_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, shuffle: LongTensor, split_k:BoolTensor):
        self.shuffle = shuffle
        self.split_k = split_k
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: Union[torch.Generator, list[torch.Generator]],
        vocab_size: int,
        split_num: int
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
            split_k = torch.cat([
                torch.randint(low=0,high=split_num,size=(1,),dtype=torch.long,generator=rng[i], device=rng[i].device)
                for i in range(batch_size)
            ],dim=0
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
            split_k = torch.randint(low=0,high=split_num,size=(1,),dtype=torch.long,device=rng.device,generator=rng)
        return cls(shuffle,split_k)


class AlignedIS_Reweight(AbstractReweight):
    watermark_code_type = Aligned_WatermarkCode

    def __init__(self, n: float, seed: int = 0, shuffle: bool = True, model_str: str = "spirit-lm-base-7b"):
        self.n = n
        self.seed = seed
        self.shuffle = shuffle
        self.model_str = model_str

        if "spirit" in model_str.lower():
            # SpiritLM: 501 audio tokens
            self.audio_token_start = 32002  # first token
            self.audio_token_end = 32502    # last token

            clusters_path = "models/embeddings/clusterings/spiritlm_even_clusterings.pkl"
            with open(clusters_path, "rb") as f:
                clusterings = pickle.load(f)

            self.clusters = {
                (token_id + 32002): cluster_id
                for token_id, cluster_id in enumerate(clusterings[self.n][self.seed])
            }
            del clusterings

        elif "speech" in model_str.lower():
            # SpeechGPT: 1000 audio tokens
            self.audio_token_start = 32000  # first token
            self.audio_token_end = 32999    # last token

            clusters_path = "models/embeddings/clusterings/speechgpt_even_clusterings.pkl"
            with open(clusters_path, "rb") as f:
                clusterings = pickle.load(f)

            self.clusters = {
                (token_id + 32000): cluster_id
                for token_id, cluster_id in enumerate(clusterings[self.n][self.seed])
            }
            del clusterings

        elif "moshi" in model_str.lower():
            # Moshi: 2048 audio tokens
            self.audio_token_start = 0
            self.audio_token_end = 2047

            clusters_path = "/ihchomes/milis/AlignedIS-dev/models/embeddings/clusterings/mimi_rvq_*_even_clusterings.pkl"
            with open(clusters_path, "rb") as f:
                clusterings = pickle.load(f)

            # The clusterings must be indexable by n and seed as in the spirit/speech branches:
            self.clusters = {
                (token_id + self.audio_token_start): cluster_id
                for token_id, cluster_id in enumerate(clusterings[self.n][self.seed])
            }
            del clusterings


    def __repr__(self):
        return f"AlignedIS_Reweight(n={self.n}, seed={self.seed}, shuffle={self.shuffle}, model_str='{self.model_str}')"


    def create_splits(self, vocab_size, code, device):
        # Initialize splits as tensor
        splits_tensor = torch.full((self.n, vocab_size), -1, dtype=torch.long, device=device)

        # Assign audio tokens to splits according to clustering
        audio_token_ids = torch.arange(self.audio_token_start, self.audio_token_end + 1, device=device)
        if self.shuffle:
            audio_token_ids = code.shuffle[0, audio_token_ids]

        # Assign shuffled audio tokens to their designated splits
        split_indices = torch.tensor(list(self.clusters.values()), dtype=torch.long, device=device)
        splits_tensor[split_indices, audio_token_ids] = audio_token_ids

        # Create remaining token ids (audio tokens are assumed to be contiguous)
        remaining_vocab_size = vocab_size - audio_token_ids.shape[0]
        remaining_token_ids = torch.cat((
            torch.arange(0, self.audio_token_start, device=device),
            torch.arange(self.audio_token_end + 1, vocab_size, device=device)
        ))
        # Block-based assignment of remaining tokens (like the original method)
        split_sizes = [
            round(remaining_vocab_size * (i + 1) / self.n) - round(remaining_vocab_size * i / self.n)
            for i in range(self.n)
        ]
        unassigned_splits = torch.split(remaining_token_ids, split_sizes)

        # Merge and mask out invalid values (assume all splits have audio tokens)
        mask = splits_tensor != -1
        final_splits = [torch.cat((splits_tensor[i][mask[i]], unassigned_splits[i])) for i in range(self.n)]
        return final_splits


    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:

        def set_nan_to_zero(x):
            x[torch.isnan(x)] = 0
            return x

        # s_ means shuffled using the randomly permuted indices in code.shuffle
        if self.shuffle:
            s_logits = torch.gather(p_logits, -1, code.shuffle)
        else:
            s_logits = p_logits
        s_probs = torch.softmax(s_logits, dim=-1)
        bsz, vocab_size = s_logits.shape

        splits = self.create_splits(vocab_size, code, s_logits.device)

        # # Assert that the audio tokens are in the correct splits according to self.clusters
        # for token_id, split_index in self.clusters.items():
        #     if self.shuffle:
        #         assert code.shuffle[0, token_id] in splits[split_index], f"{token_id} not found in the correct split {split_index}."
        #     else:
        #         assert token_id in splits[split_index], f"{token_id} not found in the correct split {split_index}."

        # Pseudorandomly selected channel to promote
        split_k = code.split_k.to(s_logits.device)

        # split_sums = P_{V_j}
        split_sums = []
        if self.n == vocab_size:
            split_sums = s_probs
        elif vocab_size % self.n == 0:
            split_sums = s_probs.view(bsz, self.n, vocab_size // self.n).sum(dim=-1)
        else:
            for n_idx in range(self.n):
                cur_split = splits[n_idx]
                split_sums.append(s_probs[:, cur_split].sum(dim=-1, keepdim=True))
            split_sums = torch.cat(split_sums, dim=-1) # [bsz, n]

        scales = torch.minimum(self.n * torch.ones_like(split_sums).to(s_probs.device), 1 / split_sums) # [bsz, n]

        # (l * P_{V_j} - 1) / P_{V_j}
        overflow_scales = (self.n * split_sums - 1) / split_sums # [bsz, n]: might be negative or nan
        overflow_scales = set_nan_to_zero(overflow_scales)
        overflow_scales[overflow_scales < 0] = 0 # [bsz, n]

        target_scales = scales[range(bsz), split_k] # [bsz]
        target_sums = split_sums[range(bsz), split_k] # [bsz]

        # 1 - l * P_{V_i}
        remain_sums = 1 - target_scales * target_sums # [bsz]
        overflow_sums = (overflow_scales * split_sums).sum(dim=-1) # [bsz]
        fill_scale = remain_sums / overflow_sums # [bsz]
        fill_scale = set_nan_to_zero(fill_scale) # [bsz]

        # Mask for where the cluster matches the probability channel
        split_mask = torch.arange(0, self.n).to(s_logits.device).view(1, -1).repeat(bsz, 1) == split_k.view(-1, 1).repeat(1, self.n)
        final_scale = torch.where(split_mask, target_scales.view(-1, 1).repeat(1, self.n), fill_scale.view(-1, 1) * overflow_scales) # [bsz, n]

        reweighted_s_probs = torch.zeros_like(s_probs).to(s_logits.device)

        # nan = torch.isnan(final_scale).nonzero().squeeze()
        # if nan.numel() > 0:
        #     print("fill_scale", fill_scale)
        #     print("split_mask", split_mask)
        #     print("final scale", final_scale)
        #     print("split", split_k)

        #     # Check if all splits have at least some audio tokens
        #     for i, split in enumerate(splits):
        #         if not any((token >= self.audio_token_start and token <= self.audio_token_end) for token in split):
        #             print(f"No audio token in split {i}")

        if self.n == vocab_size:
            reweighted_s_probs = final_scale * s_probs
        elif vocab_size % self.n == 0:
            reweighted_s_probs = final_scale.view(bsz, self.n, 1).expand((-1, -1, vocab_size // self.n)).reshape(bsz, vocab_size) * s_probs
        else:
            for n_idx in range(self.n):
                cur_split = splits[n_idx]
                reweighted_s_probs[:, cur_split] = final_scale[:, n_idx].view(-1, 1) * s_probs[:, cur_split]

        reweighted_s_probs[reweighted_s_probs < 0] = 0

        reweighted_s_logits = torch.log(reweighted_s_probs)
        if self.shuffle:
            reweighted_logits = torch.gather(reweighted_s_logits, -1, code.unshuffle)
        else:
            reweighted_logits = reweighted_s_logits
        return reweighted_logits
