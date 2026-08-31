# Maryland-style watermarking sampling for CosyVoice3
import torch
from models.moshi.utils.sampling import sample_token

GENERATOR = torch.Generator(device="cpu")

def maryland_sample(
    logits: torch.Tensor, 
    window_hash: torch.Tensor,  # shape: (b,)
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    gamma: float = 0.5, 
    delta: float = 1.0,
    clustering_map: torch.Tensor = None,
    speech_token_size: int = None
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    batch_size = logits.shape[0]
    if clustering_map is not None:
        effective_vocab_size = int(clustering_map.max().item()) + 1
    elif speech_token_size is not None:
        effective_vocab_size = speech_token_size
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
    return sample_token(modified_logits, use_sampling, temp, top_k, top_p)
