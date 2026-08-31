import torch
from watermark.wm_sample_token import wm_sample_token
from watermark.engine import get_wm_window_hash

class MusicGenWMGen:
    def __init__(
        self,
        model,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 250,
        wm: str = "none",
        wm_ngram: int = 0,
        wm_seed: int = 0,
        wm_streams: list = [0, 1, 2, 3],
        wm_aux_params: dict = None,
    ):
        self.model = model
        self.device = model.device
        self.use_sampling = use_sampling
        self.temp = temp
        self.top_k = top_k
        self.wm = wm
        self.wm_ngram = wm_ngram
        self.wm_seed = wm_seed
        self.wm_streams = wm_streams
        self.wm_aux_params = wm_aux_params or {"delta": 2.0, "gamma": 0.5}
        self.num_codebooks = 4 #model.config.audio_encoder.num_codebooks

    def wm_stream(self, stream_idx: int) -> bool:
        return stream_idx in self.wm_streams

    @torch.no_grad()
    def generate_watermarked(self, inputs, max_new_tokens=256):
        B = inputs["input_ids"].shape[0]
        encoder_outputs = self.model.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None)
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Project hidden states if dimension mismatch
        if hasattr(self.model, "enc_to_dec_proj"):
            encoder_hidden_states = self.model.enc_to_dec_proj(encoder_hidden_states)

        # Handle Start Token
        start_token = self.model.config.decoder_start_token_id or 2048
        if hasattr(self.model.decoder, "embed_tokens"):
            vocab_size = self.model.decoder.embed_tokens[0].num_embeddings
            if start_token >= vocab_size:
                start_token = 0

        # Initialize [B, K, 1]
        next_input_ids = torch.full((B, self.num_codebooks, 1), start_token, dtype=torch.long, device=self.device)
        gen_tokens = next_input_ids
        past_key_values = None

        for step in range(max_new_tokens):
            outputs = self.model.decoder(
                input_ids=next_input_ids, 
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].view(B, self.num_codebooks, -1)
            
            current_step_tokens = []
            for k in range(self.num_codebooks):
                k_logits = logits[:, k, :].unsqueeze(1).unsqueeze(1) # [B, 1, 1, V]

                if self.wm_stream(k):
                    # --- FIX 1: ANCHOR CONTEXT LOGIC ---
                    # To match evaluation, if wm_ngram > 0, we use Stream 0 as the context for EVERYONE.
                    # If wm_ngram == 0, context doesn't matter (history ignored).
                    context_stream_idx = 0 if self.wm_ngram > 0 else k

                    # context_stream_idx = k
                    
                    # 1. Get history from the ANCHOR stream (0) instead of self (k)
                    valid_history = gen_tokens[:, context_stream_idx, 1:]
                    
                    if self.wm_ngram <= 0:
                        ngrams = torch.zeros((B, 0), dtype=torch.long, device=self.device)
                    else:
                        seq_len = valid_history.shape[-1]
                        if seq_len < self.wm_ngram:
                            pad_len = self.wm_ngram - seq_len
                            pad = torch.zeros((B, pad_len), dtype=torch.long, device=self.device)
                            ngrams = torch.cat([pad, valid_history], dim=1)
                        else:
                            ngrams = valid_history[:, -self.wm_ngram:]
                    
                    # 2. Extract the specific map for this stream
                    current_clustering_map = None
                    if "clustering_maps" in self.wm_aux_params and self.wm_aux_params["clustering_maps"]:
                        current_clustering_map = self.wm_aux_params["clustering_maps"].get(k)
                    
                    # 3. Compute Hash (Correctly using the map)
                    window_hash = get_wm_window_hash(ngrams, self.wm_seed, clustering_map=current_clustering_map)
                    
                    # --- FIX 2: PASS MAP TO SAMPLER ---
                    step_params = self.wm_aux_params.copy()
                    if current_clustering_map is not None:
                        step_params["clustering_map"] = current_clustering_map

                    token = wm_sample_token(
                        k_logits, 
                        self.use_sampling, 
                        self.temp, 
                        self.top_k,
                        method=self.wm, 
                        window_hash=window_hash, 
                        aux_params=step_params 
                    )
                else:
                    probs = torch.softmax(k_logits / self.temp, dim=-1)
                    token = torch.multinomial(probs.view(B, -1), 1).view(B, 1, 1)
                
                current_step_tokens.append(token.squeeze(-1))

            next_step_tensor = torch.cat(current_step_tokens, dim=1).unsqueeze(-1)
            gen_tokens = torch.cat([gen_tokens, next_step_tensor], dim=2)
            next_input_ids = next_step_tensor

        # Return tokens EXCLUDING the start token
        return gen_tokens[:, :, 1:]