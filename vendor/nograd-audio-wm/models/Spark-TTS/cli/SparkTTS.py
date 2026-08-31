# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP

from watermark.engine import get_wm_window_hash, GENERATOR


class SparkTTSWatermarkLogitsProcessor(LogitsProcessor):
    """Bias semantic-token logits using Maryland watermarking per decoding step."""

    def __init__(
        self,
        semantic_to_vocab: dict,
        vocab_to_semantic: dict,
        gamma: float,
        delta: float,
        seed: int,
        ngram: int,
        clustering_map: torch.Tensor = None,
    ):
        super().__init__()
        self.semantic_to_vocab = semantic_to_vocab
        self.vocab_to_semantic = vocab_to_semantic
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.seed = int(seed)
        self.ngram = int(ngram)
        self.semantic_vocab_size = max(semantic_to_vocab.keys()) + 1 if semantic_to_vocab else 0
        self.clustering_map = clustering_map

    def _extract_semantic_history(self, row_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        sem_ids = [self.vocab_to_semantic[int(tok.item())] for tok in row_ids if int(tok.item()) in self.vocab_to_semantic]
        if not sem_ids:
            return torch.zeros((0,), dtype=torch.long, device=device)
        return torch.tensor(sem_ids, dtype=torch.long, device=device)

    def _build_ngram(self, sem_hist: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.ngram <= 0:
            return torch.zeros((1, 0), dtype=torch.long, device=device)
        if sem_hist.numel() >= self.ngram:
            return sem_hist[-self.ngram:].view(1, -1)
        pad = torch.zeros((self.ngram - sem_hist.numel(),), dtype=torch.long, device=device)
        return torch.cat([pad, sem_hist], dim=0).view(1, -1)

    def _green_semantic_mask(self, greenlist: torch.Tensor, device: torch.device) -> torch.Tensor:
        is_green_sem = torch.zeros((self.semantic_vocab_size,), dtype=torch.bool, device=device)
        if self.clustering_map is None:
            valid_green = greenlist[greenlist < self.semantic_vocab_size]
            is_green_sem[valid_green] = True
            return is_green_sem

        cmap = self.clustering_map.to(device).long()
        safe_idx = torch.arange(self.semantic_vocab_size, device=device)
        if cmap.numel() == 0:
            return is_green_sem
        safe_idx = torch.clamp(safe_idx, 0, cmap.numel() - 1)
        sem_clusters = cmap[safe_idx]

        green_clusters = torch.zeros((int(cmap.max().item()) + 1,), dtype=torch.bool, device=device)
        green_clusters[greenlist] = True
        is_green_sem = green_clusters[sem_clusters]
        return is_green_sem

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Debug: Log logits shape and stats
        # print(f"[DEBUG] Logits shape: {scores.shape}, semantic_vocab_size: {self.semantic_vocab_size}")
        # print(f"[DEBUG] Logits min: {scores.min().item()}, max: {scores.max().item()}, nan: {torch.isnan(scores).sum().item()}, inf: {(scores == float('inf')).sum().item()}, -inf: {(scores == float('-inf')).sum().item()}")
        # Check if most logits are -inf or nan
        total_logits = scores.numel()
        n_neg_inf = (scores == float('-inf')).sum().item()
        n_nan = torch.isnan(scores).sum().item()
        # if n_neg_inf > 0.9 * total_logits:
        #     print(f"[DEBUG] More than 90% of logits are -inf: {n_neg_inf}/{total_logits}")
        # if n_nan > 0:
        #     print(f"[DEBUG] Found NaN logits: {n_nan}/{total_logits}")

        if self.semantic_vocab_size <= 0:
            return scores

        device = scores.device
        if self.clustering_map is not None:
            effective_vocab_size = int(self.clustering_map.max().item()) + 1
        else:
            effective_vocab_size = self.semantic_vocab_size

        # # Debug: Print mapping info
        # print(f"[DEBUG] semantic_to_vocab size: {len(self.semantic_to_vocab)}, vocab_to_semantic size: {len(self.vocab_to_semantic)}")
        # print(f"[DEBUG] Example semantic_to_vocab: {list(self.semantic_to_vocab.items())[:10]}")

        for bsz in range(scores.shape[0]):
            sem_hist = self._extract_semantic_history(input_ids[bsz], device)
            ngram = self._build_ngram(sem_hist, device)
            window_hash = get_wm_window_hash(ngram, self.seed, clustering_map=self.clustering_map)

            GENERATOR.manual_seed(int(window_hash[0].item()))
            perm = torch.randperm(effective_vocab_size, generator=GENERATOR)
            greenlist = perm[: int(self.gamma * effective_vocab_size)].to(device)
            is_green_sem = self._green_semantic_mask(greenlist, device)

            if not is_green_sem.any():
                continue

            green_sem_idx = torch.nonzero(is_green_sem, as_tuple=False).squeeze(-1).tolist()
            green_vocab_ids = [self.semantic_to_vocab[idx] for idx in green_sem_idx if idx in self.semantic_to_vocab]
            # Debug: Print green/red split info
            # print(f"[DEBUG] Step bsz={bsz}: green_sem_idx={green_sem_idx[:10]}, green_vocab_ids={green_vocab_ids[:10]}, total green={len(green_vocab_ids)}")
            if green_vocab_ids:
                scores[bsz, green_vocab_ids] += self.delta

        return scores


class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self, model_dir: Path, device: torch.device = torch.device("cuda:0")):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self._initialize_inference()

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

        vocab = self.tokenizer.get_vocab()
        self.semantic_to_vocab = {}
        self.vocab_to_semantic = {}
        pattern = re.compile(r"<\|bicodec_semantic_(\d+)\|>")
        for token_str, vocab_id in vocab.items():
            m = pattern.fullmatch(token_str)
            if m is None:
                continue
            sem_id = int(m.group(1))
            self.semantic_to_vocab[sem_id] = int(vocab_id)
            self.vocab_to_semantic[int(vocab_id)] = sem_id
        self.semantic_vocab_size = max(self.semantic_to_vocab.keys()) + 1 if self.semantic_to_vocab else 0

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 3000,
        wm_enable: bool = False,
        wm_method: str = "maryland",
        wm_gamma: float = 0.25,
        wm_delta: float = 2.0,
        wm_ngram: int = 0,
        wm_seed: int = 0,
        clustering_map: torch.Tensor = None,
        return_semantic_ids: bool = False,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
            wm_enable (bool, optional): Enable semantic-token watermarking.
            wm_method (str, optional): Watermark method. Currently supports "maryland".
            wm_gamma (float, optional): Greenlist ratio for Maryland watermarking.
            wm_delta (float, optional): Logit bias for green semantic tokens.
            wm_ngram (int, optional): N-gram context size for watermark hashing.
            wm_seed (int, optional): Watermark seed.
            clustering_map (torch.Tensor, optional): Token->cluster map for clustered watermarking.
            return_semantic_ids (bool, optional): If True, also return generated semantic token IDs.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        logits_processor = None
        if wm_enable and wm_method == "maryland":
            logits_processor = LogitsProcessorList(
                [
                    SparkTTSWatermarkLogitsProcessor(
                        semantic_to_vocab=self.semantic_to_vocab,
                        vocab_to_semantic=self.vocab_to_semantic,
                        gamma=wm_gamma,
                        delta=wm_delta,
                        seed=wm_seed,
                        ngram=wm_ngram,
                        clustering_map=clustering_map,
                    )
                ]
            )

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            logits_processor=logits_processor,
        )

        # Trim the output tokens to remove the input tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated tokens into text
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        if gender is not None:
            global_token_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )

        if return_semantic_ids:
            return wav, pred_semantic_ids.squeeze(0).to(self.device)
        return wav