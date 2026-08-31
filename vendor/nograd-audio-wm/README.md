# Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio

This repository contains the code for the ICML 2026 paper above. The method is a training-free watermark for autoregressive audio generators. It works at the token level, but detection is performed on a clustered vocabulary built from retokenization confusion, which makes the watermark substantially less sensitive to codec inconsistencies after decode-encode round trips and downstream audio modifications.

The artifact covers three settings used in the paper:

- conversational audio with Moshi using the Mimi codec
- music generation with MusicGen using EnCodec
- text-to-speech extensions with CosyVoice3 and Spark-TTS

The shipped cluster maps in `models/embeddings/` are enough to run the main clustered evaluations directly.

Please don't hesitate to contact me or submit an issue if you have trouble running or reproducing the results!

## Setup

### Install dependencies

```bash
conda create -n wmar_audio python=3.11 -y
conda activate wmar_audio
bash install.sh
```

### Download checkpoints

Download the public checkpoints into `checkpoints/` (defaults) or a custom directory:

```bash
python scripts/download_models.py --all
# or with a custom location:
python scripts/download_models.py --all --checkpoints_dir /path/to/checkpoints
```

Individual model families can be selected with `--targets`:

```bash
python scripts/download_models.py --targets moshi musicgen encodec cosyvoice
```

Some model families require external checkpoints:

- Moshi and Mimi weights are downloaded automatically by the main evaluation scripts when omitted.
- MusicGen and EnCodec are loaded from Hugging Face.
- CosyVoice3 and Spark-TTS require local checkpoint directories passed explicitly to their evaluation scripts.

For WMAR finetuned codec baselines and their reconstruction details, refer to the main [WMAR repository](https://github.com/facebookresearch/wmar). This repository is focused on the gradient-free clustered-vocabulary watermark itself.

### Data

Place your audio and prompt datasets under `data/` in this repo. The directory is gitignored.

## Main scripts

These are the scripts to start from:

- `evals/main_wm.py`: conversational audio watermarking with Moshi
- `evals/main_wm_music.py`: music watermarking with MusicGen
- `evals/wm_cosyvoice.py`: CosyVoice3 extension
- `evals/wm_sparktts.py`: Spark-TTS extension
- `scripts/demo_moshi.py`: small conversational-audio demo that compares the KGW base watermark and the clustered WMAR method
- `scripts/demo_musicgen.py`: small MusicGen demo that compares the KGW base watermark and the clustered WMAR method
- `scripts/demo_cosyvoice.py`: small CosyVoice smoke test that can run the KGW base watermark, the clustered WMAR method, or both
- `scripts/demo_sparktts.py`: small Spark-TTS smoke test that can run the KGW base watermark, the clustered WMAR method, or both
- `scripts/download_models.py`: download helper for public checkpoints used in the artifact
- `reconstruction.py`: encode → decode → re-encode existing audio files to collect (original, round-trip) token pairs
- `retokenization.py`: generate audio with the Moshi LM, then immediately retokenize it (combines LM inference + RCC in one step)
- `clustering.py`: build confusion matrices and Leiden community maps from those token pairs

The repository intentionally excludes the paper-only analysis utilities, old sweep launchers, generated outputs, and unused training code from the top-level release surface.

## Conversational audio: Moshi

The conversational experiments use audio prompts. If you need to create them from text prompts, the helpers are in `scripts/textprompts.py` and `scripts/audioprompts.py`.

The full evaluation command runs one configuration at a time. Without `--wm_clustering true`, it runs the unclustered KGW-style base watermark. With `--wm_clustering true`, it runs the clustered WMAR method.

Clustered watermarking with the shipped selected configuration:

```bash
python -m evals.main_wm \
    --output_dir outputs/moshi_clustered \
    --audio_dir data/audio_prompts \
    --nsamples 100 \
    --batch_size 1 \
    --temperature 1.0 \
    --steps 200 \
    --wm_method maryland \
    --wm_streams 1 2 3 4 \
    --wm_delta 2.0 \
    --wm_ngram 0 \
    --wm_clustering true
```

The default clustering assets for this path are:

- `models/embeddings/mimi_leiden_clusterings_trainonly_allparams.pkl`
- `models/moshi_configs.json`

Quick demo:

```bash
python scripts/demo_moshi.py --nsamples 3 --steps 300 --mode both
```

If `--audio_dir` is omitted, the demo first uses `data/audio_prompts/` from this repository. If that directory is missing, it falls back to `/fs/nexus-scratch/milis/outputs/wmar_audio/beacon_outputs/audio_prompts` when available, otherwise it creates a tiny synthetic prompt. You can also pass `--python-bin /path/to/python` if your runtime environment is not the default shell interpreter. Set `--mode base` to run only KGW, `--mode clustered` to run only WMAR, or `--mode both` to compare them back to back.

## Music generation: MusicGen

MusicGen uses text prompts from a plain text file, one prompt per line.

The full evaluation command runs one configuration at a time. Without `--wm_clustering true`, it runs the unclustered KGW-style base watermark. With `--wm_clustering true`, it runs the clustered WMAR method.

Clustered watermarking with the shipped selected configuration:

```bash
python -m evals.main_wm_music \
    --output_dir outputs/musicgen_clustered \
    --prompt_file data/music_prompts.txt \
    --nsamples 100 \
    --batch_size 1 \
    --steps 256 \
    --wm_method maryland \
    --wm_streams 0 1 2 3 \
    --wm_delta 2.0 \
    --wm_ngram 0 \
    --wm_clustering true \
    --encodec_weight /path/to/finetuned_encodec.pt
```

The default clustering assets for this path are:

- `models/embeddings/encodec_leiden_clusterings_trainonly_allparams.pkl`
- `models/encodec_configs.json`

Quick demo:

```bash
python scripts/demo_musicgen.py --nsamples 3 --steps 300 --mode both
```

If `--prompt_file` is omitted, the demo first uses `data/music_prompts.txt` from this repository. If that file is missing, it creates a tiny text prompt file for a smoke test. Set `--mode base` to run only KGW, `--mode clustered` to run only WMAR, or `--mode both` to compare them back to back.

## TTS extensions

### TTS source code

The repository already includes the required CosyVoice and Spark-TTS source code
under `models/CosyVoice/` and `models/Spark-TTS/`.

There is no extra clone or editable-install step for these two codebases. The
demo and evaluation scripts add these directories to `sys.path`
automatically at import time, and `bash install.sh` installs the additional
runtime packages needed for these inference paths.

### Download TTS checkpoints

Download the model weights with:

```bash
# CosyVoice3
python scripts/download_models.py --targets cosyvoice --checkpoints_dir checkpoints
# Spark-TTS
python scripts/download_models.py --targets sparktts --checkpoints_dir checkpoints
```

This places weights under `checkpoints/cosyvoice/Fun-CosyVoice3-0.5B` and
`checkpoints/sparktts/Spark-TTS-0.5B` respectively.

### Run the TTS evals

The TTS case studies are run through:

- `evals/wm_cosyvoice.py`
- `evals/wm_sparktts.py`

Both scripts require explicit local model directories. Each full evaluation command runs one configuration at a time. Use `--run_mode base` for the KGW base watermark and `--run_mode select` for the clustered WMAR method. In TTS outputs, this clustered run is labeled `selected` in filenames such as `summary_selected.csv` and `summary_selected.pt`. Example:

```bash
python -m evals.wm_cosyvoice \
    --output_dir outputs/cosyvoice \
    --prompt_file data/tts_prompts.txt \
    --cosyvoice_model_dir checkpoints/cosyvoice/Fun-CosyVoice3-0.5B \
    --wm_method maryland \
    --wm_streams 0 \
    --run_mode select

python -m evals.wm_sparktts \
    --output_dir outputs/sparktts \
    --prompt_file data/tts_prompts.txt \
    --sparktts_model_dir checkpoints/sparktts/Spark-TTS-0.5B \
    --wm_method maryland \
    --wm_streams 0 \
    --run_mode select
```

For the corresponding base runs, use the same commands with `--run_mode base`.

Quick demos:

```bash
python scripts/demo_cosyvoice.py --nsamples 3 --mode both
python scripts/demo_sparktts.py --nsamples 3 --mode both
```

These two smoke tests expect local checkpoint directories under `checkpoints/cosyvoice/` and `checkpoints/sparktts/`. If those are missing, the demo scripts exit immediately with a short message pointing back to `scripts/download_models.py`. Both demos run directly after the single `bash install.sh` setup above. The Spark-TTS demo also creates a tiny prompt waveform and prompt text automatically when they are omitted. Set `--mode base` to run only KGW, `--mode clustered` to run only WMAR, or `--mode both` to compare them back to back.

All four demo scripts print per-sample watermark detection statistics at the end of the run.

## Attacks

To evaluate watermark robustness under augmentations, add `--eval_aug true` to any of the main eval scripts.

## RCC and clustering

The precomputed clustering pickles in `models/embeddings/` are the defaults used
by all main evaluation scripts — **you do not need to rebuild them to run the
watermarking evaluations**.

To rebuild cluster maps from new audio data, follow the three steps below.

### Step 1 — Retokenization (RCC token pairs)

Encode a set of audio files with the codec, decode back to waveform, and
re-encode to collect (original token, round-trip token) pairs:

```bash
# Mimi (for moshi) or EnCodec — operates on existing audio files
python reconstruction.py \
    --audio_dir /path/to/audio \
    --output_dir outputs/rcc/mimi

# EnCodec variant
python reconstruction.py \
    --audio_dir /path/to/audio \
    --output_dir outputs/rcc/encodec \
    --encodec

# CosyVoice or Spark-TTS — generate audio first, then retokenize
# (run from the TTS eval scripts with --save_tokens; see evals/wm_cosyvoice.py --help)
```

Each sample is saved as a subdirectory containing `orig_tokens.txt` and
`roundtrip_tokens.txt` (one line per codec channel, space-separated token ids).

### Step 2 — Confusion matrix

`clustering.py` reads the token-pair files and accumulates a
`(vocab_size × vocab_size)` confusion matrix that records how often token `i`
round-trips to token `j`.

For CosyVoice and Spark-TTS this step is handled automatically inside
`clustering.py` when you point `--samples_train_dir` at the Step 1 output.

For Mimi and EnCodec, pass the directory that contains the pre-built
`confusion_trainonly_{ch}.npy` files via `--matrix_dir` (or place them in the
default path `outputs/confusion/matrices_new` for Mimi,
`outputs/confusion/matrices_encodec` for EnCodec).

### Step 3 — Community detection (Leiden)

`clustering.py` constructs a directed weighted graph from the confusion
matrix (edges are filtered by a minimum co-occurrence count) and runs the Leiden
algorithm to find vocabulary communities. The result is saved as a `.pkl` file
indexed by `(min_count, resolution)` hyperparameter pairs:

```bash
# Mimi — expects .npy matrix files in outputs/confusion/matrices_new/
python clustering.py --model mimi

# EnCodec — expects .npy matrix files in outputs/confusion/matrices_encodec/
python clustering.py --model encodec

# CosyVoice — builds matrix from token-pair files, then clusters
python clustering.py --model cosyvoice \
    --samples_train_dir outputs/rcc/cosyvoice

# Spark-TTS
python clustering.py --model sparktts \
    --samples_train_dir outputs/rcc/sparktts
```

Optional overrides: `--matrix_dir`, `--output_pkl`, `--counts` (comma-separated
min-count thresholds), `--resolutions` (comma-separated Leiden resolution values).


## Acknowledgements

This repository is adapted from Meta's [WMAR repository](https://github.com/facebookresearch/wmar), which contains the finetuned codec baselines and the broader watermarking framework. The gradient-free clustered-vocabulary method in this repository builds on the codec infrastructure developed there.

## Citation

If you use this code or find the method interesting in your research, please cite:

```bibtex
@inproceedings{milis2026hidden,
  title={Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio},
  author={Milis, Georgios and Qin, Yubin and Wu, Yihan and Huang, Heng},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```
