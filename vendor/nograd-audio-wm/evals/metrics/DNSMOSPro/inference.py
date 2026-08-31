import pandas as pd
import numpy as np
import torch
import os
import librosa

import utils


def measure_mos_dir(audio_dir, out_file):
    audio_files = [os.path.join(root, f) for root, _, files in os.walk(audio_dir) for f in files if f.endswith(".flac") or f.endswith(".wav")]

    mean_predictions = []

    for file in audio_files:
        samples, sr = librosa.load(file, sr=None)
        # resample to 16000 if needed
        if sr != 16000:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)

        spec = torch.FloatTensor(utils.stft(samples))
        with torch.no_grad():
            prediction = model(spec[None, None, ...])

        mean = prediction[:, 0].item()
        mean_predictions.append([file, mean])

    # Convert to DataFrame
    df = pd.DataFrame(mean_predictions, columns=["path", "mos"])
    df.to_csv(out_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()


    model = torch.jit.load('runs/NISQA/model_best.pt', map_location=torch.device("cpu"))

    measure_mos_dir(args.dir, args.out)
