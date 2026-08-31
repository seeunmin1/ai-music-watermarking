import argparse
import os
import pickle
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_COUNTS = [1, 5, 10]
DEFAULT_RESOLUTIONS = [0.2, 0.5, 0.8, 1.0, 1.2]

MODEL_CONFIGS = {
    "mimi": {
        "mode": "matrix_only",
        "matrix_dir": ROOT / "outputs" / "confusion" / "matrices_new",
        "clusters_pkl": ROOT / "models" / "embeddings" / "mimi_leiden_clusterings_trainonly_allparams.pkl",
        "channel_names": [
            "rvq_first_0",
            "rvq_rest_0",
            "rvq_rest_1",
            "rvq_rest_2",
            "rvq_rest_3",
            "rvq_rest_4",
            "rvq_rest_5",
            "rvq_rest_6",
        ],
        "counts": [1, 5, 10, 25],
        "resolutions": [0.2, 0.5, 0.8, 1.0, 1.2, 1.5],
    },
    "encodec": {
        "mode": "matrix_only",
        "matrix_dir": ROOT / "outputs" / "confusion" / "matrices_encodec",
        "clusters_pkl": ROOT / "models" / "embeddings" / "encodec_leiden_clusterings_trainonly_allparams.pkl",
        "channel_names": [0, 1, 2, 3],
        "counts": [1, 5, 10],
        "resolutions": [0.2, 0.5, 0.8, 1.0, 1.2],
    },
    "cosyvoice": {
        "mode": "single_channel_samples",
        "vocab_size": 6561,
        "samples_train_dir": ROOT / "nexus_outputs" / "rcc_reconstructions" / "original_cosyvoice" / "samples",
        "matrix_dir": ROOT / "nexus_outputs" / "confusion" / "matrices_cosyvoice",
        "clusters_pkl": ROOT / "models" / "embeddings" / "clusterings_cosyvoice" / "leiden_clusterings_trainonly_allparams.pkl",
        "channel_names": [0],
        "counts": DEFAULT_COUNTS,
        "resolutions": DEFAULT_RESOLUTIONS,
    },
    "sparktts": {
        "mode": "single_channel_samples",
        "vocab_size": 8192,
        "samples_train_dir": ROOT / "nexus_outputs" / "rcc_reconstructions" / "original_sparktts" / "samples",
        "matrix_dir": ROOT / "nexus_outputs" / "confusion" / "matrices_sparktts",
        "clusters_pkl": ROOT / "models" / "embeddings" / "clusterings_sparktts" / "leiden_clusterings_trainonly_allparams.pkl",
        "channel_names": [0],
        "counts": DEFAULT_COUNTS,
        "resolutions": DEFAULT_RESOLUTIONS,
    },
}


def parse_float_list(value: str):
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value: str):
    return [int(item) for item in value.split(",") if item]


def load_confusion_matrix(matrix_dir: Path, channel_idx: int):
    candidates = [
        matrix_dir / f"confusion_trainonly_{channel_idx}.npy",
        matrix_dir / f"confusion_{channel_idx}.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path), path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Confusion matrix not found for channel {channel_idx}: tried {tried}")



def build_confusion_from_samples(train_dirs, *, num_channels: int, vocab_size: int):
    conf_matrices = np.zeros((num_channels, vocab_size, vocab_size), dtype=np.int64)
    usage_counts = np.zeros((num_channels, vocab_size), dtype=np.int64)

    for tdir in train_dirs:
        if not os.path.exists(tdir):
            print(f"Warning: train dir {tdir} not found, skipping.")
            continue
        for name in sorted(os.listdir(tdir)):
            sample_path = os.path.join(tdir, name)
            if not os.path.isdir(sample_path):
                continue

            orig_path = os.path.join(sample_path, "orig_tokens.txt")
            rt_path = os.path.join(sample_path, "roundtrip_tokens.txt")
            if not (os.path.exists(orig_path) and os.path.exists(rt_path)):
                continue

            with open(orig_path, encoding="utf-8") as handle:
                orig = np.array([[int(x) for x in line.split()] for line in handle if line.strip()], dtype=np.int64)
            with open(rt_path, encoding="utf-8") as handle:
                rt = np.array([[int(x) for x in line.split()] for line in handle if line.strip()], dtype=np.int64)

            if orig.shape != rt.shape:
                min_len = min(orig.shape[1], rt.shape[1])
                orig = orig[:, :min_len]
                rt = rt[:, :min_len]

            for channel_idx in range(min(num_channels, orig.shape[0])):
                row_orig, row_rt = orig[channel_idx], rt[channel_idx]
                np.add.at(usage_counts[channel_idx], row_orig, 1)
                mask = row_orig != row_rt
                if np.any(mask):
                    np.add.at(conf_matrices[channel_idx], (row_orig[mask], row_rt[mask]), 1)

    return conf_matrices, usage_counts



def save_single_channel_confusion_if_needed(config, matrix_dir: Path, *, vocab_size: int, samples_train_dir: Path):
    matrix_dir.mkdir(parents=True, exist_ok=True)
    try:
        matrix, path = load_confusion_matrix(matrix_dir, 0)
        print(f"Found confusion matrix at {path}")
        return matrix
    except FileNotFoundError:
        print("Confusion matrix not found; building...")

    conf_matrices, _ = build_confusion_from_samples(
        [str(samples_train_dir)],
        num_channels=1,
        vocab_size=vocab_size,
    )
    output_path = matrix_dir / "confusion_trainonly_0.npy"
    np.save(output_path, conf_matrices[0])
    print(f"Saved confusion matrix to {output_path}")
    return conf_matrices[0]



def build_clusterings_from_matrices(*, channel_names, counts, resolutions, matrix_loader):
    clusterings = {}

    for channel_idx, channel_name in enumerate(channel_names):
        matrix = matrix_loader(channel_idx)
        per_channel = {}
        print(f"channel={channel_name} shape={matrix.shape}")

        for cnt in counts:
            masked = np.where(matrix >= cnt, matrix, 0)
            graph = ig.Graph.Weighted_Adjacency(masked.tolist(), mode="directed")
            for res in resolutions:
                partition = leidenalg.find_partition(
                    graph,
                    leidenalg.RBConfigurationVertexPartition,
                    weights=graph.es["weight"],
                    resolution_parameter=res,
                    seed=27,
                )
                labels = np.array(partition.membership)
                _, labels = np.unique(labels, return_inverse=True)
                per_channel[(cnt, res)] = labels

        clusterings[channel_name] = per_channel

    return clusterings



def build_for_model(model: str, *, matrix_dir=None, output_pkl=None, samples_train_dir=None, vocab_size=None, counts=None, resolutions=None):
    config = MODEL_CONFIGS[model]
    matrix_dir = Path(matrix_dir) if matrix_dir else config["matrix_dir"]
    output_pkl = Path(output_pkl) if output_pkl else config["clusters_pkl"]
    counts = counts or config["counts"]
    resolutions = resolutions or config["resolutions"]

    if config["mode"] == "single_channel_samples":
        samples_train_dir = Path(samples_train_dir) if samples_train_dir else config["samples_train_dir"]
        vocab_size = vocab_size or config["vocab_size"]
        single_matrix = save_single_channel_confusion_if_needed(
            config,
            matrix_dir,
            vocab_size=vocab_size,
            samples_train_dir=samples_train_dir,
        )
        clusterings = build_clusterings_from_matrices(
            channel_names=config["channel_names"],
            counts=counts,
            resolutions=resolutions,
            matrix_loader=lambda _channel_idx: single_matrix,
        )
    else:
        clusterings = build_clusterings_from_matrices(
            channel_names=config["channel_names"],
            counts=counts,
            resolutions=resolutions,
            matrix_loader=lambda channel_idx: load_confusion_matrix(matrix_dir, channel_idx)[0],
        )

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as handle:
        pickle.dump(clusterings, handle)

    print(f"Saved clusterings to {output_pkl}")



def build_parser(description="Build clustering pickles from confusion matrices or RCC token reconstructions."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument("--matrix_dir", type=str, default=None)
    parser.add_argument("--output_pkl", type=str, default=None)
    parser.add_argument("--samples_train_dir", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument(
        "--counts",
        type=parse_int_list,
        default=None,
        help="Comma-separated minimum-count thresholds. Defaults depend on the selected model.",
    )
    parser.add_argument(
        "--resolutions",
        type=parse_float_list,
        default=None,
        help="Comma-separated Leiden resolutions. Defaults depend on the selected model.",
    )
    return parser



def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    build_for_model(
        args.model,
        matrix_dir=args.matrix_dir,
        output_pkl=args.output_pkl,
        samples_train_dir=args.samples_train_dir,
        vocab_size=args.vocab_size,
        counts=args.counts,
        resolutions=args.resolutions,
    )


if __name__ == "__main__":
    main()
