#!/usr/bin/env python3

import importlib.util
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_MODES = ("base", "clustered", "both")


def has_required_modules(python_bin: str, required_modules):
    command = [
        python_bin,
        "-c",
        (
            "import importlib.util; "
            f"mods={tuple(required_modules)!r}; "
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
            "print('|'.join(missing))"
        ),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return False, list(required_modules)

    missing = [item for item in result.stdout.strip().split("|") if item]
    return len(missing) == 0, missing


def choose_python_bin(requested: str = None, required_modules=()):
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend([
        sys.executable,
        os.environ.get("VIRTUAL_ENV") and str(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"),
        "python",
        "python3",
        "/usr/bin/python3",
        "/usr/bin/python3.11",
    ])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ok, _missing = has_required_modules(candidate, required_modules)
        if ok:
            return candidate

    missing_hint = ", ".join(required_modules)
    raise SystemExit(
        "No Python interpreter with the required runtime packages was found. "
        f"Install the artifact environment first, then rerun the demo. Missing modules: {missing_hint}."
    )


def ensure_prompt_file(prompt_file: str = None, *, subdir: str, filename: str, lines):
    if prompt_file:
        return prompt_file

    prompt_dir = ROOT / "tmp" / subdir
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / filename
    with open(prompt_path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
    return str(Path("tmp") / subdir / filename)


def iter_demo_modes(mode: str):
    if mode == "both":
        return ["base", "clustered"]
    return [mode]


def run_eval_command(eval_command):
    command = list(eval_command)
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def summarize_results(summary_path: Path, *, orig_key: str, rt_key: str):
    import torch

    summary = torch.load(summary_path, map_location="cpu")
    results = summary.get("results", [])
    if not results:
        return None

    orig_pvals = [row[orig_key] for row in results if row.get(orig_key) is not None]
    rt_pvals = [row[rt_key] for row in results if row.get(rt_key) is not None]

    def median(values):
        if not values:
            return None
        s = sorted(values)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    def mean_log10(values):
        safe = [max(value, 1e-300) for value in values]
        return sum(-math.log10(value) for value in safe) / len(safe) if safe else None

    return {
        "n": len(results),
        "orig_median_pval": median(orig_pvals),
        "orig_mean_logp": mean_log10(orig_pvals),
        "rt_median_pval": median(rt_pvals),
        "rt_mean_logp": mean_log10(rt_pvals),
        "rows": results,
        "orig_key": orig_key,
        "rt_key": rt_key,
    }


def print_summary(label: str, data, *, row_rt_key: str):
    if data is None:
        print(f"{label}: no results")
        return

    print(f"\n[{label}]")
    print(f"samples: {data['n']}")
    print(f"median original p-value: {data['orig_median_pval']:.4e}")
    print(f"mean original -log10 p-value: {data['orig_mean_logp']:.4f}")
    if data["rt_median_pval"] is not None:
        print(f"median round-trip p-value: {data['rt_median_pval']:.4e}")
        print(f"mean round-trip -log10 p-value: {data['rt_mean_logp']:.4f}")


def resolve_summary_path(output_dir: Path, *, clustered: bool):
    if not clustered:
        return output_dir / "summary_base.pt"

    candidates = sorted(output_dir.glob("summary_*.pt"))
    non_standard = [path for path in candidates if path.name != "summary_base.pt"]
    if len(non_standard) == 1:
        return non_standard[0]
    if (output_dir / "summary_clustered.pt").exists():
        return output_dir / "summary_clustered.pt"
    raise SystemExit(f"Could not find clustered summary file in {output_dir}.")


def cleanup_demo_tmp(subdir: str):
    """Remove temporary demo files created during the run."""
    tmp_dir = ROOT / "tmp" / subdir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    
    # Remove parent tmp directory if it's now empty
    tmp_parent = ROOT / "tmp"
    if tmp_parent.exists() and not any(tmp_parent.iterdir()):
        tmp_parent.rmdir()


def require_existing_dir(path_value: str, label: str):
    if not path_value:
        raise SystemExit(f"{label} is required for this demo.")
    path = Path(path_value)
    if not path.exists():
        raise SystemExit(
            f"{label} does not exist: {path}. Download the public checkpoints first with scripts/download_models.py."
        )
    return str(path)