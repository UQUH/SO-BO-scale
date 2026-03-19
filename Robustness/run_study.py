#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import scipy.io as sio

from common.aggregate_example import aggregate_example
from common.robustness_utils import (
    CANONICAL_METHODS,
    EXAMPLE_CONFIGS,
    ROBUSTNESS_ROOT,
    build_truth_payload,
    deterministic_result_path,
    ensure_dir,
    get_truth_path,
    load_seeds,
    seed_result_path,
)


def matlab_quote(value: str) -> str:
    return value.replace("'", "''")


def ensure_truth_data(example: str, rebuild: bool) -> None:
    out_path = get_truth_path(example)
    if out_path.exists() and not rebuild:
        return
    ensure_dir(out_path.parent)
    sio.savemat(out_path, build_truth_payload(example))
    print(f"Saved truth data: {out_path}")


def run_1d(example: str, skip_existing: bool) -> None:
    out_path = deterministic_result_path(example, "1D")
    if skip_existing and out_path.exists():
        print(f"Skipping existing 1D result: {out_path}")
        return
    script = ROBUSTNESS_ROOT / "common" / "run_1d_reference.py"
    subprocess.run([sys.executable, str(script), "--example", example], check=True)


def run_gp(example: str, seed: int, skip_existing: bool) -> None:
    out_path = seed_result_path(example, "GP", seed)
    if skip_existing and out_path.exists():
        print(f"Skipping existing GP result: {out_path}")
        return
    script = ROBUSTNESS_ROOT / "common" / "run_gp_seeded.py"
    subprocess.run(
        [sys.executable, str(script), "--example", example, "--seed", str(seed)],
        check=True,
    )


def run_sota(example: str, seed: int, skip_existing: bool) -> None:
    out_path = seed_result_path(example, "SOTA_BO", seed)
    if skip_existing and out_path.exists():
        print(f"Skipping existing SOTA result: {out_path}")
        return
    script = ROBUSTNESS_ROOT / "common" / "run_sota_qlognei_seeded.py"
    subprocess.run(
        [sys.executable, str(script), "--example", example, "--seed", str(seed)],
        check=True,
    )


def run_bo(example: str, seed: int, skip_existing: bool) -> None:
    out_path = seed_result_path(example, "BO", seed)
    if skip_existing and out_path.exists():
        print(f"Skipping existing BO result: {out_path}")
        return
    ensure_dir(out_path.parent)
    common_root = ROBUSTNESS_ROOT / "common"
    batch_cmd = (
        f"addpath('{matlab_quote(str(common_root))}'); "
        f"run_bo_seeded('{example}', {seed}, '{matlab_quote(str(out_path))}');"
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    subprocess.run(["matlab", "-batch", batch_cmd], check=True, env=env)


def run_example(example: str, methods: list[str], seeds: list[int], skip_existing: bool, rebuild_truth: bool) -> None:
    ensure_truth_data(example, rebuild_truth)
    print(f"\n=== Robustness Study: {example} ===")

    if "1D" in methods:
        run_1d(example, skip_existing)

    for seed in seeds:
        print(f"\n[{example}] Seed {seed}")
        if "BO" in methods:
            run_bo(example, seed, skip_existing)
        if "GP" in methods:
            run_gp(example, seed, skip_existing)
        if "SOTA_BO" in methods:
            run_sota(example, seed, skip_existing)

    aggregate_example(example)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed robustness study for one or more examples.")
    parser.add_argument("--example", choices=["Ex1", "Ex2", "Ex3", "all"], default="all")
    parser.add_argument("--methods", nargs="+", choices=CANONICAL_METHODS, default=list(CANONICAL_METHODS))
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebuild-truth", action="store_true")
    args = parser.parse_args()

    seeds = load_seeds(args.max_seeds)
    examples = list(EXAMPLE_CONFIGS) if args.example == "all" else [args.example]
    for example in examples:
        run_example(example, args.methods, seeds, args.skip_existing, args.rebuild_truth)


if __name__ == "__main__":
    main()
