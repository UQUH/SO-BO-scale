#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from robustness_utils import (
    EXAMPLE_SCRIPT_KEY,
    cumulative_best,
    ensure_dir,
    load_truth,
    raw_results_dir,
    score_beta,
    seed_result_path,
)


def postprocess_gp_seeded(example: str, seed: int) -> None:
    raw_dir = raw_results_dir(example, "GP", seed)
    key = EXAMPLE_SCRIPT_KEY.get(example, example.lower())
    raw_mat = raw_dir / f"bo_{key}.mat"
    if not raw_mat.exists():
        raise FileNotFoundError(f"Missing native GP output: {raw_mat}")
    native = sio.loadmat(raw_mat)
    truth = load_truth(example)

    evaluated_beta_seq = np.asarray(native["beta_samples"]).reshape(-1).astype(float)
    eval_true_mse_seq = score_beta(evaluated_beta_seq, truth["beta_grid"], truth["true_mse"])
    best_true_mse_so_far_seq = cumulative_best(eval_true_mse_seq)
    eval_count_seq = np.arange(1, evaluated_beta_seq.size + 1, dtype=float)

    if "beta_opt_gp_exact" in native:
        final_recommended_beta = float(np.asarray(native["beta_opt_gp_exact"]).reshape(-1)[0])
    else:
        final_recommended_beta = float(np.asarray(native["beta_opt_gp"]).reshape(-1)[0])
    final_recommended_true_mse = float(
        score_beta(final_recommended_beta, truth["beta_grid"], truth["true_mse"])
    )

    best_idx = int(np.argmin(eval_true_mse_seq))
    final_best_evaluated_beta = float(evaluated_beta_seq[best_idx])
    final_best_true_mse = float(eval_true_mse_seq[best_idx])

    if "n_eval" in native:
        stop_eval_count = int(np.asarray(native["n_eval"]).reshape(-1)[0])
    else:
        stop_eval_count = int(evaluated_beta_seq.size)
    if "n_eval_mc" in native:
        total_mc_evaluations = int(np.asarray(native["n_eval_mc"]).reshape(-1)[0])
    else:
        total_mc_evaluations = stop_eval_count
    if "n_mc_var" in native:
        n_mc_per_objective = int(np.asarray(native["n_mc_var"]).reshape(-1)[0])
    else:
        n_mc_per_objective = 1

    out_path = seed_result_path(example, "GP", seed)
    ensure_dir(out_path.parent)
    sio.savemat(
        out_path,
        {
            "seed": seed,
            "method": "GP",
            "example": example,
            "evaluated_beta_seq": evaluated_beta_seq,
            "eval_true_mse_seq": eval_true_mse_seq,
            "best_true_mse_so_far_seq": best_true_mse_so_far_seq,
            "eval_count_seq": eval_count_seq,
            "final_recommended_beta": final_recommended_beta,
            "final_recommended_true_mse": final_recommended_true_mse,
            "final_best_evaluated_beta": final_best_evaluated_beta,
            "final_best_true_mse": final_best_true_mse,
            "stop_eval_count": stop_eval_count,
            "total_mc_evaluations": total_mc_evaluations,
            "n_mc_per_objective": n_mc_per_objective,
            "is_deterministic": False,
            "native_output_path": str(raw_mat),
        },
    )
    print(f"Saved GP robustness result: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess one seeded heteroscedastic GP robustness job.")
    parser.add_argument("--example", required=True, choices=["Ex1", "Ex2", "Ex3", "Ex3_sigma03"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()
    out_path = seed_result_path(args.example, "GP", args.seed)
    if args.skip_if_exists and out_path.exists():
        print(f"Skipping existing GP robustness result: {out_path}")
        return
    postprocess_gp_seeded(args.example, args.seed)


if __name__ == "__main__":
    main()
