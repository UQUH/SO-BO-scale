#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from robustness_utils import (
    CANONICAL_METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    deterministic_result_path,
    get_example_root,
    load_truth,
    method_results_dir,
    pad_with_last,
    write_summary_csv,
)


def load_method_runs(example: str, method: str) -> list[dict[str, np.ndarray | float]]:
    if method == "1D":
        path = deterministic_result_path(example, method)
        return [normalize_loaded_result(sio.loadmat(path))]

    results_dir = method_results_dir(example, method)
    runs = []
    for path in sorted(results_dir.glob("seed_*.mat")):
        runs.append(normalize_loaded_result(sio.loadmat(path)))
    return runs


def normalize_loaded_result(data: dict) -> dict[str, np.ndarray | float]:
    normalized = {}
    for key, value in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "S"}:
            normalized[key] = "".join(arr.reshape(-1).tolist()).strip()
            continue
        if arr.dtype.kind == "O":
            try:
                normalized[key] = "".join(arr.astype(str).reshape(-1).tolist()).strip()
                continue
            except ValueError:
                pass
        if arr.size == 1:
            normalized[key] = float(arr.reshape(-1)[0])
        else:
            normalized[key] = arr.reshape(-1).astype(float)
    return normalized


def aggregate_example(example: str) -> None:
    truth = load_truth(example)
    horizon = 0
    method_runs: dict[str, list[dict[str, np.ndarray | float]]] = {}

    for method in CANONICAL_METHODS:
        runs = load_method_runs(example, method)
        if not runs:
            raise FileNotFoundError(f"No results found for {example} {method}")
        method_runs[method] = runs
        for run in runs:
            horizon = max(horizon, int(run["stop_eval_count"]))

    summary_rows: list[dict[str, object]] = []
    mat_payload: dict[str, object] = {
        "example": example,
        "eval_horizon": horizon,
        "beta_true_opt": truth["beta_true_opt"],
        "mse_true_opt": truth["mse_true_opt"],
        "beta_grid": truth["beta_grid"],
        "true_mse": truth["true_mse"],
    }

    plt.figure(figsize=(8, 6))
    x = np.arange(1, horizon + 1, dtype=float)

    for method in CANONICAL_METHODS:
        runs = method_runs[method]
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        method_slug = method.lower().replace(" ", "_")

        padded_curves = []
        final_beta = []
        final_true_mse = []
        final_best_true_mse = []
        stop_eval_count = []
        total_mc_evaluations = []
        n_mc_per_objective = []

        for run in runs:
            curve = np.asarray(run["best_true_mse_so_far_seq"], dtype=float)
            ratio_curve = curve / truth["mse_true_opt"]
            padded_curves.append(pad_with_last(ratio_curve, horizon))
            final_beta.append(float(run["final_recommended_beta"]))
            final_true_mse.append(float(run["final_recommended_true_mse"]))
            final_best_true_mse.append(float(run["final_best_true_mse"]))
            stop_eval_count.append(float(run["stop_eval_count"]))
            total_mc_evaluations.append(float(run.get("total_mc_evaluations", run["stop_eval_count"])))
            n_mc_per_objective.append(float(run.get("n_mc_per_objective", 1.0)))

        padded_curves = np.vstack(padded_curves)
        if method == "1D":
            curve = padded_curves[0]
            plt.plot(x, curve, color=color, linewidth=2, label=label)
            mat_payload[f"{method_slug}_curve"] = curve
            mat_payload[f"{method_slug}_final_beta"] = final_beta[0]
            mat_payload[f"{method_slug}_final_recommended_true_mse"] = final_true_mse[0]
            mat_payload[f"{method_slug}_final_best_true_mse"] = final_best_true_mse[0]
            mat_payload[f"{method_slug}_stop_eval_count"] = stop_eval_count[0]
            mat_payload[f"{method_slug}_total_mc_evaluations"] = total_mc_evaluations[0]
            mat_payload[f"{method_slug}_n_mc_per_objective"] = n_mc_per_objective[0]
            summary_rows.append(
                {
                    "method": label,
                    "n_runs": 1,
                    "final_beta_mean": final_beta[0],
                    "final_beta_std": 0.0,
                    "final_recommended_true_mse_mean": final_true_mse[0],
                    "final_recommended_true_mse_std": 0.0,
                    "final_best_true_mse_mean": final_best_true_mse[0],
                    "final_best_true_mse_std": 0.0,
                    "stop_eval_count_mean": stop_eval_count[0],
                    "stop_eval_count_std": 0.0,
                    "total_mc_evaluations_mean": total_mc_evaluations[0],
                    "total_mc_evaluations_std": 0.0,
                    "n_mc_per_objective_mean": n_mc_per_objective[0],
                    "n_mc_per_objective_std": 0.0,
                }
            )
            continue

        curve_mean = np.mean(padded_curves, axis=0)
        curve_std = np.std(padded_curves, axis=0, ddof=0)
        plt.plot(x, curve_mean, color=color, linewidth=2, label=label)
        plt.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, color=color, alpha=0.2)

        mat_payload[f"{method_slug}_curve_mean"] = curve_mean
        mat_payload[f"{method_slug}_curve_std"] = curve_std
        mat_payload[f"{method_slug}_final_beta_mean"] = np.mean(final_beta)
        mat_payload[f"{method_slug}_final_beta_std"] = np.std(final_beta, ddof=0)
        mat_payload[f"{method_slug}_final_recommended_true_mse_mean"] = np.mean(final_true_mse)
        mat_payload[f"{method_slug}_final_recommended_true_mse_std"] = np.std(final_true_mse, ddof=0)
        mat_payload[f"{method_slug}_final_best_true_mse_mean"] = np.mean(final_best_true_mse)
        mat_payload[f"{method_slug}_final_best_true_mse_std"] = np.std(final_best_true_mse, ddof=0)
        mat_payload[f"{method_slug}_stop_eval_count_mean"] = np.mean(stop_eval_count)
        mat_payload[f"{method_slug}_stop_eval_count_std"] = np.std(stop_eval_count, ddof=0)
        mat_payload[f"{method_slug}_total_mc_evaluations_mean"] = np.mean(total_mc_evaluations)
        mat_payload[f"{method_slug}_total_mc_evaluations_std"] = np.std(total_mc_evaluations, ddof=0)
        mat_payload[f"{method_slug}_n_mc_per_objective_mean"] = np.mean(n_mc_per_objective)
        mat_payload[f"{method_slug}_n_mc_per_objective_std"] = np.std(n_mc_per_objective, ddof=0)

        summary_rows.append(
            {
                "method": label,
                "n_runs": len(runs),
                "final_beta_mean": np.mean(final_beta),
                "final_beta_std": np.std(final_beta, ddof=0),
                "final_recommended_true_mse_mean": np.mean(final_true_mse),
                "final_recommended_true_mse_std": np.std(final_true_mse, ddof=0),
                "final_best_true_mse_mean": np.mean(final_best_true_mse),
                "final_best_true_mse_std": np.std(final_best_true_mse, ddof=0),
                "stop_eval_count_mean": np.mean(stop_eval_count),
                "stop_eval_count_std": np.std(stop_eval_count, ddof=0),
                "total_mc_evaluations_mean": np.mean(total_mc_evaluations),
                "total_mc_evaluations_std": np.std(total_mc_evaluations, ddof=0),
                "n_mc_per_objective_mean": np.mean(n_mc_per_objective),
                "n_mc_per_objective_std": np.std(n_mc_per_objective, ddof=0),
            }
        )

    plt.xlabel("Objective evaluations")
    plt.ylabel(r"Best true MSE / $f^*$")
    plt.legend(frameon=False)
    plt.tight_layout()

    example_root = get_example_root(example)
    figure_path = example_root / f"robustness_convergence_{example}.pdf"
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

    csv_path = example_root / f"summary_metrics_{example}.csv"
    write_summary_csv(summary_rows, csv_path)

    mat_path = example_root / f"robustness_summary_{example}.mat"
    sio.savemat(mat_path, mat_payload)

    print(f"Saved robustness figure: {figure_path}")
    print(f"Saved robustness CSV: {csv_path}")
    print(f"Saved robustness MAT: {mat_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate robustness runs for one example.")
    parser.add_argument("--example", required=True, choices=["Ex1", "Ex2", "Ex3", "Ex3_sigma03"])
    args = parser.parse_args()
    aggregate_example(args.example)


if __name__ == "__main__":
    main()
