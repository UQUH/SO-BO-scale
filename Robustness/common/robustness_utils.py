from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_ROOT = REPO_ROOT / "Robustness"
COMMON_ROOT = ROBUSTNESS_ROOT / "common"
SEED_FILE = COMMON_ROOT / "seeds.txt"


@dataclass(frozen=True)
class ExampleConfig:
    name: str
    do_exp: float | None
    k: int | None
    bo_n_initial: int
    bo_n_batch: int
    bo_n_post: int
    bo_max_iter: int


EXAMPLE_CONFIGS: dict[str, ExampleConfig] = {
    "Ex1": ExampleConfig(
        name="Ex1",
        do_exp=3.456255510150231e-04,
        k=8,
        bo_n_initial=40,
        bo_n_batch=10,
        bo_n_post=100,
        bo_max_iter=50,
    ),
    "Ex2": ExampleConfig(
        name="Ex2",
        do_exp=16.1938,
        k=10,
        bo_n_initial=40,
        bo_n_batch=10,
        bo_n_post=100,
        bo_max_iter=50,
    ),
    "Ex3": ExampleConfig(
        name="Ex3",
        do_exp=None,
        k=None,
        bo_n_initial=10,
        bo_n_batch=10,
        bo_n_post=100,
        bo_max_iter=50,
    ),
    "Ex3_sigma03": ExampleConfig(
        name="Ex3_sigma03",
        do_exp=None,
        k=None,
        bo_n_initial=10,
        bo_n_batch=10,
        bo_n_post=100,
        bo_max_iter=50,
    ),
}

# Maps example name to the lowercase key used in native script/file names.
# Ex3_sigma03 scripts are still named bo_ex3_*.py / bo_ex3_*.mat.
EXAMPLE_SCRIPT_KEY: dict[str, str] = {
    "Ex1": "ex1",
    "Ex2": "ex2",
    "Ex3": "ex3",
    "Ex3_sigma03": "ex3",
}

CANONICAL_METHODS = ("1D", "BO", "GP", "SOTA_BO", "SOTA_BO_QNEI")
STOCHASTIC_METHODS = ("BO", "GP", "SOTA_BO", "SOTA_BO_QNEI")
METHOD_LABELS = {
    "1D": "1D",
    "BO": "BO",
    "GP": "Hetero GP",
    "SOTA_BO": "qLogNEI",
    "SOTA_BO_QNEI": "qNEI",
}
METHOD_COLORS = {
    "1D": "#000000",
    "BO": "#1f77b4",
    "GP": "#d62728",
    "SOTA_BO": "#2ca02c",
    "SOTA_BO_QNEI": "#ff7f0e",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_seeds(max_seeds: int | None = None) -> list[int]:
    seeds = np.loadtxt(SEED_FILE, dtype=int).tolist()
    if isinstance(seeds, int):
        seeds = [seeds]
    if max_seeds is not None:
        return seeds[:max_seeds]
    return seeds


def get_example_root(example: str) -> Path:
    return ROBUSTNESS_ROOT / example


def get_example_method_root(example: str, method: str) -> Path:
    return get_example_root(example) / method


def get_truth_path(example: str) -> Path:
    return get_example_root(example) / "truth_data.mat"


def get_native_example_root(example: str) -> Path:
    return REPO_ROOT / example


def load_truth(example: str) -> dict[str, np.ndarray | float]:
    truth_path = get_truth_path(example)
    data = sio.loadmat(truth_path)
    return {
        "beta_grid": np.asarray(data["beta_grid"]).reshape(-1).astype(float),
        "true_mse": np.asarray(data["true_mse"]).reshape(-1).astype(float),
        "beta_true_opt": float(np.asarray(data["beta_true_opt"]).reshape(-1)[0]),
        "mse_true_opt": float(np.asarray(data["mse_true_opt"]).reshape(-1)[0]),
        "do_exp": float(np.asarray(data["do_exp"]).reshape(-1)[0]),
    }


def build_truth_payload(example: str) -> dict[str, np.ndarray | float]:
    cfg = EXAMPLE_CONFIGS[example]
    dist_path = get_native_example_root(example) / "Model" / "dist.mat"
    dist_data = sio.loadmat(dist_path)
    dist = np.asarray(dist_data["dist"], dtype=float)

    if "beta" in dist_data:
        beta_grid = np.asarray(dist_data["beta"]).reshape(-1).astype(float)
    else:
        beta_grid = np.arange(cfg.k, cfg.k + dist.shape[1], dtype=float)

    if "doExp" in dist_data:
        do_exp = float(np.asarray(dist_data["doExp"]).reshape(-1)[0])
    else:
        if cfg.do_exp is None:
            raise ValueError(f"Missing doExp for {example}")
        do_exp = cfg.do_exp

    true_mse = np.mean((dist - do_exp) ** 2, axis=0)
    min_idx = int(np.argmin(true_mse))
    return {
        "beta_grid": beta_grid,
        "true_mse": true_mse,
        "beta_true_opt": float(beta_grid[min_idx]),
        "mse_true_opt": float(true_mse[min_idx]),
        "do_exp": do_exp,
    }


def score_beta(beta_values: Iterable[float] | float, beta_grid: np.ndarray, true_mse: np.ndarray) -> np.ndarray:
    beta_arr = np.asarray(beta_values, dtype=float)
    return np.interp(beta_arr, beta_grid, true_mse)


def cumulative_best(values: Iterable[float]) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def pad_with_last(values: Iterable[float], horizon: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.full(horizon, np.nan)
    if arr.size >= horizon:
        return arr[:horizon]
    pad_value = arr[-1]
    return np.concatenate([arr, np.full(horizon - arr.size, pad_value)])


def method_results_dir(example: str, method: str) -> Path:
    return ensure_dir(get_example_method_root(example, method) / "Results")


def seed_result_path(example: str, method: str, seed: int) -> Path:
    return method_results_dir(example, method) / f"seed_{seed:03d}.mat"


def deterministic_result_path(example: str, method: str) -> Path:
    return method_results_dir(example, method) / "deterministic.mat"


def raw_results_dir(example: str, method: str, seed: int) -> Path:
    return ensure_dir(method_results_dir(example, method) / "raw" / f"seed_{seed:03d}")


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def native_gp_script(example: str) -> Path:
    key = EXAMPLE_SCRIPT_KEY.get(example, example.lower())
    return get_native_example_root(example) / "GP" / f"bo_{key}_hetero.py"


def native_sota_script(example: str) -> Path:
    key = EXAMPLE_SCRIPT_KEY.get(example, example.lower())
    return get_native_example_root(example) / "SOTA_BO" / f"bo_{key}_sota.py"

