#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scipy.io as sio

COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from robustness_utils import EXAMPLE_CONFIGS, get_truth_path, build_truth_payload, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic truth-data artifacts for robustness study.")
    parser.add_argument("--example", choices=["Ex1", "Ex2", "Ex3", "Ex3_sigma03", "all"], default="all")
    args = parser.parse_args()

    examples = list(EXAMPLE_CONFIGS) if args.example == "all" else [args.example]
    for example in examples:
        out_path = get_truth_path(example)
        ensure_dir(out_path.parent)
        payload = build_truth_payload(example)
        sio.savemat(out_path, payload)
        print(f"Saved truth data: {out_path}")


if __name__ == "__main__":
    main()
