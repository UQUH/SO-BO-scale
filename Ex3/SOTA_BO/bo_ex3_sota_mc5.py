#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import sys


def main():
    script_path = Path(__file__).with_name("bo_ex3_sota.py")
    env = os.environ.copy()
    env["SOTA_N_MC_EVAL"] = "5"
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)


if __name__ == "__main__":
    main()
