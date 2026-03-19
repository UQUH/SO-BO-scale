# SO-BO-scale: Bayesian Optimization under Uncertainty for Training a Scale Parameter in Stochastic Models

> A Bayesian optimization framework for efficiently optimizing a scale parameter in stochastic models.

Optimization under uncertainty is challenging due to noisy function evaluations.

This repository provides an approach that leverages a **statistical surrogate** for the objective function, enabling **analytical evaluation of expectations**. We also derive a **closed-form solution** for optimizing a random acquisition function, significantly reducing the computational cost per iteration.

The code demonstrates the effectiveness of the proposed approach for optimizing the scale parameter of SS-PPCA across static and dynamic problems, and compares against GP-based BO and SOTA noisy-BO baselines.

## Repository Layout

```
Ex1/   — Linear static problem
Ex2/   — Linear dynamics problem
Ex3/   — Non power-law problem
```

Each example folder contains:

- `BO/`       — Proposed Bayesian optimization (MATLAB)
- `GP/`       — GP-based BO baseline (Python + MATLAB plotting)
- `1D/`       — 1D Monte Carlo baseline (MATLAB)
- `SOTA_BO/`  — SOTA noisy-BO baselines: `qLogNEI` and `qNEI` (Python + MATLAB plotting)
- `Model/`    — Precomputed data and model-specific files

Additionally:

- `Robustness/` — Multi-seed robustness study across all methods and examples

## Requirements

### MATLAB
- R2023b or later
- Statistics and Machine Learning Toolbox
- Optimization Toolbox

### Python
- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `gpytorch`
- `botorch`

The code was developed for a Linux environment where `python` and `matlab` are both available on `PATH`.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/UQUH/SO-BO-scale.git
cd SO-BO-scale
```

### Run the full robustness study

```bash
bash Robustness/run_study.sh --example all
```

Results are written to `Robustness/Ex*/` and include per-seed `.mat` files, aggregated summary CSVs, and convergence plots.

### Dry run on one seed

```bash
bash Robustness/run_study.sh --example Ex1 --max-seeds 1 --rebuild-truth
```

## Reproducing Individual Example Runs

### Proposed BO method

Open MATLAB, set the example `BO/` folder as the working directory, and run the corresponding script (e.g., `Ex1_BO.m`).

### GP baseline

```bash
# Example 1
python Ex1/GP/bo_ex1.py
matlab -batch "cd('Ex1/GP'); plot_gp_bo_ex1"

# Example 2
python Ex2/GP/bo_ex2.py
matlab -batch "cd('Ex2/GP'); plot_gp_bo_ex2"

# Example 3
python Ex3/GP/bo_ex3.py
matlab -batch "cd('Ex3/GP'); plot_gp_bo_ex3"
```

### SOTA noisy-BO baselines

`qLogNEI` and `qNEI` wrappers are provided for `1`, `5`, and `10` MC samples per objective evaluation.

Example: `qLogNEI` with 10 MC samples:

```bash
python Ex1/SOTA_BO/bo_ex1_sota_mc10.py
matlab -batch "cd('Ex1/SOTA_BO'); plot_sota_bo_ex1_mc10"
```

For `qNEI`, use the corresponding `bo_ex*_qnei_mc*.py` and `plot_sota_bo_ex*_qnei_mc*.m` files.

## Citation

If you find the SO-BO-scale helpful, please cite the following paper:

```bibtex
@misc{yadav2025SO-BO-scale,
      title={Bayesian Optimization under Uncertainty for Training a Scale Parameter in Stochastic Models},
      author={Akash Yadav and Ruda Zhang},
      year={2025},
      eprint={2510.06439},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.06439},
}
```

## The Team

The SO-BO-scale method was developed by the [Uncertainty Quantification Lab](https://uq.uh.edu/group-members) at the University of Houston.

Primary contributors:
- Akash Yadav
- Ruda Zhang
