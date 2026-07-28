# HAA6Br SRCS

Research code for chronological few-shot calibration and the SRCS policy used in the HAA6Br study. The repository contains the historical benchmark, the optimized SRCS experiment, risk-budget sensitivity reconstruction, and locked-result analysis. It contains no monitoring records or generated results.

## Scientific scope

- `SRCS-12` means the pre-specified primary algorithmic negative-transfer budget of 12% (`0.12`). It does not mean 12 actions. The policy evaluates 20 actions: five action families at four shrinkage levels.
- The primary optimization searches budgets `0.08`, `0.10`, `0.12`, and `0.15`; `0.12` is the locked primary operating point.
- The 6% operating point is introduced only by `evaluate_risk_budget_variants.py`. It is a post-hoc exploratory safety operating point selected after the main results were viewed.
- Leave-one-EPA-region-out evaluation is geographic internal-external validation within UCMR4, not independent prospective external validation.
- The United Kingdom analysis is a retrospective stress test because those outcomes had been viewed before this optimization.
- "Risk" and "safety" refer to algorithmic negative transfer. They are not drinking-water health-safety guarantees.

The strict CVaR statistic is the mean of the largest `ceil(0.10 * n_systems)` system regrets. This fixed-count definition avoids dilution when regrets are tied at the quantile boundary.

## Installation

Python 3.11 or 3.12 is recommended. The XGBoost model is configured for an NVIDIA CUDA device. The historical runner also verifies CUDA through PyTorch.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

## Data contract

All experiment runners accept only the cleaned `haa6br_integrated_v1` directory through `--data-package`. Raw downloads, archives, and ad hoc source files are intentionally unsupported. See [DATA.md](DATA.md) for the directory and field contract.

## Run order

Use placeholders for paths outside the repository; no local path is embedded in the code.

```bash
python run_new_experiments.py \
  --data-package <INTEGRATED_V1> \
  --output-dir <HISTORICAL_OUTPUT>

python run_optimized_experiments.py \
  --data-package <INTEGRATED_V1> \
  --output-dir <OPTIMIZED_OUTPUT> \
  --historical-output <HISTORICAL_OUTPUT>

python evaluate_risk_budget_variants.py \
  --data-package <INTEGRATED_V1> \
  --optimized-output <OPTIMIZED_OUTPUT>

python analyze_optimized_results.py \
  --optimized-output <OPTIMIZED_OUTPUT> \
  --historical-output <HISTORICAL_OUTPUT>
```

Add `--skip-uk` to the optimized runner when the retrospective UK table is unavailable. The historical runner reproduces all historical tracks and therefore requires both UK files listed in `DATA.md`. `--prediction-cache-source` may point to a validated, read-only cache, but caches are never distributed by this repository.

For a short diagnostic run, use `--smoke --skip-uk`; it still requires the cleaned US data and a CUDA-capable XGBoost installation. Full reproducibility details and evidence labels are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Outputs

Runners create protocol locks, tables, figures, metadata, and rebuildable caches beneath the selected output directory. These artifacts are ignored by Git. Do not commit row-level predictions or caches unless their data-sharing status has been reviewed separately.

## License and citation

Code is released under the MIT License. Citation metadata are provided in [CITATION.cff](CITATION.cff). Dataset licenses and citations remain separate from this code license.
