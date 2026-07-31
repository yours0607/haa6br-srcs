# HAA6Br SRCS

Research code for chronological few-shot calibration and the SRCS policy used in the HAA6Br study. The repository contains the historical benchmark, the optimized SRCS experiment, risk-budget sensitivity reconstruction, locked-result analysis, post-hoc Stage 4 robustness and mechanism analyses, final directional/decision-overlap audits, and the MATLAB submission-figure builder. It contains no monitoring records, row-level predictions, caches, or generated results.

## Scientific scope

- `SRCS-12` means the pre-specified primary algorithmic negative-transfer budget of 12% (`0.12`). It does not mean 12 actions. The policy evaluates 20 actions: five action families at four shrinkage levels.
- The primary optimization searches budgets `0.08`, `0.10`, `0.12`, and `0.15`; `0.12` is the locked primary operating point.
- The 6% operating point is introduced only by `evaluate_risk_budget_variants.py`. It is a post-hoc exploratory prediction-harm operating point selected after the main results were viewed.
- Leave-one-EPA-region-out evaluation is geographic internal-external validation within UCMR4, not independent prospective external validation.
- The United Kingdom analysis is a retrospective stress test because those outcomes had been viewed before this optimization.
- "Risk" and "safety" refer to algorithmic negative transfer. They are not drinking-water health-safety guarantees.
- All `stage4_*` analyses are post-hoc, non-confirmatory revision sensitivities. They must not be used to claim accuracy superiority, equivalence, non-inferiority, prospective transport, or health safety.

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

python stage4_revision_analyses.py \
  --optimized-output <OPTIMIZED_OUTPUT> \
  --data-package <INTEGRATED_V1> \
  --output-dir <STAGE4_OUTPUT> \
  --bootstrap 5000

python stage4_selector_feature_ablation.py \
  --locked-root <OPTIMIZED_OUTPUT> \
  --data-package <INTEGRATED_V1> \
  --output-root <STAGE4_OUTPUT>/selector_feature_ablation

python stage4_mechanism_attribution.py \
  --optimized-output <OPTIMIZED_OUTPUT> \
  --data-package <INTEGRATED_V1> \
  --output-dir <STAGE4_OUTPUT>/mechanism_attribution

python stage4_reviewer2_analyses.py \
  --optimized-output <OPTIMIZED_OUTPUT> \
  --data-package <INTEGRATED_V1> \
  --mechanism-output <STAGE4_OUTPUT>/mechanism_attribution \
  --output-dir <STAGE4_OUTPUT>/reviewer2_analyses

python final_reviewer_revision_analyses.py \
  --predictions <OPTIMIZED_OUTPUT>/tables/us_predictions.csv \
  --coverage-outcomes <STAGE4_OUTPUT>/reviewer2_analyses/analysis/coverage_matched_system_outcomes.csv \
  --risk-systems <OPTIMIZED_OUTPUT>/analysis/risk_budget_system_results.csv \
  --risk-coverage <STAGE4_OUTPUT>/reviewer2_analyses/analysis/risk_coverage_figure_data.csv \
  --output-dir <FINAL_REVIEW_OUTPUT>
```

`build_stage4_figures_matlab.m` is the exact MATLAB R2023a submission-figure builder. It consumes the aggregate `source_data/` tables distributed with the manuscript figure package; those generated tables are not duplicated in this code-only repository.

Add `--skip-uk` to the optimized runner when the retrospective UK table is unavailable. The historical runner reproduces all historical tracks and therefore requires both UK files listed in `DATA.md`. `--prediction-cache-source` may point to a validated, read-only cache, but caches are never distributed by this repository. The selector-ablation, mechanism, and reviewer-audit scripts require the locked strict-v4 result tables or candidate-cache records named above. Without those records they can be inspected and tested, but the complete manuscript analysis cannot be replayed from this source repository alone.

For a short diagnostic run, use `--smoke --skip-uk`; it still requires the cleaned US data and a CUDA-capable XGBoost installation. Full reproducibility details and evidence labels are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Integrity and provenance

[PROVENANCE_MANIFEST.json](PROVENANCE_MANIFEST.json) records logical input, protocol-lock, run-metadata, and manuscript-facing output hashes. The referenced monitoring data and locked results are not distributed by this code repository. The manifest also records that the exact strict-v4 executed runner is unavailable here, so this release is a public implementation rather than a byte-identical archive of the locked execution.

## Outputs

Runners create protocol locks, tables, figures, metadata, and rebuildable caches beneath the selected output directory. These artifacts are ignored by Git. Do not commit row-level predictions or caches unless their data-sharing status has been reviewed separately.

## License and citation

Code is released under the MIT License. Citation metadata are provided in [CITATION.cff](CITATION.cff). Dataset licenses and citations remain separate from this code license.
