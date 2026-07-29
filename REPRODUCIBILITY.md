# Reproducibility Notes

## Evidence order

The intended analysis order is:

1. Run the historical benchmark, if a before/after comparison is needed.
2. Run the optimized SRCS experiment with the primary 12% budget locked in code.
3. Reconstruct the full risk-budget sensitivity table. This is where the post-hoc 4% and 6% modes are introduced.
4. Analyze the locked tables without model retuning.

The analyzer expects the risk-budget reconstruction tables to exist. Running it before step 3 fails explicitly.

## Locked choices

- Random seed: `20260728`.
- Primary risk budget: `0.12`.
- Main sensitivity budgets: `0.08`, `0.10`, `0.12`, `0.15`.
- Exploratory-only budgets: `0.04`, `0.06`.
- Adaptation shift cap: 12 micrograms per liter.
- Strict worst-decile CVaR: mean of the largest `ceil(10% * n_systems)` regrets.
- Candidate set: 20 actions plus the zero-shot fallback.
- Few-shot calibration rounds: `k = 1, 2, 3`.

`SRCS-12` always refers to the 12% algorithmic negative-transfer budget. The exploratory 6% point must not be presented as pre-specified or confirmatory.

## Split integrity

US evaluation holds out one EPA region at a time. Source-model and policy-fitting audit tables check system separation, exclusion of held regions, calibration-only policy features, and invariance of fixed decisions to future outcomes. These checks support geographic internal-external validation within UCMR4; they do not create a new independent cohort.

The UK outcomes had already been viewed in earlier work. The runtime method lock prevents tuning during that particular execution but cannot restore prospective independence. Report the UK analysis as a retrospective stress test.

## Environment

The compatibility ranges are in `requirements.txt`. The public package was smoke-checked with Python 3.12, NumPy 1.26.4, pandas 2.3.3, SciPy 1.13.1, XGBoost 3.2.0, and matplotlib 3.9.2. Full execution additionally needs scikit-learn 1.6 or newer because the protocol uses shuffled `GroupKFold`, plus an XGBoost-compatible NVIDIA CUDA runtime. GPU library and driver versions should be recorded with any published rerun.

CUDA training can exhibit small platform-dependent floating-point differences. Treat the saved protocol, script and input hashes, selected actions, and tolerance-based audits as the reproducibility record rather than expecting byte-identical GPU artifacts across hardware.

The logical paths, byte counts, and SHA256 values for the locked inputs, protocol, metadata, and manuscript-facing outputs are recorded in [PROVENANCE_MANIFEST.json](PROVENANCE_MANIFEST.json). The manifest explicitly distinguishes the unavailable exact executed runner from this public implementation.

## Clean rerun

Use a new output directory for an independent rerun. Do not seed it with old tables. A validated read-only prediction cache can reduce runtime, but the cache manifest and array hashes are checked before reuse. Deleteable caches are rebuildable and must not be included in a source release.

Generated metadata intentionally records logical artifact names rather than private absolute paths. The repository ignores outputs, caches, local environments, and data-like files.

## Interpretation limits

The negative-transfer budget controls an empirical algorithmic error criterion. It is not a toxicological threshold, regulatory compliance statement, uncertainty guarantee for every system, or drinking-water health-safety guarantee. Prediction bands are source-derived empirical intervals, not formal cluster-conformal coverage guarantees.
