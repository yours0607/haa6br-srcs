# Reproducibility Notes

## Evidence order

The intended analysis order is:

1. Run the historical benchmark, if a before/after comparison is needed.
2. Run the optimized SRCS experiment with the primary 12% budget locked in code.
3. Reconstruct the full risk-budget sensitivity table. This is where the post-hoc 4% and 6% modes are introduced.
4. Analyze the locked tables without model retuning.
5. Run `stage4_revision_analyses.py` on the locked predictions with 5,000 paired region/system bootstrap replicates.
6. Run `stage4_selector_feature_ablation.py` with all ten held regions and `k = 1, 2, 3`; the ablated selector is retuned on source regions only.
7. Run `stage4_mechanism_attribution.py` with all ten held regions and `k = 1, 2, 3`.

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

Stage 4 uses the same cleaned `haa6br_integrated_v1` package and the locked strict-v4 predictions/caches. The six-endpoint multiplicity sensitivity draws one shared ten-region sample per bootstrap replicate across all three monitoring depths, then resamples systems within the sampled region separately for each depth cohort. Selector ablation removes `calibration_samples` and `calibration_sites` jointly, locks the source-only policy before loading target candidates, and never uses held-region outcomes to select a variant.

## Environment

The compatibility ranges are in `requirements.txt`. The public package was smoke-checked with Python 3.12, NumPy 1.26.4, pandas 2.3.3, SciPy 1.13.1, XGBoost 3.2.0, and matplotlib 3.9.2. Full execution additionally needs scikit-learn 1.6 or newer because the protocol uses shuffled `GroupKFold`, plus an XGBoost-compatible NVIDIA CUDA runtime. GPU library and driver versions should be recorded with any published rerun.

CUDA training can exhibit small platform-dependent floating-point differences. Treat the saved protocol, script and input hashes, selected actions, and tolerance-based audits as the reproducibility record rather than expecting byte-identical GPU artifacts across hardware.

The logical paths, byte counts, and SHA256 values for the locked inputs, protocol, metadata, and manuscript-facing outputs are recorded in [PROVENANCE_MANIFEST.json](PROVENANCE_MANIFEST.json). The manifest explicitly distinguishes the unavailable exact executed runner from this public implementation.

The Stage 4 mechanism analysis reproduces the locked selector decisions before computing any counterfactual variant. The application-layer no-cap sensitivity retains the original selector and therefore is not an independently retrained algorithm. The count/site ablation and all mechanism contrasts are post-hoc and non-confirmatory.

## Clean rerun

Use a new output directory for an independent rerun. Do not seed it with old tables. A validated read-only prediction cache can reduce runtime, but the cache manifest and array hashes are checked before reuse. Deleteable caches are rebuildable and must not be included in a source release.

Generated metadata intentionally records logical artifact names rather than private absolute paths. The repository ignores outputs, caches, local environments, and data-like files.

## Interpretation limits

The negative-transfer budget controls an empirical algorithmic error criterion. It is not a toxicological threshold, regulatory compliance statement, uncertainty guarantee for every system, or drinking-water health-safety guarantee. Prediction bands are source-derived empirical intervals, not formal cluster-conformal coverage guarantees.

The Stage 4 bootstrap tails are finite Monte Carlo estimates. At 5,000 replicates, the Bonferroni 0.00417 tail is supported by roughly 21 order statistics; report the interval as a conservative sensitivity analysis rather than a precise inferential boundary.
