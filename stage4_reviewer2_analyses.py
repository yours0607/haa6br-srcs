from __future__ import annotations

import argparse
import hashlib
import json
import platform
import site
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / ".deps"))

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.distance import jensenshannon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The locked experimental runtime keeps NumPy/Pandas in the base environment,
# while torch is installed in the user site and is imported by shared modules.
user_site = Path(site.getusersitepackages())
if user_site.exists() and str(user_site) not in sys.path:
    sys.path.append(str(user_site))

import run_optimized_experiments as optimized
import stage4_mechanism_attribution as mechanism
from run_new_experiments import GROUP, REGION, Paths, load_data, sha256_file
from stage4_revision_analyses import ANALYSIS_STATUS, strict_cvar90


PRIMARY_BUDGET = 0.12
GATE_ACTIONS = {
    "coverage_matched_capped_raw_mean": optimized.action_name("RawMean", 1.0),
    "coverage_matched_capped_history_mean": optimized.action_name("HistoryMean", 1.0),
}


def stable_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    return np.unique(
        np.round(
            np.concatenate(
                [
                    np.array([0.0]),
                    np.quantile(values, np.linspace(0.0, 1.0, 101)),
                    np.array([float(np.max(values) + 1e-9)]),
                ]
            ),
            decimals=12,
        )
    )


def gated_system_outcomes(
    systems: pd.DataFrame,
    threshold: float,
    action: str,
) -> pd.DataFrame:
    required = {
        GROUP,
        REGION,
        "k",
        "base_mae",
        "abs_mean_residual",
        f"actual__{action}",
    }
    missing = sorted(required.difference(systems.columns))
    if missing:
        raise ValueError(f"Coverage-matched gate is missing columns: {missing}")

    adapted = systems["abs_mean_residual"].to_numpy(float) >= float(threshold)
    action_regret = systems[f"actual__{action}"].to_numpy(float)
    regret = np.where(adapted, action_regret, 0.0)
    return pd.DataFrame(
        {
            REGION: systems[REGION].to_numpy(),
            GROUP: systems[GROUP].to_numpy(),
            "k": systems["k"].to_numpy(int),
            "base_abs": systems["base_mae"].to_numpy(float),
            "variant_abs": systems["base_mae"].to_numpy(float) + regret,
            "regret": regret,
            "adapted": adapted,
            "selected_action": np.where(adapted, action, "Zero-shot"),
        }
    )


def select_source_coverage_gate(
    source_systems: pd.DataFrame,
    target_coverage: float,
    action: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    candidates = threshold_candidates(
        source_systems["abs_mean_residual"].to_numpy(float)
    )
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        outcomes = gated_system_outcomes(source_systems, float(threshold), action)
        regret = outcomes["regret"].to_numpy(float)
        coverage = float(outcomes["adapted"].mean())
        rows.append(
            {
                "threshold_abs_mean_residual_ug_l": float(threshold),
                "coverage_target": float(target_coverage),
                "adaptation_rate": coverage,
                "absolute_coverage_gap": abs(coverage - float(target_coverage)),
                "mean_regret": float(np.mean(regret)),
                "negative_transfer_rate": float(np.mean(regret > 1e-12)),
                "strict_cvar90_regret": strict_cvar90(regret),
            }
        )
    search = pd.DataFrame(rows)
    winner = search.sort_values(
        [
            "absolute_coverage_gap",
            "mean_regret",
            "strict_cvar90_regret",
            "threshold_abs_mean_residual_ug_l",
        ],
        kind="mergesort",
    ).iloc[0]
    return winner.to_dict(), search


def summarize_system_outcomes(frame: pd.DataFrame, variant: str, k: int) -> dict:
    regret = frame["regret"].to_numpy(float)
    adapted = frame["adapted"].to_numpy(bool)
    corrected = regret[adapted]
    if corrected.size == 0:
        conditional = {
            "corrected_systems": 0,
            "conditional_mean_regret": np.nan,
            "conditional_negative_transfer_rate": np.nan,
            "conditional_strict_cvar90_regret": np.nan,
            "conditional_p95_regret": np.nan,
            "conditional_maximum_regret": np.nan,
        }
    else:
        conditional = {
            "corrected_systems": int(corrected.size),
            "conditional_mean_regret": float(np.mean(corrected)),
            "conditional_negative_transfer_rate": float(
                np.mean(corrected > 1e-12)
            ),
            "conditional_strict_cvar90_regret": strict_cvar90(corrected),
            "conditional_p95_regret": float(np.quantile(corrected, 0.95)),
            "conditional_maximum_regret": float(np.max(corrected)),
        }
    return {
        "analysis_status": ANALYSIS_STATUS,
        "confirmatory_status": "not_confirmatory",
        "k": int(k),
        "variant": variant,
        "systems": int(len(frame)),
        "regions": int(frame[REGION].nunique()),
        "equal_system_mae": float(frame["variant_abs"].mean()),
        "mean_regret_vs_zero_shot": float(np.mean(regret)),
        "negative_transfer_rate": float(np.mean(regret > 1e-12)),
        "strict_cvar90_regret": strict_cvar90(regret),
        "p95_regret": float(np.quantile(regret, 0.95)),
        "maximum_regret": float(np.max(regret)),
        "adaptation_rate": float(np.mean(adapted)),
        "fallback_rate": float(1.0 - np.mean(adapted)),
        **conditional,
    }


def coverage_matched_analysis(
    optimized_output: Path,
    data_package: Path,
    mechanism_output: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = Paths(
        data_package,
        output_dir,
        output_dir / "tables",
        output_dir / "figures",
        output_dir / "locks",
    )
    core, _, _, feature_sets = load_data(paths)
    features = list(feature_sets["us_operational_core"])
    all_regions = tuple(sorted(int(value) for value in core[REGION].unique()))

    prediction_cache = optimized.RegionalPredictionCache(
        core,
        features,
        optimized.MODEL_NAME,
        output_dir / "cache",
        "us_operational",
        optimized_output / "cache",
    )
    locked_candidate_dir = (
        optimized_output
        / "cache"
        / "us_operational"
        / optimized.MODEL_NAME.replace(" ", "_")
        / f"candidate_tables_{optimized.CANDIDATE_CACHE_VERSION}"
    )
    candidates = mechanism.ReadOnlyCandidateTableCache(
        prediction_cache,
        locked_candidate_dir,
        output_dir / "cache" / "candidate_fallback",
    )
    policy_specs = pd.read_csv(
        optimized_output / "analysis" / "risk_budget_policy_specs.csv",
        low_memory=False,
    )
    policy_specs = policy_specs.loc[
        np.isclose(policy_specs["risk_budget"], PRIMARY_BUDGET)
    ].copy()

    existing = pd.read_csv(
        mechanism_output / "analysis" / "mechanism_variant_system_outcomes.csv",
        low_memory=False,
    )
    existing = existing.loc[
        existing["variant"].isin(
            ["full_srcs_reproduced", "fixed_threshold_capped_raw_mean"]
        )
    ].copy()

    target_outputs: list[pd.DataFrame] = []
    search_outputs: list[pd.DataFrame] = []
    spec_rows: list[dict] = []
    for outer_region in all_regions:
        source_regions = tuple(
            region for region in all_regions if region != outer_region
        )
        for k in (1, 2, 3):
            print(
                f"[coverage match] outer={outer_region}, k={k}",
                flush=True,
            )
            source_frames = []
            for held_region in source_regions:
                systems, _ = candidates.get(
                    (outer_region, held_region), held_region, k, False
                )
                source_frames.append(systems)
            source_systems = pd.concat(source_frames, ignore_index=True)
            target_systems, _ = candidates.get(
                (outer_region,), outer_region, k, False
            )

            spec = policy_specs.loc[
                policy_specs["outer_target_region"].eq(outer_region)
                & policy_specs["k"].eq(k)
            ]
            if len(spec) != 1:
                raise AssertionError(
                    f"Expected one 12% source policy spec for outer={outer_region}, k={k}"
                )
            target_coverage = float(spec.iloc[0]["source_adaptation_rate"])

            for variant, action in GATE_ACTIONS.items():
                winner, search = select_source_coverage_gate(
                    source_systems,
                    target_coverage,
                    action,
                )
                search.insert(0, "action", action)
                search.insert(0, "variant", variant)
                search.insert(0, "k", k)
                search.insert(0, "outer_target_region", outer_region)
                search_outputs.append(search)

                threshold = float(winner["threshold_abs_mean_residual_ug_l"])
                target = gated_system_outcomes(target_systems, threshold, action)
                target["variant"] = variant
                target["outer_target_region"] = outer_region
                target_outputs.append(target)
                spec_rows.append(
                    {
                        "outer_target_region": outer_region,
                        "k": k,
                        "variant": variant,
                        "action": action,
                        "threshold_abs_mean_residual_ug_l": threshold,
                        "source_coverage_target": target_coverage,
                        "source_adaptation_rate": float(winner["adaptation_rate"]),
                        "source_absolute_coverage_gap": float(
                            winner["absolute_coverage_gap"]
                        ),
                        "source_mean_regret": float(winner["mean_regret"]),
                        "source_negative_transfer_rate": float(
                            winner["negative_transfer_rate"]
                        ),
                        "source_strict_cvar90_regret": float(
                            winner["strict_cvar90_regret"]
                        ),
                        "selection_evidence": (
                            "source-only absolute-mean-residual threshold chosen to "
                            "match the source adaptation rate of the locked 12% SRCS policy"
                        ),
                    }
                )

    target_table = pd.concat(target_outputs, ignore_index=True)
    existing = existing[
        [
            REGION,
            GROUP,
            "k",
            "base_abs",
            "variant_abs",
            "regret",
            "adapted",
            "selected_action",
            "variant",
            "outer_target_region",
        ]
    ]
    target_table = pd.concat([existing, target_table], ignore_index=True)

    summaries = []
    for (k, variant), frame in target_table.groupby(["k", "variant"]):
        summaries.append(summarize_system_outcomes(frame, str(variant), int(k)))
    summary = pd.DataFrame(summaries)

    contrast_rows = []
    for k, frame in target_table.groupby("k"):
        full = frame.loc[frame["variant"].eq("full_srcs_reproduced")].set_index(
            GROUP
        )
        for variant in GATE_ACTIONS:
            comparator = frame.loc[frame["variant"].eq(variant)].set_index(GROUP)
            if set(full.index) != set(comparator.index):
                raise AssertionError(f"Coverage-matched cohort mismatch: {variant}, k={k}")
            contrast_rows.append(
                {
                    "analysis_status": ANALYSIS_STATUS,
                    "confirmatory_status": "not_confirmatory",
                    "k": int(k),
                    "method": "full_srcs_reproduced",
                    "comparator": variant,
                    "equal_system_mae_difference": float(
                        full["variant_abs"].mean()
                        - comparator["variant_abs"].mean()
                    ),
                    "negative_transfer_rate_difference": float(
                        np.mean(full["regret"] > 1e-12)
                        - np.mean(comparator["regret"] > 1e-12)
                    ),
                    "strict_cvar90_regret_difference": float(
                        strict_cvar90(full["regret"].to_numpy(float))
                        - strict_cvar90(comparator["regret"].to_numpy(float))
                    ),
                    "adaptation_rate_difference": float(
                        full["adapted"].mean() - comparator["adapted"].mean()
                    ),
                    "target_coverage_matching_note": (
                        "Thresholds were coverage-matched on source regions; held-target "
                        "coverage was not used for selection."
                    ),
                }
            )
    return (
        summary,
        pd.DataFrame(contrast_rows),
        pd.DataFrame(spec_rows),
        target_table,
    )


def hierarchical_bootstrap_contrasts(
    system_table: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for k, depth in system_table.groupby("k"):
        full = depth.loc[depth["variant"].eq("full_srcs_reproduced")].copy()
        for comparator_name in GATE_ACTIONS:
            comparator = depth.loc[depth["variant"].eq(comparator_name)].copy()
            paired = full.merge(
                comparator,
                on=[REGION, GROUP, "k"],
                how="inner",
                suffixes=("_full", "_comparator"),
                validate="one_to_one",
            )
            if len(paired) != len(full) or len(paired) != len(comparator):
                raise AssertionError(
                    f"Coverage-matched bootstrap cohort mismatch: {comparator_name}, k={k}"
                )
            regions = np.array(sorted(paired[REGION].unique()), dtype=int)
            region_indices = {
                region: np.flatnonzero(paired[REGION].to_numpy(int) == region)
                for region in regions
            }
            distributions = {
                "equal_system_mae_difference": np.empty(n_boot),
                "negative_transfer_rate_difference": np.empty(n_boot),
                "strict_cvar90_regret_difference": np.empty(n_boot),
                "adaptation_rate_difference": np.empty(n_boot),
            }
            full_mae = paired["variant_abs_full"].to_numpy(float)
            comparator_mae = paired["variant_abs_comparator"].to_numpy(float)
            full_regret = paired["regret_full"].to_numpy(float)
            comparator_regret = paired["regret_comparator"].to_numpy(float)
            full_adapted = paired["adapted_full"].to_numpy(bool)
            comparator_adapted = paired["adapted_comparator"].to_numpy(bool)
            for iteration in range(n_boot):
                sampled_regions = rng.choice(regions, size=len(regions), replace=True)
                sampled_systems = []
                for sampled_region in sampled_regions:
                    candidates = region_indices[int(sampled_region)]
                    sampled_systems.append(
                        rng.choice(candidates, size=len(candidates), replace=True)
                    )
                index = np.concatenate(sampled_systems)
                distributions["equal_system_mae_difference"][iteration] = float(
                    np.mean(full_mae[index] - comparator_mae[index])
                )
                distributions["negative_transfer_rate_difference"][iteration] = float(
                    np.mean(full_regret[index] > 1e-12)
                    - np.mean(comparator_regret[index] > 1e-12)
                )
                distributions["strict_cvar90_regret_difference"][iteration] = float(
                    strict_cvar90(full_regret[index])
                    - strict_cvar90(comparator_regret[index])
                )
                distributions["adaptation_rate_difference"][iteration] = float(
                    np.mean(full_adapted[index])
                    - np.mean(comparator_adapted[index])
                )

            point_values = {
                "equal_system_mae_difference": float(
                    np.mean(full_mae - comparator_mae)
                ),
                "negative_transfer_rate_difference": float(
                    np.mean(full_regret > 1e-12)
                    - np.mean(comparator_regret > 1e-12)
                ),
                "strict_cvar90_regret_difference": float(
                    strict_cvar90(full_regret) - strict_cvar90(comparator_regret)
                ),
                "adaptation_rate_difference": float(
                    np.mean(full_adapted) - np.mean(comparator_adapted)
                ),
            }
            for metric, values in distributions.items():
                rows.append(
                    {
                        "analysis_status": ANALYSIS_STATUS,
                        "confirmatory_status": "not_confirmatory",
                        "k": int(k),
                        "method": "full_srcs_reproduced",
                        "comparator": comparator_name,
                        "metric": metric,
                        "point_estimate": point_values[metric],
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "bootstrap_replicates": int(n_boot),
                        "bootstrap_seed": int(seed),
                        "multiplicity": "unadjusted descriptive interval",
                        "direction": "negative values favor full SRCS",
                    }
                )
    return pd.DataFrame(rows)


def mode_or_missing(series: pd.Series) -> str:
    values = series.fillna("__MISSING__").astype(str)
    modes = values.mode(dropna=False)
    return str(modes.iloc[0]) if len(modes) else "__MISSING__"


def system_feature_frame(core: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    aggregations: dict[str, object] = {}
    for feature in features:
        if feature == "month":
            aggregations[feature] = "median"
        else:
            aggregations[feature] = mode_or_missing
    systems = core.groupby([GROUP, REGION], as_index=False).agg(aggregations)
    systems[REGION] = systems[REGION].astype(int)
    return systems


def geographic_shift_diagnostics(
    core: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    systems = system_feature_frame(core, features)
    numeric = ["month"]
    categorical = [feature for feature in features if feature not in numeric]
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant", fill_value="__MISSING__"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=5,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    classifier = Pipeline(
        [
            ("features", transformer),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    classifier_rows = []
    for region in sorted(systems[REGION].unique()):
        y = systems[REGION].eq(region).astype(int).to_numpy()
        probability = cross_val_predict(
            classifier,
            systems[features],
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        classifier_rows.append(
            {
                "epa_region": int(region),
                "systems_in_region": int(np.sum(y)),
                "systems_outside_region": int(np.sum(1 - y)),
                "one_vs_rest_system_level_auroc": float(
                    roc_auc_score(y, probability)
                ),
                "one_vs_rest_system_level_brier": float(
                    brier_score_loss(y, probability)
                ),
                "diagnostic_role": (
                    "post-hoc feature-separability diagnostic; not a transport estimate"
                ),
            }
        )

    jsd_rows = []
    for region in sorted(systems[REGION].unique()):
        in_region = systems.loc[systems[REGION].eq(region)]
        outside = systems.loc[systems[REGION].ne(region)]
        for feature in features:
            left = in_region[feature].fillna("__MISSING__").astype(str)
            right = outside[feature].fillna("__MISSING__").astype(str)
            levels = sorted(set(left.unique()).union(right.unique()))
            left_probability = (
                left.value_counts(normalize=True).reindex(levels, fill_value=0.0)
            )
            right_probability = (
                right.value_counts(normalize=True).reindex(levels, fill_value=0.0)
            )
            jsd_rows.append(
                {
                    "epa_region": int(region),
                    "feature": feature,
                    "levels": int(len(levels)),
                    "jensen_shannon_distance": float(
                        jensenshannon(
                            left_probability.to_numpy(float),
                            right_probability.to_numpy(float),
                            base=2.0,
                        )
                    ),
                }
            )

    target_rows = []
    for region, frame in core.groupby(REGION):
        target = frame[optimized.TARGET].to_numpy(float)
        target_rows.append(
            {
                "epa_region": int(region),
                "records": int(len(frame)),
                "systems": int(frame[GROUP].nunique()),
                "haa6br_mean_ug_l": float(np.mean(target)),
                "haa6br_median_ug_l": float(np.median(target)),
                "haa6br_q25_ug_l": float(np.quantile(target, 0.25)),
                "haa6br_q75_ug_l": float(np.quantile(target, 0.75)),
                "haa6br_p95_ug_l": float(np.quantile(target, 0.95)),
            }
        )
    return (
        pd.DataFrame(classifier_rows),
        pd.DataFrame(jsd_rows),
        pd.DataFrame(target_rows),
    )


def risk_coverage_figure_data(
    optimized_output: Path,
    conditional_summary: pd.DataFrame,
) -> pd.DataFrame:
    budget = pd.read_csv(
        optimized_output / "analysis" / "risk_budget_full_summary.csv",
        low_memory=False,
    )
    budget = budget[
        [
            "k",
            "risk_budget",
            "system_round_balanced_mae",
            "negative_transfer_rate",
            "strict_cvar90_regret",
            "adaptation_rate",
        ]
    ].rename(columns={"system_round_balanced_mae": "equal_system_mae"})
    budget["series"] = "SRCS budget path"
    budget["display_label"] = budget["risk_budget"].map(
        lambda value: f"{100 * value:.0f}% budget"
    )

    comparator_names = {
        "coverage_matched_capped_history_mean": "Coverage-matched capped History gate",
        "fixed_threshold_capped_raw_mean": "Conservative capped RawMean gate",
    }
    comparator = conditional_summary.loc[
        conditional_summary["variant"].isin(comparator_names)
    ][
        [
            "k",
            "variant",
            "equal_system_mae",
            "negative_transfer_rate",
            "strict_cvar90_regret",
            "adaptation_rate",
        ]
    ].copy()
    comparator["series"] = comparator["variant"].map(comparator_names)
    comparator["display_label"] = comparator["series"]
    comparator["risk_budget"] = np.nan
    comparator.drop(columns=["variant"], inplace=True)
    return pd.concat([budget, comparator], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc analyses requested in the second Stage 4 review"
    )
    parser.add_argument(
        "--optimized-output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "optimized_srcs_strict_v4_20260728",
    )
    parser.add_argument(
        "--data-package",
        type=Path,
        default=PROJECT_DIR.parent / "haa6br_data" / "haa6br_integrated_v1",
    )
    parser.add_argument(
        "--mechanism-output",
        type=Path,
        default=(
            PROJECT_DIR
            / "outputs"
            / "stage4_revision_20260730"
            / "mechanism_attribution"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "outputs"
            / "stage4_revision_20260730"
            / "reviewer2_analyses"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimized_output = args.optimized_output.resolve()
    data_package = args.data_package.resolve()
    mechanism_output = args.mechanism_output.resolve()
    output_dir = args.output_dir.resolve()
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "locked_us_predictions": optimized_output / "tables" / "us_predictions.csv",
        "risk_budget_policy_specs": optimized_output
        / "analysis"
        / "risk_budget_policy_specs.csv",
        "mechanism_system_outcomes": mechanism_output
        / "analysis"
        / "mechanism_variant_system_outcomes.csv",
        "risk_budget_full_summary": optimized_output
        / "analysis"
        / "risk_budget_full_summary.csv",
        "clean_us_core": data_package / "data" / "us_ucmr4_core.csv",
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reviewer-analysis inputs: {missing}")
    hashes_before = {name: stable_sha256(path) for name, path in input_paths.items()}

    summary, contrasts, specifications, system_outcomes = coverage_matched_analysis(
        optimized_output,
        data_package,
        mechanism_output,
        output_dir,
    )
    bootstrap = hierarchical_bootstrap_contrasts(
        system_outcomes,
        n_boot=5000,
        seed=args.seed + 1000,
    )

    paths = Paths(
        data_package,
        output_dir,
        output_dir / "tables",
        output_dir / "figures",
        output_dir / "locks",
    )
    core, _, _, feature_sets = load_data(paths)
    features = list(feature_sets["us_operational_core"])
    classifier, jsd, target = geographic_shift_diagnostics(
        core,
        features,
        args.seed,
    )
    risk_coverage = risk_coverage_figure_data(optimized_output, summary)

    outputs = {
        "coverage_matched_conditional_summary.csv": summary,
        "coverage_matched_contrasts.csv": contrasts,
        "coverage_matched_source_specs.csv": specifications,
        "coverage_matched_bootstrap_contrasts.csv": bootstrap,
        "coverage_matched_system_outcomes.csv": system_outcomes,
        "geographic_shift_classifier.csv": classifier,
        "geographic_shift_feature_jsd.csv": jsd,
        "geographic_shift_target_distribution.csv": target,
        "risk_coverage_figure_data.csv": risk_coverage,
    }
    for name, frame in outputs.items():
        frame.to_csv(analysis_dir / name, index=False, encoding="utf-8-sig")

    hashes_after = {name: stable_sha256(path) for name, path in input_paths.items()}
    if hashes_before != hashes_after:
        raise AssertionError("Locked reviewer-analysis inputs changed during execution")

    metadata = {
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": ANALYSIS_STATUS,
        "confirmatory_status": "not_confirmatory",
        "scope": (
            "source-coverage-matched simple gates, corrected-system conditional regret, "
            "and descriptive geographic-shift diagnostics"
        ),
        "coverage_matching": (
            "Thresholds used source regions only and targeted each locked 12% SRCS "
            "policy's source adaptation rate; held-target coverage was not matched or tuned."
        ),
        "geographic_classifier": (
            "Five-fold system-level one-vs-rest logistic classification of EPA region "
            "from the eight operational base-model fields."
        ),
        "limitations": [
            "All analyses were requested after earlier result inspection.",
            "Geographic separability does not establish a causal shift mechanism.",
            "Intervals from earlier analyses remain conditional on fixed fitted predictions.",
        ],
        "seed": args.seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "script_sha256": sha256_file(Path(__file__)),
        "input_sha256": hashes_before,
        "output_sha256": {
            name: stable_sha256(analysis_dir / name) for name in outputs
        },
    }
    (output_dir / "stage4_reviewer2_analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
