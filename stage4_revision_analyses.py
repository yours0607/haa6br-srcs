from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / ".deps"))

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t

from analyze_optimized_results import GROUP, REGION, sha256_file, strict_cvar90


ANALYSIS_STATUS = "posthoc_revision_sensitivity"
BASE_SEED = 20260730
METHODS = (
    "SRCS",
    "History mean",
    "Capped History mean",
    "Raw residual",
    "Capped Raw residual",
    "Zero-shot",
)
INTERVAL_COMPARATORS = (
    "History mean",
    "Capped History mean",
    "Capped Raw residual",
)
REGRET_THRESHOLDS = (1e-12, 0.01, 0.10, 0.50)
UK_SPECIES = (
    ("MBAA", "monobromoacetic acid", "mbaa_ug_l_lower_bound", "mbaa_result_sign"),
    ("BCAA", "bromochloroacetic acid", "bcaa_ug_l_lower_bound", "bcaa_result_sign"),
    ("DBAA", "dibromoacetic acid", "dbaa_ug_l_lower_bound", "dbaa_result_sign"),
    ("BDCAA", "bromodichloroacetic acid", "bdcaa_ug_l_lower_bound", "bdcaa_result_sign"),
    ("CDBAA", "chlorodibromoacetic acid", "cdbaa_ug_l_lower_bound", "cdbaa_result_sign"),
    ("TBAA", "tribromoacetic acid", "tbaa_ug_l_lower_bound", "tbaa_result_sign"),
)


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    unknown = sorted(set(normalized.dropna()).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported boolean values: {unknown[:5]}")
    return normalized.isin({"true", "1", "yes"})


def assign_rounds(frame: pd.DataFrame) -> pd.DataFrame:
    required = {GROUP, "sample_id", "sample_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Round assignment is missing columns: {missing}")
    output = frame.copy()
    output["sample_date"] = pd.to_datetime(output["sample_date"], errors="coerce")
    output = output.loc[output["sample_date"].notna()].copy()
    output.sort_values([GROUP, "sample_date", "sample_id"], inplace=True)
    output["round_index"] = (
        output.groupby(GROUP, sort=False)["sample_date"]
        .rank(method="dense")
        .astype(int)
    )
    output["system_rounds"] = (
        output.groupby(GROUP)["round_index"].transform("max").astype(int)
    )
    return output.reset_index(drop=True)


def system_method_errors(
    predictions: pd.DataFrame,
    k: int,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    required = {
        "k",
        REGION,
        GROUP,
        "round_index",
        "observed",
        "Zero-shot",
        *methods,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")
    subset = predictions.loc[predictions["k"].eq(k)].copy()
    if subset.empty:
        raise ValueError(f"No prediction rows for k={k}")

    id_columns = [REGION, GROUP, "round_index"]
    round_frame = subset[id_columns].copy()
    metric_columns: list[str] = []
    for method in methods:
        round_frame[method] = np.abs(
            subset["observed"].to_numpy(float) - subset[method].to_numpy(float)
        )
        bias_column = f"bias__{method}"
        round_frame[bias_column] = (
            subset[method].to_numpy(float) - subset["observed"].to_numpy(float)
        )
        metric_columns.extend([method, bias_column])
    system = (
        round_frame.groupby(id_columns, as_index=False)[metric_columns]
        .mean()
        .groupby([REGION, GROUP], as_index=False)[metric_columns]
        .mean()
    )
    for method in methods:
        system[f"regret__{method}"] = system[method] - system["Zero-shot"]

    if {"adapted", "selected_action"}.issubset(subset.columns):
        actions = subset.groupby([REGION, GROUP], as_index=False).agg(
            adapted=("adapted", "first"),
            selected_action=("selected_action", "first"),
        )
        actions["adapted"] = parse_bool(actions["adapted"])
        system = system.merge(
            actions,
            on=[REGION, GROUP],
            how="left",
            validate="one_to_one",
        )
    return system


def summarize_system_frame(
    system: pd.DataFrame,
    k: int,
    cohort: str,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in methods:
        errors = system[method].to_numpy(float)
        regrets = system[f"regret__{method}"].to_numpy(float)
        row: dict[str, object] = {
            "analysis_status": ANALYSIS_STATUS,
            "cohort": cohort,
            "k": int(k),
            "method": method,
            "systems": int(len(system)),
            "regions": int(system[REGION].nunique()),
            "equal_system_mae": float(np.mean(errors)),
            "equal_system_signed_bias": float(
                np.mean(system[f"bias__{method}"].to_numpy(float))
            ),
            "mean_regret_vs_zero_shot": float(np.mean(regrets)),
            "strict_cvar90_regret": strict_cvar90(regrets),
            "p95_regret": float(np.quantile(regrets, 0.95)),
            "maximum_regret": float(np.max(regrets)),
            "minimum_regret": float(np.min(regrets)),
        }
        for threshold in REGRET_THRESHOLDS:
            row[f"negative_transfer_rate_gt_{threshold:g}"] = float(
                np.mean(regrets > threshold)
            )
        if method == "SRCS" and "adapted" in system:
            row["adaptation_rate"] = float(parse_bool(system["adapted"]).mean())
            row["fallback_rate"] = 1.0 - float(row["adaptation_rate"])
        else:
            row["adaptation_rate"] = np.nan
            row["fallback_rate"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_hierarchical_positions(
    system: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    region_array = system[REGION].to_numpy()
    regions = pd.unique(region_array)
    positions = {
        region: np.flatnonzero(region_array == region) for region in regions
    }
    sampled_regions = rng.choice(regions, size=len(regions), replace=True)
    return np.concatenate(
        [
            rng.choice(positions[region], size=len(positions[region]), replace=True)
            for region in sampled_regions
        ]
    )


def _sample_positions_for_regions(
    system: pd.DataFrame,
    sampled_regions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    region_array = system[REGION].to_numpy()
    positions = {
        region: np.flatnonzero(region_array == region)
        for region in pd.unique(region_array)
    }
    missing = sorted(set(sampled_regions).difference(positions))
    if missing:
        raise ValueError(f"Joint bootstrap regions are absent from a k cohort: {missing}")
    return np.concatenate(
        [
            rng.choice(positions[region], size=len(positions[region]), replace=True)
            for region in sampled_regions
        ]
    )


def joint_family_bootstrap(
    system_by_k: dict[int, pd.DataFrame],
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Jointly resample EPA regions for the locked six-endpoint family."""
    if n_boot < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    if sorted(system_by_k) != [1, 2, 3]:
        raise ValueError(f"Expected k=1,2,3; found {sorted(system_by_k)}")

    region_sets = {
        k: set(pd.unique(system[REGION])) for k, system in system_by_k.items()
    }
    reference_regions = region_sets[1]
    if any(regions != reference_regions for regions in region_sets.values()):
        raise ValueError(f"Joint bootstrap requires the same region frame: {region_sets}")
    regions = np.asarray(sorted(reference_regions))
    rng = np.random.default_rng(seed)
    distributions = {
        f"k{k}|History mean|negative_transfer_rate_difference|{1e-12}": np.empty(
            n_boot, dtype=float
        )
        for k in (1, 2, 3)
    }
    distributions.update(
        {
            f"k{k}|History mean|strict_cvar90_regret_difference|None": np.empty(
                n_boot, dtype=float
            )
            for k in (1, 2, 3)
        }
    )

    regrets = {
        k: {
            "SRCS": system["regret__SRCS"].to_numpy(float),
            "History mean": system["regret__History mean"].to_numpy(float),
        }
        for k, system in system_by_k.items()
    }
    for iteration in range(n_boot):
        sampled_regions = rng.choice(regions, size=len(regions), replace=True)
        for k, system in system_by_k.items():
            idx = _sample_positions_for_regions(system, sampled_regions, rng)
            srcs = regrets[k]["SRCS"][idx]
            history = regrets[k]["History mean"][idx]
            distributions[
                f"k{k}|History mean|negative_transfer_rate_difference|{1e-12}"
            ][iteration] = float(
                np.mean(srcs > 1e-12) - np.mean(history > 1e-12)
            )
            distributions[
                f"k{k}|History mean|strict_cvar90_regret_difference|None"
            ][iteration] = strict_cvar90(srcs) - strict_cvar90(history)
    return distributions


def joint_family_draws_frame(
    distributions: dict[str, np.ndarray],
    seed: int,
) -> pd.DataFrame:
    frames = []
    for token, values in sorted(distributions.items()):
        k_token, comparator, metric, threshold = token.split("|", maxsplit=3)
        frames.append(
            pd.DataFrame(
                {
                    "analysis_status": ANALYSIS_STATUS,
                    "bootstrap_replicate": np.arange(1, len(values) + 1),
                    "bootstrap_seed": int(seed),
                    "k": int(k_token.removeprefix("k")),
                    "comparator": comparator,
                    "metric": metric,
                    "regret_threshold_ug_l": (
                        np.nan if threshold == "None" else float(threshold)
                    ),
                    "estimate": values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def matched_bootstrap_contrasts(
    system: pd.DataFrame,
    k: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if n_boot < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    arrays = {method: system[method].to_numpy(float) for method in METHODS}
    regrets = {
        method: system[f"regret__{method}"].to_numpy(float) for method in METHODS
    }

    metric_specs: list[tuple[str, float | None]] = [
        ("equal_system_mae_difference", None),
        ("strict_cvar90_regret_difference", None),
        ("p95_regret_difference", None),
        ("maximum_regret_difference", None),
    ]
    metric_specs.extend(
        ("negative_transfer_rate_difference", threshold)
        for threshold in REGRET_THRESHOLDS
    )
    distributions: dict[str, np.ndarray] = {}
    for comparator in INTERVAL_COMPARATORS:
        for metric, threshold in metric_specs:
            token = f"k{k}|{comparator}|{metric}|{threshold}"
            distributions[token] = np.empty(n_boot, dtype=float)

    for iteration in range(n_boot):
        idx = _sample_hierarchical_positions(system, rng)
        for comparator in INTERVAL_COMPARATORS:
            for metric, threshold in metric_specs:
                token = f"k{k}|{comparator}|{metric}|{threshold}"
                if metric == "equal_system_mae_difference":
                    value = float(np.mean(arrays["SRCS"][idx] - arrays[comparator][idx]))
                elif metric == "strict_cvar90_regret_difference":
                    value = strict_cvar90(regrets["SRCS"][idx]) - strict_cvar90(
                        regrets[comparator][idx]
                    )
                elif metric == "p95_regret_difference":
                    value = float(
                        np.quantile(regrets["SRCS"][idx], 0.95)
                        - np.quantile(regrets[comparator][idx], 0.95)
                    )
                elif metric == "maximum_regret_difference":
                    value = float(
                        np.max(regrets["SRCS"][idx])
                        - np.max(regrets[comparator][idx])
                    )
                else:
                    assert threshold is not None
                    value = float(
                        np.mean(regrets["SRCS"][idx] > threshold)
                        - np.mean(regrets[comparator][idx] > threshold)
                    )
                distributions[token][iteration] = value

    rows: list[dict[str, object]] = []
    for comparator in INTERVAL_COMPARATORS:
        for metric, threshold in metric_specs:
            token = f"k{k}|{comparator}|{metric}|{threshold}"
            estimates = distributions[token]
            if metric == "equal_system_mae_difference":
                point = float(np.mean(arrays["SRCS"] - arrays[comparator]))
            elif metric == "strict_cvar90_regret_difference":
                point = strict_cvar90(regrets["SRCS"]) - strict_cvar90(
                    regrets[comparator]
                )
            elif metric == "p95_regret_difference":
                point = float(
                    np.quantile(regrets["SRCS"], 0.95)
                    - np.quantile(regrets[comparator], 0.95)
                )
            elif metric == "maximum_regret_difference":
                point = float(
                    np.max(regrets["SRCS"]) - np.max(regrets[comparator])
                )
            else:
                assert threshold is not None
                point = float(
                    np.mean(regrets["SRCS"] > threshold)
                    - np.mean(regrets[comparator] > threshold)
                )
            below = float(np.mean(estimates < 0.0))
            equal = float(np.mean(estimates == 0.0))
            above = float(np.mean(estimates > 0.0))
            rows.append(
                {
                    "analysis_status": ANALYSIS_STATUS,
                    "k": int(k),
                    "method": "SRCS",
                    "comparator": comparator,
                    "contrast": f"SRCS - {comparator}",
                    "metric": metric,
                    "regret_threshold_ug_l": threshold,
                    "systems": int(len(system)),
                    "regions": int(system[REGION].nunique()),
                    "point_estimate": point,
                    "ci_low": float(np.quantile(estimates, 0.025)),
                    "ci_high": float(np.quantile(estimates, 0.975)),
                    "bootstrap_probability_lt_zero": below,
                    "bootstrap_probability_eq_zero": equal,
                    "bootstrap_probability_gt_zero": above,
                    "two_sided_descriptive_tail_probability": min(
                        1.0, 2.0 * min(below + equal, above + equal)
                    ),
                    "bootstrap_replicates": int(n_boot),
                    "bootstrap_seed": int(seed),
                    "direction": "Lower values favor SRCS",
                }
            )
    return pd.DataFrame(rows), distributions


def bonferroni_family_sensitivity(
    contrasts: pd.DataFrame,
    distributions: dict[str, np.ndarray],
) -> pd.DataFrame:
    family = contrasts.loc[
        contrasts["comparator"].eq("History mean")
        & (
            contrasts["metric"].eq("strict_cvar90_regret_difference")
            | (
                contrasts["metric"].eq("negative_transfer_rate_difference")
                & contrasts["regret_threshold_ug_l"].eq(1e-12)
            )
        )
    ].copy()
    if len(family) != 6:
        raise AssertionError(f"Expected six multiplicity-family rows, found {len(family)}")
    alpha = 0.05
    family_size = len(family)
    lower_q = alpha / (2.0 * family_size)
    upper_q = 1.0 - lower_q
    rows = []
    for row in family.to_dict("records"):
        threshold = row["regret_threshold_ug_l"]
        threshold_token = None if pd.isna(threshold) else threshold
        token = (
            f"k{int(row['k'])}|History mean|{row['metric']}|{threshold_token}"
        )
        estimates = distributions[token]
        rows.append(
            {
                **row,
                "family": "History-relative negative-transfer and strict-CVaR90 at k=1,2,3",
                "family_size": family_size,
                "adjustment": "Bonferroni percentile-bootstrap sensitivity",
                "familywise_alpha": alpha,
                "adjusted_lower_quantile": lower_q,
                "adjusted_upper_quantile": upper_q,
                "joint_unadjusted_ci_low": float(np.quantile(estimates, 0.025)),
                "joint_unadjusted_ci_high": float(np.quantile(estimates, 0.975)),
                "adjusted_ci_low": float(np.quantile(estimates, lower_q)),
                "adjusted_ci_high": float(np.quantile(estimates, upper_q)),
                "confirmatory_status": "not_confirmatory",
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "k"]).reset_index(drop=True)


def exact_sign_flip_p(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    if clean.ndim != 1 or len(clean) == 0:
        raise ValueError("Sign-flip input must be a non-empty vector")
    observed = abs(float(np.mean(clean)))
    distribution = np.fromiter(
        (
            abs(float(np.mean(clean * np.asarray(signs, dtype=float))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(clean))
        ),
        dtype=float,
        count=2 ** len(clean),
    )
    return float(np.mean(distribution >= observed - 1e-15))


def region_influence_tables(
    system_by_k: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regional_rows = []
    loo_rows = []
    small_cluster_rows = []
    for k, system in system_by_k.items():
        for comparator in ("History mean", "Capped History mean"):
            regional_effects: dict[str, list[float]] = {
                "equal_system_mae_difference": [],
                "negative_transfer_rate_difference": [],
                "strict_cvar90_regret_difference": [],
            }
            for region, subset in system.groupby(REGION, sort=True):
                srcs_regret = subset["regret__SRCS"].to_numpy(float)
                comparator_regret = subset[f"regret__{comparator}"].to_numpy(float)
                effects = {
                    "equal_system_mae_difference": float(
                        np.mean(subset["SRCS"] - subset[comparator])
                    ),
                    "negative_transfer_rate_difference": float(
                        np.mean(srcs_regret > 1e-12)
                        - np.mean(comparator_regret > 1e-12)
                    ),
                    "strict_cvar90_regret_difference": strict_cvar90(srcs_regret)
                    - strict_cvar90(comparator_regret),
                }
                regional_rows.append(
                    {
                        "analysis_status": ANALYSIS_STATUS,
                        "k": int(k),
                        "region": region,
                        "comparator": comparator,
                        "systems": int(len(subset)),
                        **effects,
                    }
                )
                for metric, value in effects.items():
                    regional_effects[metric].append(value)

            for omitted_region in sorted(pd.unique(system[REGION])):
                subset = system.loc[~system[REGION].eq(omitted_region)]
                srcs_regret = subset["regret__SRCS"].to_numpy(float)
                comparator_regret = subset[f"regret__{comparator}"].to_numpy(float)
                loo_rows.append(
                    {
                        "analysis_status": ANALYSIS_STATUS,
                        "k": int(k),
                        "omitted_region": omitted_region,
                        "comparator": comparator,
                        "systems": int(len(subset)),
                        "regions": int(subset[REGION].nunique()),
                        "equal_system_mae_difference": float(
                            np.mean(subset["SRCS"] - subset[comparator])
                        ),
                        "negative_transfer_rate_difference": float(
                            np.mean(srcs_regret > 1e-12)
                            - np.mean(comparator_regret > 1e-12)
                        ),
                        "strict_cvar90_regret_difference": strict_cvar90(srcs_regret)
                        - strict_cvar90(comparator_regret),
                    }
                )

            for metric, values in regional_effects.items():
                array = np.asarray(values, dtype=float)
                n_regions = len(array)
                mean_effect = float(np.mean(array))
                se = float(np.std(array, ddof=1) / math.sqrt(n_regions))
                critical = float(student_t.ppf(0.975, df=n_regions - 1))
                small_cluster_rows.append(
                    {
                        "analysis_status": ANALYSIS_STATUS,
                        "target_population": "fixed ten-EPA-region frame",
                        "k": int(k),
                        "comparator": comparator,
                        "metric": metric,
                        "regions": n_regions,
                        "equal_region_mean_effect": mean_effect,
                        "student_t_ci_low": mean_effect - critical * se,
                        "student_t_ci_high": mean_effect + critical * se,
                        "exact_two_sided_sign_flip_p": exact_sign_flip_p(array),
                        "interpretation": "small-cluster sensitivity, not confirmatory inference",
                    }
                )
    return (
        pd.DataFrame(regional_rows),
        pd.DataFrame(loo_rows),
        pd.DataFrame(small_cluster_rows),
    )


def fixed_cohort_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    systems_by_k = {
        int(k): set(subset[GROUP].unique())
        for k, subset in predictions.groupby("k")
    }
    if sorted(systems_by_k) != [1, 2, 3]:
        raise ValueError(f"Expected k=1,2,3; found {sorted(systems_by_k)}")
    common = set.intersection(*systems_by_k.values())
    members = pd.DataFrame({GROUP: sorted(common)})
    summaries = []
    for k in (1, 2, 3):
        full = system_method_errors(predictions, k)
        fixed_predictions = predictions.loc[
            predictions["k"].eq(k) & predictions[GROUP].isin(common)
        ]
        fixed = system_method_errors(fixed_predictions, k)
        summaries.extend(
            [
                summarize_system_frame(full, k, "full_depth_specific_cohort"),
                summarize_system_frame(fixed, k, "common_k1_k2_k3_system_cohort"),
            ]
        )
    return pd.concat(summaries, ignore_index=True), members


def site_continuity_tables(
    core: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rounds = assign_rounds(core)
    detail_frames = []
    stable_summaries = []
    intensity_frames = []
    for k in (1, 2, 3):
        eligible = set(
            rounds.groupby(GROUP)["round_index"].max().loc[lambda x: x > k].index
        )
        work = rounds.loc[rounds[GROUP].isin(eligible)].copy()
        calibration = work.loc[work["round_index"] <= k].copy()
        evaluation = work.loc[work["round_index"] > k].copy()
        calibration_sets = calibration.groupby(GROUP)["group_site_id"].agg(
            lambda x: frozenset(x.dropna())
        )
        evaluation_sets = evaluation.groupby(GROUP)["group_site_id"].agg(
            lambda x: frozenset(x.dropna())
        )
        detail = pd.DataFrame({GROUP: sorted(eligible)}).set_index(GROUP)
        detail["calibration_sites_set"] = calibration_sets
        detail["evaluation_sites_set"] = evaluation_sets
        detail["calibration_samples"] = calibration.groupby(GROUP).size()
        detail["evaluation_samples"] = evaluation.groupby(GROUP).size()
        detail["calibration_rounds"] = calibration.groupby(GROUP)["round_index"].nunique()
        detail["evaluation_rounds"] = evaluation.groupby(GROUP)["round_index"].nunique()
        detail["calibration_last_date"] = calibration.groupby(GROUP)["sample_date"].max()
        detail["evaluation_first_date"] = evaluation.groupby(GROUP)["sample_date"].min()
        detail["calibration_site_count"] = detail["calibration_sites_set"].map(len)
        detail["evaluation_site_count"] = detail["evaluation_sites_set"].map(len)
        detail["shared_site_count"] = [
            len(a.intersection(b))
            for a, b in zip(
                detail["calibration_sites_set"], detail["evaluation_sites_set"]
            )
        ]
        detail["any_site_continuity"] = detail["shared_site_count"] > 0
        detail["elapsed_days"] = (
            detail["evaluation_first_date"] - detail["calibration_last_date"]
        ).dt.days
        calibration_lookup = detail["calibration_sites_set"].to_dict()
        evaluation["stable_site_row"] = [
            site in calibration_lookup[system]
            for system, site in zip(evaluation[GROUP], evaluation["group_site_id"])
        ]
        stable_counts = evaluation.groupby(GROUP)["stable_site_row"].sum()
        detail["stable_site_evaluation_rows"] = stable_counts
        detail["stable_site_evaluation_fraction"] = (
            detail["stable_site_evaluation_rows"] / detail["evaluation_samples"]
        )
        detail["k"] = k

        system = system_method_errors(predictions, k).set_index(GROUP)
        detail = detail.join(
            system[
                [
                    REGION,
                    "adapted",
                    "selected_action",
                    "regret__SRCS",
                    "regret__History mean",
                    "SRCS",
                    "History mean",
                ]
            ],
            how="left",
            validate="one_to_one",
        )
        if detail[REGION].isna().any():
            raise AssertionError(f"Site diagnostics failed to align predictions at k={k}")
        detail["adapted"] = parse_bool(detail["adapted"])
        detail_frames.append(
            detail.drop(columns=["calibration_sites_set", "evaluation_sites_set"])
            .reset_index()
        )

        stable_ids = set(
            evaluation.loc[evaluation["stable_site_row"], "sample_id"].astype(str)
        )
        stable_predictions = predictions.loc[
            predictions["k"].eq(k)
            & predictions["sample_id"].astype(str).isin(stable_ids)
        ].copy()
        if not stable_predictions.empty:
            stable_system = system_method_errors(stable_predictions, k)
            stable_summaries.append(
                summarize_system_frame(
                    stable_system,
                    k,
                    "future_rows_from_calibration_observed_sites",
                )
            )

        intensity = detail.reset_index()
        intensity["calibration_sample_quartile"] = pd.qcut(
            intensity["calibration_samples"],
            q=4,
            labels=False,
            duplicates="drop",
        )
        intensity["calibration_site_group"] = pd.cut(
            intensity["calibration_site_count"],
            bins=[-np.inf, 1, 2, np.inf],
            labels=["1", "2", "3+"],
        )
        for variable in (
            "calibration_sample_quartile",
            "calibration_site_group",
            "any_site_continuity",
        ):
            grouped = intensity.groupby(variable, dropna=False, observed=True)
            frame = grouped.agg(
                systems=(GROUP, "nunique"),
                adaptation_rate=("adapted", "mean"),
                mean_srcs_regret=("regret__SRCS", "mean"),
                srcs_negative_transfer_rate=(
                    "regret__SRCS",
                    lambda x: float(np.mean(np.asarray(x, dtype=float) > 1e-12)),
                ),
                mean_history_regret=("regret__History mean", "mean"),
            ).reset_index()
            frame.insert(0, "stratifier", variable)
            frame.rename(columns={variable: "stratum"}, inplace=True)
            frame.insert(0, "k", k)
            intensity_frames.append(frame)

    detail_all = pd.concat(detail_frames, ignore_index=True)
    summary = (
        detail_all.groupby("k", as_index=False)
        .agg(
            systems=(GROUP, "nunique"),
            median_elapsed_days=("elapsed_days", "median"),
            median_calibration_sites=("calibration_site_count", "median"),
            median_evaluation_sites=("evaluation_site_count", "median"),
            systems_with_any_site_continuity=("any_site_continuity", "sum"),
            proportion_with_any_site_continuity=("any_site_continuity", "mean"),
            mean_stable_site_evaluation_fraction=(
                "stable_site_evaluation_fraction",
                "mean",
            ),
        )
    )
    return (
        detail_all,
        summary,
        pd.concat(stable_summaries, ignore_index=True),
        pd.concat(intensity_frames, ignore_index=True),
    )


def us_target_audit(
    core: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "sample_id",
        "haa6br_ug_l",
        "target_method",
        "target_quality_flag",
        "target_units",
        "target_result_sign",
        "analytical_method",
        "primary_analysis_eligible",
        "is_formation_potential",
        "is_us_primary_period",
        "is_2021_sensitivity",
    }
    missing = sorted(required.difference(core.columns))
    if missing:
        raise ValueError(f"U.S. target audit is missing columns: {missing}")
    summary = pd.DataFrame(
        [
            {
                "rows": int(len(core)),
                "unique_sample_ids": int(core["sample_id"].nunique()),
                "duplicate_sample_ids": int(core["sample_id"].duplicated().sum()),
                "duplicate_key_rule": "sample_id must be unique after the integrated-package duplicate resolution",
                "target_missing": int(core["haa6br_ug_l"].isna().sum()),
                "target_negative": int((core["haa6br_ug_l"] < 0).sum()),
                "primary_analysis_rows": int(
                    parse_bool(core["primary_analysis_eligible"]).sum()
                ),
                "excluded_from_primary_rows": int(
                    (~parse_bool(core["primary_analysis_eligible"])).sum()
                ),
                "primary_exclusion_rule": "primary_analysis_eligible == True; formation-potential and 2021 sensitivity rows remain outside the primary modelling cohort",
                "target_method": "|".join(sorted(core["target_method"].dropna().unique())),
                "target_units": "|".join(sorted(core["target_units"].dropna().unique())),
                "result_signs": "|".join(
                    sorted(core["target_result_sign"].fillna("MISSING").astype(str).unique())
                ),
                "analytical_methods": int(core["analytical_method"].nunique(dropna=True)),
                "definition": "unweighted sum of MBAA+BCAA+DBAA+BDCAA+CDBAA+TBAA",
                "implementation": "EPA-reported HAA6Br aggregate consumed directly; component rows were not re-summed",
                "source_fields": "haa6br_ug_l|target_method|target_quality_flag|target_units|target_result_sign|analytical_method",
                "health_boundary": "occurrence concentration, not a toxicity index or HAA5 compliance metric",
            }
        ]
    )
    methods = (
        core.groupby("analytical_method", dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    count_frames = []
    primary = parse_bool(core["primary_analysis_eligible"])
    for field in (
        "target_method",
        "target_quality_flag",
        "target_units",
        "target_result_sign",
        "is_formation_potential",
        "is_us_primary_period",
        "is_2021_sensitivity",
    ):
        work = pd.DataFrame(
            {
                "value": core[field].fillna("MISSING").astype(str),
                "primary_analysis_eligible": primary,
            }
        )
        counts = (
            work.groupby("value", dropna=False)
            .agg(
                rows=("value", "size"),
                primary_analysis_rows=("primary_analysis_eligible", "sum"),
            )
            .reset_index()
        )
        counts.insert(0, "field", field)
        count_frames.append(counts)
    return summary, methods, pd.concat(count_frames, ignore_index=True)


def uk_target_audit(
    uk: pd.DataFrame,
    uk_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for code, name, value_column, sign_column in UK_SPECIES:
        signs = uk[sign_column].fillna("MISSING").astype(str)
        rows.append(
            {
                "species_code": code,
                "species_name": name,
                "value_field": value_column,
                "sign_field": sign_column,
                "rows": int(len(uk)),
                "nondetect_rows": int(signs.eq("<").sum()),
                "nondetect_fraction": float(signs.eq("<").mean()),
                "not_sampled_rows": int(signs.eq("NS").sum()),
                "lower_bound_min_ug_l": float(uk[value_column].min()),
                "lower_bound_max_ug_l": float(uk[value_column].max()),
            }
        )
    species = pd.DataFrame(rows)
    difference = uk["haa6br_target_abs_difference_ug_l"].to_numpy(float)
    construction = pd.DataFrame(
        [
            {
                "rows": int(len(uk)),
                "systems": int(uk[GROUP].nunique()),
                "primary_method": "HAA9 - MCAA - DCAA - TCAA; explicit less-than values set to zero",
                "sensitivity_method": "direct lower-bound sum of MBAA+BCAA+DBAA+BDCAA+CDBAA+TBAA",
                "rows_with_any_nondetect": int((uk["n_species_nondetect"] > 0).sum()),
                "total_species_nondetects": int(uk["n_species_nondetect"].sum()),
                "mean_abs_reconstruction_difference_ug_l": float(np.mean(difference)),
                "median_abs_reconstruction_difference_ug_l": float(np.median(difference)),
                "p95_abs_reconstruction_difference_ug_l": float(
                    np.quantile(difference, 0.95)
                ),
                "maximum_abs_reconstruction_difference_ug_l": float(np.max(difference)),
                "rows_abs_difference_gt_0_5_ug_l": int(np.sum(difference > 0.5)),
                "censoring_limit": "integrated v1 retains signs and lower-bound values, not original reporting limits; MRL/2 cannot be reconstructed",
            }
        ]
    )

    target = uk[["sample_id", "haa6br_ug_l", "haa6br_component_lower_bound_ug_l"]]
    merged = uk_predictions.drop(columns=["observed"]).merge(
        target,
        on="sample_id",
        how="inner",
        validate="many_to_one",
    )
    target_summaries = []
    for target_name, target_column in (
        ("HAA9_minus_three_primary", "haa6br_ug_l"),
        ("direct_six_species_lower_bound", "haa6br_component_lower_bound_ug_l"),
    ):
        frame = merged.copy()
        frame["observed"] = frame[target_column]
        for k in (1, 2, 3):
            system = system_method_errors(frame, k)
            summary = summarize_system_frame(system, k, target_name)
            summary.insert(2, "target_construction", target_name)
            target_summaries.append(summary)
    return construction, species, pd.concat(target_summaries, ignore_index=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def validate_integrated_package(data_package: Path) -> dict:
    if data_package.name != "haa6br_integrated_v1":
        raise ValueError("Only the cleaned haa6br_integrated_v1 package is permitted")
    report_path = data_package / "metadata" / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = str(report.get("status", report.get("overall_status", ""))).upper()
    if status != "PASS":
        raise ValueError(f"Integrated package validation status is not PASS: {status}")
    return report


def validate_locked_output(optimized_output: Path) -> dict:
    if optimized_output.name != "optimized_srcs_strict_v4_20260728":
        raise ValueError("Stage 4 may read only optimized_srcs_strict_v4_20260728")
    metadata_path = optimized_output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS_EXECUTION_AND_AUDIT":
        raise ValueError(f"Locked strict-v4 run did not pass: {metadata.get('status')}")
    leakage_checks = metadata.get("leakage_checks", {})
    if not leakage_checks or not all(bool(value) for value in leakage_checks.values()):
        raise ValueError(f"Locked strict-v4 leakage checks are incomplete: {leakage_checks}")
    if metadata.get("outer_regions") != list(range(1, 11)):
        raise ValueError("Locked strict-v4 metadata does not contain all ten EPA regions")
    if metadata.get("k_values") != [1, 2, 3]:
        raise ValueError("Locked strict-v4 metadata does not contain k=1,2,3")
    return metadata


def ensure_output_is_separate(
    output_dir: Path,
    protected_roots: tuple[Path, ...],
) -> None:
    for root in protected_roots:
        if output_dir == root or root in output_dir.parents:
            raise ValueError(f"Output directory must not be inside protected input: {root}")


def repository_state(repository: Path) -> dict[str, object]:
    if not (repository / ".git").exists():
        return {"path": str(repository), "commit": None, "dirty": None}

    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )

    head = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "path": str(repository),
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def validate_analysis_frames(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "details": details})
        if not passed:
            raise AssertionError(f"Stage 4 output validation failed: {name}: {details}")

    matched = frames["matched_comparator_outcome_matrix.csv"]
    record(
        "matched method-depth registry",
        len(matched) == 18
        and set(matched["method"]) == set(METHODS)
        and set(matched["k"]) == {1, 2, 3},
        f"rows={len(matched)}, methods={matched['method'].nunique()}, k={sorted(matched['k'].unique())}",
    )
    finite_columns = [
        "equal_system_mae",
        "equal_system_signed_bias",
        "mean_regret_vs_zero_shot",
        "strict_cvar90_regret",
        "p95_regret",
        "maximum_regret",
    ]
    record(
        "matched finite outcomes",
        bool(np.isfinite(matched[finite_columns].to_numpy(float)).all()),
        "all core matched outcomes finite",
    )
    rate_columns = [
        column for column in matched if column.startswith("negative_transfer_rate_gt_")
    ]
    rates = matched[rate_columns].to_numpy(float)
    record(
        "negative-transfer rate bounds",
        bool(((rates >= 0.0) & (rates <= 1.0)).all()),
        f"min={rates.min():.6g}, max={rates.max():.6g}",
    )

    multiplicity = frames["multiplicity_bonferroni_sensitivity.csv"]
    record(
        "six-endpoint multiplicity family",
        len(multiplicity) == 6
        and set(multiplicity["k"]) == {1, 2, 3}
        and set(multiplicity["metric"])
        == {
            "negative_transfer_rate_difference",
            "strict_cvar90_regret_difference",
        },
        f"rows={len(multiplicity)}",
    )
    record(
        "Bonferroni intervals contain same-resample unadjusted intervals",
        bool(
            (
                multiplicity["adjusted_ci_low"]
                <= multiplicity["joint_unadjusted_ci_low"] + 1e-15
            ).all()
            and (
                multiplicity["adjusted_ci_high"]
                >= multiplicity["joint_unadjusted_ci_high"] - 1e-15
            ).all()
        ),
        "adjusted tails are no narrower than the joint 95% percentile intervals",
    )

    fixed_members = frames["fixed_cohort_members.csv"]
    fixed_summary = frames["fixed_cohort_summary.csv"]
    fixed_rows = fixed_summary.loc[
        fixed_summary["cohort"].eq("common_k1_k2_k3_system_cohort")
    ]
    record(
        "fixed-cohort denominator",
        bool(fixed_rows["systems"].eq(len(fixed_members)).all()),
        f"members={len(fixed_members)}, summary systems={sorted(fixed_rows['systems'].unique())}",
    )

    site_detail = frames["site_continuity_system_detail.csv"]
    record(
        "site-detail unique system-depth keys",
        not site_detail.duplicated(["k", GROUP]).any(),
        f"rows={len(site_detail)}",
    )
    expected_by_k = matched.loc[matched["method"].eq("SRCS")].set_index("k")["systems"]
    observed_by_k = site_detail.groupby("k")[GROUP].nunique()
    record(
        "site-detail matches locked prediction cohorts",
        expected_by_k.equals(observed_by_k),
        f"expected={expected_by_k.to_dict()}, observed={observed_by_k.to_dict()}",
    )

    us_target = frames["us_target_construction_audit.csv"].iloc[0]
    record(
        "U.S. target unique and complete",
        int(us_target["duplicate_sample_ids"]) == 0
        and int(us_target["target_missing"]) == 0
        and int(us_target["target_negative"]) == 0,
        f"duplicates={us_target['duplicate_sample_ids']}, missing={us_target['target_missing']}, negative={us_target['target_negative']}",
    )
    record(
        "all output tables non-empty",
        all(len(frame) > 0 for frame in frames.values()),
        f"tables={len(frames)}",
    )
    return pd.DataFrame(checks)


def run_stage4_analyses(
    optimized_output: Path,
    data_package: Path,
    output_dir: Path,
    n_boot: int,
    seed: int,
) -> dict:
    optimized_output = optimized_output.resolve()
    data_package = data_package.resolve()
    output_dir = output_dir.resolve()
    ensure_output_is_separate(output_dir, (optimized_output, data_package))
    validation = validate_integrated_package(data_package)
    locked_metadata = validate_locked_output(optimized_output)

    predictions_path = optimized_output / "tables" / "us_predictions.csv"
    uk_predictions_path = optimized_output / "tables" / "uk_predictions.csv"
    locked_metadata_path = optimized_output / "run_metadata.json"
    locked_protocol_path = optimized_output / "locks" / "protocol_lock_before_optimized_run.json"
    us_core_path = data_package / "data" / "us_ucmr4_core.csv"
    uk_path = data_package / "data" / "uk_dwi242_locked_external.csv"
    validation_path = data_package / "metadata" / "validation_report.json"
    package_hash_manifest_path = data_package / "metadata" / "SHA256SUMS.txt"
    input_paths = {
        "locked_us_predictions": predictions_path,
        "locked_uk_predictions": uk_predictions_path,
        "locked_run_metadata": locked_metadata_path,
        "locked_protocol": locked_protocol_path,
        "clean_us_core": us_core_path,
        "clean_uk_dwi242": uk_path,
        "integrated_validation_report": validation_path,
        "integrated_hash_manifest": package_hash_manifest_path,
    }
    input_hashes_before = {
        name: sha256_file(path) for name, path in input_paths.items()
    }
    predictions = pd.read_csv(predictions_path, low_memory=False)
    uk_predictions = pd.read_csv(uk_predictions_path, low_memory=False)
    us_core_all = pd.read_csv(us_core_path, low_memory=False)
    us_core_all["primary_analysis_eligible"] = parse_bool(
        us_core_all["primary_analysis_eligible"]
    )
    us_core = us_core_all.loc[us_core_all["primary_analysis_eligible"]].copy()
    uk = pd.read_csv(uk_path, low_memory=False)

    system_by_k = {k: system_method_errors(predictions, k) for k in (1, 2, 3)}
    matched_summary = pd.concat(
        [
            summarize_system_frame(system_by_k[k], k, "full_depth_specific_cohort")
            for k in (1, 2, 3)
        ],
        ignore_index=True,
    )
    contrast_frames = []
    distributions: dict[str, np.ndarray] = {}
    for k in (1, 2, 3):
        frame, dist = matched_bootstrap_contrasts(
            system_by_k[k], k, n_boot=n_boot, seed=seed + 100 * k
        )
        contrast_frames.append(frame)
        distributions.update(dist)
    matched_contrasts = pd.concat(contrast_frames, ignore_index=True)
    joint_family_seed = seed + 7000
    joint_family_distributions = joint_family_bootstrap(
        system_by_k,
        n_boot=n_boot,
        seed=joint_family_seed,
    )
    distributions.update(joint_family_distributions)
    multiplicity = bonferroni_family_sensitivity(matched_contrasts, distributions)
    joint_family_draws = joint_family_draws_frame(
        joint_family_distributions,
        seed=joint_family_seed,
    )
    regional, leave_one_out, small_cluster = region_influence_tables(system_by_k)
    fixed_summary, fixed_members = fixed_cohort_tables(predictions)
    site_detail, site_summary, stable_site, intensity = site_continuity_tables(
        us_core, predictions
    )
    us_target_summary, us_method_counts, us_target_field_counts = us_target_audit(
        us_core_all
    )
    uk_construction, uk_species, uk_sensitivity = uk_target_audit(uk, uk_predictions)

    frames = {
        "matched_comparator_outcome_matrix.csv": matched_summary,
        "matched_comparator_bootstrap_contrasts.csv": matched_contrasts,
        "multiplicity_bonferroni_sensitivity.csv": multiplicity,
        "multiplicity_joint_bootstrap_draws.csv": joint_family_draws,
        "region_level_paired_effects.csv": regional,
        "leave_one_region_out_influence.csv": leave_one_out,
        "small_cluster_fixed_frame_sensitivity.csv": small_cluster,
        "fixed_cohort_summary.csv": fixed_summary,
        "fixed_cohort_members.csv": fixed_members,
        "site_continuity_system_detail.csv": site_detail,
        "site_continuity_summary.csv": site_summary,
        "stable_site_prediction_sensitivity.csv": stable_site,
        "sampling_intensity_action_diagnostics.csv": intensity,
        "us_target_construction_audit.csv": us_target_summary,
        "us_analytical_method_counts.csv": us_method_counts,
        "us_target_field_counts.csv": us_target_field_counts,
        "uk_target_construction_audit.csv": uk_construction,
        "uk_species_censoring_audit.csv": uk_species,
        "uk_target_sensitivity_results.csv": uk_sensitivity,
    }
    validation_checks = validate_analysis_frames(frames)
    frames["stage4_analysis_validation_checks.csv"] = validation_checks
    analysis_dir = output_dir / "analysis"
    for name, frame in frames.items():
        _write_csv(frame, analysis_dir / name)

    metadata_path = output_dir / "stage4_revision_analysis_metadata.json"
    input_hashes_after = {
        name: sha256_file(path) for name, path in input_paths.items()
    }
    locked_inputs_unchanged = input_hashes_after == input_hashes_before
    if not locked_inputs_unchanged:
        changed = sorted(
            name
            for name in input_hashes_before
            if input_hashes_before[name] != input_hashes_after[name]
        )
        raise AssertionError(f"Protected Stage 4 inputs changed during analysis: {changed}")

    public_repository = PROJECT_DIR.parent / "open_source" / "haa6br-srcs"
    metadata = {
        "status": "PASS",
        "analysis_status": ANALYSIS_STATUS,
        "confirmatory_status": "not_confirmatory",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "precommitment": "haa6br_manuscript/review/stage4/00_revision_analysis_precommitment.md",
        "data_contract": {
            "permitted_package": "haa6br_integrated_v1",
            "resolved_path": str(data_package),
            "validation_status": validation.get(
                "status", validation.get("overall_status")
            ),
            "raw_or_uncleaned_data_used": False,
        },
        "inputs": {
            name: {"path": str(path), "sha256": input_hashes_before[name]}
            for name, path in input_paths.items()
        },
        "locked_run_contract": {
            "status": locked_metadata["status"],
            "outer_regions": locked_metadata["outer_regions"],
            "k_values": locked_metadata["k_values"],
            "leakage_checks": locked_metadata["leakage_checks"],
        },
        "code_provenance": {
            "stage4_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "locked_generator_script_sha256": json.loads(
                locked_protocol_path.read_text(encoding="utf-8")
            )["script_sha256"],
            "public_repository": repository_state(public_repository),
        },
        "bootstrap": {
            "replicates": int(n_boot),
            "base_seed": int(seed),
            "per_k_seed_rule": "base_seed + 100*k",
            "sampling": "paired EPA-region then system bootstrap",
            "multiplicity_family_size": 6,
            "multiplicity_adjustment": "Bonferroni percentile-bootstrap sensitivity",
            "multiplicity_joint_seed": int(joint_family_seed),
            "multiplicity_joint_sampling": "one shared ten-region draw per replicate across k=1,2,3; systems resampled within sampled region separately for each k cohort",
        },
        "regret_thresholds_ug_l": list(REGRET_THRESHOLDS),
        "threshold_boundary": "prediction-error increments only; no health or regulatory meaning",
        "outputs": {},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "limitations": [
            "All analyses are post-hoc revision sensitivities on retrospective locked predictions.",
            "Only ten EPA regions are available for outer-geography sensitivity.",
            "Fixed-cohort and stable-site analyses change the target population and are not replacements for the full-cohort estimand.",
            "The integrated UK table retains non-detect signs and lower-bound values but not original reporting limits, so MRL/2 cannot be reconstructed.",
            "Prediction negative transfer and regret are not drinking-water health-safety outcomes.",
        ],
        "locked_inputs_hashes_verified_before_and_after": True,
        "locked_strict_v4_outputs_modified": not locked_inputs_unchanged,
    }
    for name, frame in frames.items():
        path = analysis_dir / name
        metadata["outputs"][name] = {
            "rows": int(len(frame)),
            "sha256": sha256_file(path),
        }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    default_optimized = PROJECT_DIR / "outputs" / "optimized_srcs_strict_v4_20260728"
    default_data = PROJECT_DIR.parent / "haa6br_data" / "haa6br_integrated_v1"
    default_output = PROJECT_DIR / "outputs" / "stage4_revision_20260730"
    parser = argparse.ArgumentParser(
        description="Post-hoc Stage 4 robustness and reconstructability analyses"
    )
    parser.add_argument("--optimized-output", type=Path, default=default_optimized)
    parser.add_argument("--data-package", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_stage4_analyses(
        optimized_output=args.optimized_output,
        data_package=args.data_package,
        output_dir=args.output_dir,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
