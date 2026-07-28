from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


GROUP = "group_system_id"
REGION = "epa_region"
SEED = 20260728


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_cvar90(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    count = max(1, int(math.ceil(0.10 * len(values))))
    return float(np.mean(np.sort(values)[-count:]))


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = observed - predicted
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else np.nan,
        "medae": float(np.median(np.abs(residual))),
        "bias": float(np.mean(predicted - observed)),
    }


def system_round_errors(predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    columns = [REGION, GROUP, "round_index", "observed", "Zero-shot"]
    if method != "Zero-shot":
        columns.append(method)
    work = predictions[columns].copy()
    work["base_abs"] = np.abs(work["observed"] - work["Zero-shot"])
    work["method_abs"] = (
        work["base_abs"]
        if method == "Zero-shot"
        else np.abs(work["observed"] - work[method])
    )
    rounds = (
        work.groupby([REGION, GROUP, "round_index"], as_index=False)[
            ["base_abs", "method_abs"]
        ]
        .mean()
    )
    systems = rounds.groupby([REGION, GROUP], as_index=False)[
        ["base_abs", "method_abs"]
    ].mean()
    systems["delta"] = systems["method_abs"] - systems["base_abs"]
    return systems


def sample_balanced_system_errors(predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    columns = [REGION, GROUP, "observed", "Zero-shot"]
    if method != "Zero-shot":
        columns.append(method)
    work = predictions[columns].copy()
    work["base_abs"] = np.abs(work["observed"] - work["Zero-shot"])
    work["method_abs"] = (
        work["base_abs"]
        if method == "Zero-shot"
        else np.abs(work["observed"] - work[method])
    )
    systems = work.groupby([REGION, GROUP], as_index=False)[
        ["base_abs", "method_abs"]
    ].mean()
    systems["delta"] = systems["method_abs"] - systems["base_abs"]
    return systems


def hierarchical_paired_bootstrap(
    systems: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    region_values = {
        region: frame["delta"].to_numpy(float)
        for region, frame in systems.groupby(REGION)
    }
    regions = np.asarray(list(region_values), dtype=object)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        sampled_regions = rng.choice(regions, size=len(regions), replace=True)
        draws = []
        for region in sampled_regions:
            values = region_values[region]
            draws.append(rng.choice(values, size=len(values), replace=True))
        estimates[iteration] = float(np.mean(np.concatenate(draws)))
    below = float(np.mean(estimates < 0.0))
    above = float(np.mean(estimates > 0.0))
    return {
        "estimate": float(systems["delta"].mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_probability_below_zero": below,
        "two_sided_bootstrap_p": min(1.0, 2.0 * min(below, above)),
        "bootstrap_replicates": int(n_boot),
    }


def primary_contrasts(
    tables: Path,
    n_boot: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    comparators = [
        "Zero-shot",
        "History mean",
        "Raw residual",
        "Capped History mean",
        "Capped Raw residual",
    ]
    for dataset, filename, evidence in (
        ("US", "us_predictions.csv", "Geographic internal-external validation"),
        ("UK", "uk_predictions.csv", "Retrospective stress test; outcomes previously viewed"),
    ):
        predictions = pd.read_csv(tables / filename, low_memory=False)
        for k in sorted(predictions["k"].unique()):
            subset = predictions.loc[predictions["k"] == k]
            srcs = system_round_errors(subset, "SRCS").set_index([REGION, GROUP])
            for index, comparator in enumerate(comparators):
                other = system_round_errors(subset, comparator).set_index([REGION, GROUP])
                paired = srcs[["method_abs"]].rename(
                    columns={"method_abs": "srcs_mae"}
                ).join(
                    other[["method_abs"]].rename(columns={"method_abs": "comparator_mae"}),
                    how="inner",
                    validate="one_to_one",
                )
                paired["delta"] = paired["srcs_mae"] - paired["comparator_mae"]
                paired = paired.reset_index()
                result = hierarchical_paired_bootstrap(
                    paired,
                    n_boot,
                    SEED + (10000 if dataset == "UK" else 0) + 100 * int(k) + index,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "evidence": evidence,
                        "k": int(k),
                        "method": "SRCS",
                        "comparator": comparator,
                        "contrast": f"SRCS - {comparator}",
                        "estimand": "Equal-system, equal-future-round MAE difference",
                        "systems": int(len(paired)),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def optimization_comparison(
    optimized_tables: Path,
    historical_tables: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    configurations = [
        (
            "Historical v2",
            "core_predictions.csv",
            ["Zero-shot", "History mean", "Raw residual", "HRC", "SafeShrink HRC"],
            "Historical geographic validation; source safety tuning not fully nested",
        ),
        (
            "Optimized SRCS v4",
            "us_predictions.csv",
            [
                "Zero-shot",
                "History mean",
                "Raw residual",
                "Capped History mean",
                "Capped Raw residual",
                "SRCS",
            ],
            "Post-hoc nested geographic internal-external validation",
        ),
    ]
    for version, filename, methods, evidence in configurations:
        source = historical_tables if version == "Historical v2" else optimized_tables
        predictions = pd.read_csv(source / filename, low_memory=False)
        for k in sorted(predictions["k"].unique()):
            subset = predictions.loc[predictions["k"] == k].copy()
            zero_mae = float(np.mean(np.abs(subset["observed"] - subset["Zero-shot"])))
            for method in methods:
                metrics = regression_metrics(subset["observed"], subset[method])
                systems = system_round_errors(subset, method)
                legacy = sample_balanced_system_errors(subset, method)
                coverage = np.nan
                interval_width = np.nan
                if version == "Historical v2" and method == "SafeShrink HRC":
                    low = subset[method] - subset["q_safe"]
                    high = subset[method] + subset["q_safe"]
                    coverage = float(np.mean((subset["observed"] >= low) & (subset["observed"] <= high)))
                    interval_width = float(np.mean(high - low))
                elif version == "Optimized SRCS v4" and method == "SRCS":
                    coverage = float(
                        np.mean(
                            (subset["observed"] >= subset["interval_low"])
                            & (subset["observed"] <= subset["interval_high"])
                        )
                    )
                    interval_width = float(np.mean(subset["interval_high"] - subset["interval_low"]))
                adaptation = np.nan
                if version == "Historical v2" and method == "SafeShrink HRC":
                    decisions = subset[[GROUP, "safe_adapt"]].drop_duplicates()
                    adaptation = float(decisions["safe_adapt"].mean())
                elif version == "Optimized SRCS v4" and method == "SRCS":
                    decisions = subset[[GROUP, "adapted"]].drop_duplicates()
                    adaptation = float(decisions["adapted"].mean())
                rows.append(
                    {
                        "version": version,
                        "evidence": evidence,
                        "k": int(k),
                        "method": method,
                        "rows": int(len(subset)),
                        "systems": int(subset[GROUP].nunique()),
                        **metrics,
                        "relative_sample_mae_improvement": (zero_mae - metrics["mae"])
                        / max(zero_mae, 1e-12),
                        "system_round_balanced_mae": float(systems["method_abs"].mean()),
                        "negative_transfer_equal_round": float(np.mean(systems["delta"] > 1e-12)),
                        "negative_transfer_sample_balanced": float(np.mean(legacy["delta"] > 1e-12)),
                        "mean_regret": float(systems["delta"].mean()),
                        "strict_cvar90_regret": strict_cvar90(systems["delta"].to_numpy(float)),
                        "p95_regret": float(np.quantile(systems["delta"], 0.95)),
                        "max_regret": float(systems["delta"].max()),
                        "coverage_90": coverage,
                        "mean_interval_width": interval_width,
                        "system_adaptation_rate": adaptation,
                    }
                )
    return pd.DataFrame(rows)


def risk_frontier(tables: Path) -> pd.DataFrame:
    region_summary = pd.read_csv(tables / "us_region_summary.csv")
    system_summary = pd.read_csv(tables / "us_system_summary.csv")
    sensitivity = pd.read_csv(tables / "risk_budget_sensitivity.csv")
    rows: list[dict] = []
    methods = [
        "Zero-shot",
        "History mean",
        "Raw residual",
        "Capped History mean",
        "Capped Raw residual",
        "SRCS",
    ]
    for (k, method), frame in region_summary.loc[
        region_summary["method"].isin(methods)
    ].groupby(["k", "method"]):
        rows.append(
            {
                "k": int(k),
                "method": method,
                "risk_budget": 0.12 if method == "SRCS" else np.nan,
                "metric_scope": "Observed held-region results",
                "region_balanced_sample_mae": float(frame["mae"].mean()),
                "region_balanced_system_round_mae": float(
                    frame["system_round_balanced_mae"].mean()
                ),
                "mean_region_negative_transfer": float(frame["negative_transfer_rate"].mean()),
                "worst_region_negative_transfer": float(frame["negative_transfer_rate"].max()),
                "mean_region_cvar90": float(frame["cvar90_regret"].mean()),
                "worst_region_cvar90": float(frame["cvar90_regret"].max()),
                "maximum_regret": float(
                    system_summary.loc[
                        (system_summary["k"] == k)
                        & (system_summary["method"] == method),
                        "delta",
                    ].max()
                ),
                "mean_region_adaptation_rate": np.nan,
            }
        )
    zero = region_summary.loc[region_summary["method"] == "Zero-shot", [
        "k",
        REGION,
        "system_round_balanced_mae",
    ]].rename(columns={"system_round_balanced_mae": "zero_system_mae"})
    sensitivity = sensitivity.merge(
        zero,
        left_on=["k", "outer_target_region"],
        right_on=["k", REGION],
        how="left",
        validate="many_to_one",
    )
    sensitivity["method_system_mae"] = (
        sensitivity["zero_system_mae"] + sensitivity["target_mean_delta"]
    )
    for (k, budget), frame in sensitivity.groupby(["k", "risk_budget"]):
        rows.append(
            {
                "k": int(k),
                "method": f"SRCS risk budget {100 * budget:.0f}%",
                "risk_budget": float(budget),
                "metric_scope": "Observed held-region sensitivity; region-balanced",
                "region_balanced_sample_mae": np.nan,
                "region_balanced_system_round_mae": float(frame["method_system_mae"].mean()),
                "mean_region_negative_transfer": float(frame["target_negative_transfer"].mean()),
                "worst_region_negative_transfer": float(frame["target_negative_transfer"].max()),
                "mean_region_cvar90": float(frame["target_cvar90"].mean()),
                "worst_region_cvar90": float(frame["target_cvar90"].max()),
                "maximum_regret": float(frame["target_max_regret"].max()),
                "mean_region_adaptation_rate": float(frame["target_adaptation_rate"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "method"]).reset_index(drop=True)


def regional_failure_modes(tables: Path) -> pd.DataFrame:
    region_summary = pd.read_csv(tables / "us_region_summary.csv")
    system_summary = pd.read_csv(tables / "us_system_summary.csv")
    tail = (
        system_summary.groupby(["k", REGION, "method"])["delta"]
        .agg(
            p95_regret=lambda values: float(np.quantile(values, 0.95)),
            max_regret="max",
        )
        .reset_index()
    )
    region_summary = region_summary.merge(
        tail,
        on=["k", REGION, "method"],
        how="left",
        validate="one_to_one",
    )
    metrics = [
        "mae",
        "system_round_balanced_mae",
        "negative_transfer_rate",
        "mean_regret",
        "cvar90_regret",
        "p95_regret",
        "max_regret",
    ]
    selected = region_summary.loc[
        region_summary["method"].isin(["SRCS", "History mean", "Capped History mean"])
    ]
    pivot = selected.pivot(index=["k", REGION], columns="method", values=metrics)
    pivot.columns = [
        f"{metric}__{method.replace(' ', '_').lower()}" for metric, method in pivot.columns
    ]
    output = pivot.reset_index()
    output["sample_mae_delta_vs_history"] = (
        output["mae__srcs"] - output["mae__history_mean"]
    )
    output["system_mae_delta_vs_history"] = (
        output["system_round_balanced_mae__srcs"]
        - output["system_round_balanced_mae__history_mean"]
    )
    output["sample_mae_delta_vs_capped_history"] = (
        output["mae__srcs"] - output["mae__capped_history_mean"]
    )
    coverage = pd.read_csv(tables / "us_coverage_detail.csv")
    coverage = coverage.loc[coverage["level"] == "region", ["k", "unit", "coverage"]].copy()
    coverage[REGION] = pd.to_numeric(coverage["unit"], errors="raise").astype(int)
    output = output.merge(
        coverage[["k", REGION, "coverage"]],
        on=["k", REGION],
        how="left",
        validate="one_to_one",
    )
    eligibility = pd.read_csv(tables / "eligibility_selection_bias.csv")
    eligibility[REGION] = pd.to_numeric(eligibility[REGION], errors="coerce")
    eligibility = eligibility.loc[eligibility[REGION].notna()].copy()
    eligibility[REGION] = eligibility[REGION].astype(int)
    output = output.merge(
        eligibility[["k", REGION, "eligibility_rate"]],
        on=["k", REGION],
        how="left",
        validate="one_to_one",
    )
    return output.sort_values(["k", REGION]).reset_index(drop=True)


def action_frequencies(tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(tables / "us_predictions.csv", low_memory=False)
    decisions = predictions[[
        "k",
        REGION,
        GROUP,
        "selected_action",
        "selected_family",
        "selected_shrink",
        "adapted",
    ]].drop_duplicates()
    if decisions.duplicated(["k", GROUP]).any():
        raise AssertionError("Multiple final decisions for a U.S. system/k pair")
    overall = (
        decisions.groupby(
            ["k", "selected_action", "selected_family", "selected_shrink", "adapted"],
            dropna=False,
        )
        .size()
        .rename("systems")
        .reset_index()
    )
    totals = decisions.groupby("k").size().rename("eligible_systems")
    overall = overall.merge(totals, on="k", validate="many_to_one")
    overall["system_fraction"] = overall["systems"] / overall["eligible_systems"]
    regional = (
        decisions.groupby(
            ["k", REGION, "selected_action", "selected_family", "selected_shrink", "adapted"],
            dropna=False,
        )
        .size()
        .rename("systems")
        .reset_index()
    )
    totals_region = decisions.groupby(["k", REGION]).size().rename("eligible_systems")
    regional = regional.merge(totals_region, on=["k", REGION], validate="many_to_one")
    regional["system_fraction"] = regional["systems"] / regional["eligible_systems"]
    return overall, regional


def uk_plant_changes(tables: Path) -> pd.DataFrame:
    systems = pd.read_csv(tables / "uk_system_summary.csv")
    methods = ["Zero-shot", "Persistence", "History mean", "Raw residual", "SRCS"]
    selected = systems.loc[systems["method"].isin(methods)]
    pivot = selected.pivot(
        index=["k", REGION, GROUP], columns="method", values="method_abs"
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(
        columns={
            "Zero-shot": "zero_shot_mae",
            "Persistence": "persistence_mae",
            "History mean": "history_mean_mae",
            "Raw residual": "raw_residual_mae",
            "SRCS": "srcs_mae",
        },
        inplace=True,
    )
    pivot["srcs_delta_vs_zero"] = pivot["srcs_mae"] - pivot["zero_shot_mae"]
    pivot["srcs_delta_vs_history"] = pivot["srcs_mae"] - pivot["history_mean_mae"]
    pivot["srcs_delta_vs_persistence"] = pivot["srcs_mae"] - pivot["persistence_mae"]
    pivot["srcs_negative_transfer"] = pivot["srcs_delta_vs_zero"] > 1e-12
    return pivot.sort_values(["k", GROUP]).reset_index(drop=True)


def safety_mode_comparison(
    optimized_tables: Path,
    historical_tables: Path,
    analysis_dir: Path,
    n_boot: int,
    safety_budget: float = 0.06,
) -> pd.DataFrame:
    variant_summary = pd.read_csv(analysis_dir / "risk_budget_full_summary.csv")
    variant_systems = pd.read_csv(analysis_dir / "risk_budget_system_results.csv")
    historical_predictions = pd.read_csv(
        historical_tables / "core_predictions.csv", low_memory=False
    )
    historical_comparison = optimization_comparison(
        optimized_tables, historical_tables
    )
    rows = []
    for k in (1, 2, 3):
        new_summary = variant_summary.loc[
            (variant_summary["k"] == k)
            & np.isclose(variant_summary["risk_budget"], safety_budget)
        ].iloc[0]
        old_summary = historical_comparison.loc[
            (historical_comparison["version"] == "Historical v2")
            & (historical_comparison["k"] == k)
            & (historical_comparison["method"] == "SafeShrink HRC")
        ].iloc[0]
        old_systems = system_round_errors(
            historical_predictions.loc[historical_predictions["k"] == k],
            "SafeShrink HRC",
        ).set_index([REGION, GROUP])
        new_systems = variant_systems.loc[
            (variant_systems["k"] == k)
            & np.isclose(variant_systems["risk_budget"], safety_budget)
        ].set_index([REGION, GROUP])
        paired = new_systems[["method_mae"]].rename(
            columns={"method_mae": "srcs_safety_mae"}
        ).join(
            old_systems[["method_abs"]].rename(
                columns={"method_abs": "historical_safe_mae"}
            ),
            how="inner",
            validate="one_to_one",
        )
        paired["delta"] = (
            paired["srcs_safety_mae"] - paired["historical_safe_mae"]
        )
        paired = paired.reset_index()
        inference = hierarchical_paired_bootstrap(
            paired, n_boot, SEED + 6000 + k
        )
        dominance = bool(
            new_summary["mae"] <= old_summary["mae"]
            and new_summary["system_round_balanced_mae"]
            <= old_summary["system_round_balanced_mae"]
            and new_summary["negative_transfer_rate"]
            <= old_summary["negative_transfer_equal_round"]
            and new_summary["strict_cvar90_regret"]
            <= old_summary["strict_cvar90_regret"]
            and new_summary["p95_regret"] <= old_summary["p95_regret"]
            and new_summary["max_regret"] <= old_summary["max_regret"]
            and new_summary["coverage_90"] >= old_summary["coverage_90"]
        )
        rows.append(
            {
                "k": k,
                "safety_budget": safety_budget,
                "new_method": "SRCS safety mode (6% source risk budget)",
                "comparator": "Historical SafeShrink HRC",
                "new_sample_mae": new_summary["mae"],
                "old_sample_mae": old_summary["mae"],
                "sample_mae_difference": new_summary["mae"] - old_summary["mae"],
                "new_system_round_mae": new_summary["system_round_balanced_mae"],
                "old_system_round_mae": old_summary["system_round_balanced_mae"],
                "new_negative_transfer": new_summary["negative_transfer_rate"],
                "old_negative_transfer": old_summary["negative_transfer_equal_round"],
                "new_strict_cvar90": new_summary["strict_cvar90_regret"],
                "old_strict_cvar90": old_summary["strict_cvar90_regret"],
                "new_p95_regret": new_summary["p95_regret"],
                "old_p95_regret": old_summary["p95_regret"],
                "new_max_regret": new_summary["max_regret"],
                "old_max_regret": old_summary["max_regret"],
                "new_coverage_90": new_summary["coverage_90"],
                "old_coverage_90": old_summary["coverage_90"],
                "dominates_all_reported_metrics": dominance,
                "selection_status": (
                    "Post-hoc exploratory operating point; held regions remained excluded "
                    "during each source policy fit"
                ),
                **{
                    f"paired_system_mae_{key}": value
                    for key, value in inference.items()
                },
            }
        )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, output: Path, name: str) -> None:
    frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze locked HAA6Br SRCS results")
    parser.add_argument(
        "--optimized-output",
        type=Path,
        required=True,
        help="Output directory created by run_optimized_experiments.py",
    )
    parser.add_argument(
        "--historical-output",
        type=Path,
        required=True,
        help="Output directory created by run_new_experiments.py",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimized = args.optimized_output.resolve()
    historical = args.historical_output.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else optimized / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    optimized_tables = optimized / "tables"
    historical_tables = historical / "tables"
    required = [
        optimized_tables / "us_predictions.csv",
        optimized_tables / "uk_predictions.csv",
        optimized_tables / "us_region_summary.csv",
        optimized_tables / "risk_budget_sensitivity.csv",
        historical_tables / "core_predictions.csv",
        output / "risk_budget_full_summary.csv",
        output / "risk_budget_system_results.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required locked result tables: {missing}")

    contrasts = primary_contrasts(optimized_tables, args.bootstrap)
    comparison = optimization_comparison(optimized_tables, historical_tables)
    frontier = risk_frontier(optimized_tables)
    regional = regional_failure_modes(optimized_tables)
    action_overall, action_region = action_frequencies(optimized_tables)
    uk_plants = uk_plant_changes(optimized_tables)
    risk_summary_path = output / "risk_budget_full_summary.csv"
    risk_system_path = output / "risk_budget_system_results.csv"
    if not risk_summary_path.exists() or not risk_system_path.exists():
        raise FileNotFoundError(
            "Run evaluate_risk_budget_variants.py before final analysis"
        )
    safety_comparison = safety_mode_comparison(
        optimized_tables,
        historical_tables,
        output,
        args.bootstrap,
    )

    for name, frame in {
        "primary_contrasts": contrasts,
        "optimization_comparison": comparison,
        "risk_frontier": frontier,
        "regional_failure_modes": regional,
        "action_frequency_overall": action_overall,
        "action_frequency_region": action_region,
        "uk_plant_changes": uk_plants,
        "safety_mode_comparison": safety_comparison,
    }.items():
        write_csv(frame, output, name)

    metadata = {
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "Post-hoc interpretation of locked predictions; no model retuning",
        "optimized_output": optimized.name,
        "historical_output": historical.name,
        "bootstrap_replicates": int(args.bootstrap),
        "strict_cvar90_definition": "Mean of largest ceil(10% * n_systems) regrets",
        "script_sha256": sha256_file(Path(__file__)),
        "optimized_run_metadata_sha256": sha256_file(optimized / "run_metadata.json"),
        "input_sha256": {path.name: sha256_file(path) for path in required},
        "tables": sorted(path.stem for path in output.glob("*.csv")),
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
