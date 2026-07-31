from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / ".deps"))

import numpy as np
import pandas as pd

from stage4_revision_analyses import ANALYSIS_STATUS, parse_bool, strict_cvar90


REGION = "epa_region"
GROUP = "group_system_id"
METHODS = (
    "SRCS",
    "History mean",
    "Capped History mean",
    "Raw residual",
    "Capped Raw residual",
    "Zero-shot",
)
OVERLAP_VARIANTS = (
    "full_srcs_reproduced",
    "coverage_matched_capped_history_mean",
)
POLICY_LABELS = {
    "full_srcs_reproduced": "SRCS",
    "coverage_matched_capped_history_mean": "Coverage-matched capped History gate",
    "fixed_threshold_capped_raw_mean": "Conservative capped RawMean gate",
}
EPSILON = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def system_method_frame(predictions: pd.DataFrame, k: int) -> pd.DataFrame:
    subset = predictions.loc[predictions["k"].eq(k)].copy()
    if subset.empty:
        raise ValueError(f"No locked US prediction rows for k={k}")

    round_keys = [REGION, GROUP, "round_index"]
    round_frame = subset[round_keys].copy()
    metric_columns: list[str] = []
    observed = subset["observed"].to_numpy(float)
    for method in METHODS:
        predicted = subset[method].to_numpy(float)
        round_frame[f"mae__{method}"] = np.abs(predicted - observed)
        round_frame[f"bias__{method}"] = predicted - observed
        metric_columns.extend([f"mae__{method}", f"bias__{method}"])

    system = (
        round_frame.groupby(round_keys, as_index=False)[metric_columns]
        .mean()
        .groupby([REGION, GROUP], as_index=False)[metric_columns]
        .mean()
    )
    actions = subset.groupby([REGION, GROUP], as_index=False).agg(
        srcs_adapted=("adapted", "first"),
        srcs_selected_action=("selected_action", "first"),
    )
    actions["srcs_adapted"] = parse_bool(actions["srcs_adapted"])
    return system.merge(
        actions,
        on=[REGION, GROUP],
        how="left",
        validate="one_to_one",
    )


def directional_metrics(values: np.ndarray) -> dict[str, float]:
    signed = np.asarray(values, dtype=float)
    underprediction = np.maximum(-signed, 0.0)
    return {
        "equal_system_signed_bias": float(np.mean(signed)),
        "system_underprediction_rate_gt_0": float(np.mean(signed < -EPSILON)),
        "system_underprediction_rate_gt_0_5": float(np.mean(signed < -0.5)),
        "system_underprediction_rate_gt_1_0": float(np.mean(signed < -1.0)),
        "mean_underprediction_magnitude": float(np.mean(underprediction)),
        "worst_decile_underprediction": strict_cvar90(underprediction),
    }


def summarize_directional(
    systems_by_k: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    system_rows: list[pd.DataFrame] = []
    for k, system in systems_by_k.items():
        scopes = {
            "all_evaluable_systems": system,
            "srcs_corrected_systems": system.loc[system["srcs_adapted"]].copy(),
        }
        for scope, frame in scopes.items():
            for method in METHODS:
                bias = frame[f"bias__{method}"].to_numpy(float)
                mae = frame[f"mae__{method}"].to_numpy(float)
                rows.append(
                    {
                        "analysis_status": ANALYSIS_STATUS,
                        "confirmatory_status": "not_confirmatory",
                        "scope": scope,
                        "k": int(k),
                        "method": method,
                        "systems": int(len(frame)),
                        "regions": int(frame[REGION].nunique()),
                        "equal_system_mae": float(np.mean(mae)),
                        **directional_metrics(bias),
                    }
                )

        out = system[[REGION, GROUP, "srcs_adapted"]].copy()
        out["k"] = int(k)
        for method in METHODS:
            out[f"mae__{method}"] = system[f"mae__{method}"]
            out[f"bias__{method}"] = system[f"bias__{method}"]
        system_rows.append(out)
    return pd.DataFrame(rows), pd.concat(system_rows, ignore_index=True)


def selected_predictions(
    joined: pd.DataFrame,
    action_column: str = "selected_action",
) -> np.ndarray:
    selected = joined["Zero-shot"].to_numpy(float).copy()
    actions = joined[action_column].fillna("Zero-shot").astype(str)
    for action in sorted(actions.unique()):
        if action == "Zero-shot":
            continue
        if action not in joined.columns:
            raise ValueError(f"Selected action is absent from locked predictions: {action}")
        mask = actions.eq(action).to_numpy()
        selected[mask] = joined.loc[mask, action].to_numpy(float)
    return selected


def derive_policy_system_bias(
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        REGION,
        GROUP,
        "k",
        "variant",
        "variant_abs",
        "regret",
        "adapted",
        "selected_action",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"Policy decisions are missing columns: {missing}")

    unique = decisions[list(required)].drop_duplicates(
        [REGION, GROUP, "k", "variant"]
    )
    unique["adapted"] = parse_bool(unique["adapted"])
    unique = unique.rename(
        columns={
            "selected_action": "policy_selected_action",
            "adapted": "policy_adapted",
        }
    )
    joined = predictions.merge(
        unique,
        on=[REGION, GROUP, "k"],
        how="inner",
        validate="many_to_many",
    )
    chosen = selected_predictions(joined, "policy_selected_action")
    observed = joined["observed"].to_numpy(float)
    joined["derived_signed_error"] = chosen - observed
    joined["derived_absolute_error"] = np.abs(chosen - observed)

    round_keys = ["variant", REGION, GROUP, "k", "round_index"]
    round_metrics = (
        joined.groupby(round_keys, as_index=False)[
            ["derived_signed_error", "derived_absolute_error"]
        ]
        .mean()
    )
    system_metrics = (
        round_metrics.groupby(["variant", REGION, GROUP, "k"], as_index=False)[
            ["derived_signed_error", "derived_absolute_error"]
        ]
        .mean()
        .rename(
            columns={
                "derived_signed_error": "signed_bias",
                "derived_absolute_error": "derived_mae",
            }
        )
    )
    output = system_metrics.merge(
        unique,
        on=["variant", REGION, GROUP, "k"],
        how="left",
        validate="one_to_one",
    )
    maximum_difference = float(
        np.max(np.abs(output["derived_mae"] - output["variant_abs"]))
    )
    if maximum_difference > 1e-10:
        raise AssertionError(
            "Policy signed-error reconstruction changed frozen MAE: "
            f"maximum difference={maximum_difference:.3e}"
        )
    output = output.rename(
        columns={
            "policy_selected_action": "selected_action",
            "policy_adapted": "adapted",
        }
    )
    output["policy_label"] = output["variant"].map(POLICY_LABELS)
    return output


def overlap_tables(
    policy_systems: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = policy_systems.loc[
        policy_systems["variant"].isin(OVERLAP_VARIANTS)
    ].copy()
    decision = subset.pivot(
        index=[REGION, GROUP, "k"], columns="variant", values="adapted"
    ).reset_index()
    if decision[list(OVERLAP_VARIANTS)].isna().any().any():
        raise AssertionError("Coverage-overlap decisions are incomplete")
    srcs = decision[OVERLAP_VARIANTS[0]].astype(bool)
    gate = decision[OVERLAP_VARIANTS[1]].astype(bool)
    decision["overlap_group"] = np.select(
        [srcs & gate, srcs & ~gate, ~srcs & gate],
        ["both_correct", "srcs_only", "gate_only"],
        default="both_fallback",
    )

    system_rows = subset.merge(
        decision[[REGION, GROUP, "k", "overlap_group"]],
        on=[REGION, GROUP, "k"],
        how="left",
        validate="many_to_one",
    )
    summary_rows: list[dict[str, object]] = []
    group_order = ("both_correct", "srcs_only", "gate_only", "both_fallback")
    for k in (1, 2, 3):
        total = int(decision["k"].eq(k).sum())
        for group in group_order:
            group_frame = system_rows.loc[
                system_rows["k"].eq(k)
                & system_rows["overlap_group"].eq(group)
            ]
            systems = int(group_frame[[REGION, GROUP]].drop_duplicates().shape[0])
            for variant in OVERLAP_VARIANTS:
                frame = group_frame.loc[group_frame["variant"].eq(variant)]
                regret = frame["regret"].to_numpy(float)
                bias = frame["signed_bias"].to_numpy(float)
                summary_rows.append(
                    {
                        "analysis_status": ANALYSIS_STATUS,
                        "confirmatory_status": "not_confirmatory",
                        "k": int(k),
                        "overlap_group": group,
                        "systems": systems,
                        "share_of_evaluable_systems": systems / total,
                        "variant": variant,
                        "policy": POLICY_LABELS[variant],
                        "equal_system_mae": float(frame["variant_abs"].mean()),
                        "negative_transfer_rate": float(np.mean(regret > EPSILON)),
                        "strict_cvar90_regret": strict_cvar90(regret),
                        **directional_metrics(bias),
                    }
                )

    agreement_rows: list[dict[str, object]] = []
    group_count_rows: list[dict[str, object]] = []
    for k in (1, 2, 3):
        frame = decision.loc[decision["k"].eq(k)]
        srcs = frame[OVERLAP_VARIANTS[0]].astype(bool).to_numpy()
        gate = frame[OVERLAP_VARIANTS[1]].astype(bool).to_numpy()
        union = int(np.sum(srcs | gate))
        intersection = int(np.sum(srcs & gate))
        agreement_rows.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "k": int(k),
                "systems": int(len(frame)),
                "decision_agreement_rate": float(np.mean(srcs == gate)),
                "decision_disagreement_rate": float(np.mean(srcs != gate)),
                "correction_set_jaccard": intersection / union if union else 1.0,
                "both_correct_systems": intersection,
                "correction_union_systems": union,
            }
        )
        counts = frame["overlap_group"].value_counts()
        for group in group_order:
            count = int(counts.get(group, 0))
            group_count_rows.append(
                {
                    "analysis_status": ANALYSIS_STATUS,
                    "k": int(k),
                    "overlap_group": group,
                    "systems": count,
                    "share_of_evaluable_systems": count / len(frame),
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(agreement_rows),
        pd.DataFrame(group_count_rows),
    )


def risk_budget_directional(
    predictions: pd.DataFrame,
    risk_systems: pd.DataFrame,
) -> pd.DataFrame:
    decisions = risk_systems[
        [REGION, GROUP, "k", "risk_budget", "selected_action", "adapted"]
    ].copy()
    decisions["adapted"] = parse_bool(decisions["adapted"])
    decisions = decisions.rename(
        columns={
            "selected_action": "policy_selected_action",
            "adapted": "policy_adapted",
        }
    )
    joined = predictions.merge(
        decisions,
        on=[REGION, GROUP, "k"],
        how="inner",
        validate="many_to_many",
    )
    chosen = selected_predictions(joined, "policy_selected_action")
    joined["signed_error"] = chosen - joined["observed"].to_numpy(float)
    round_bias = (
        joined.groupby(
            ["risk_budget", REGION, GROUP, "k", "round_index"],
            as_index=False,
        )["signed_error"]
        .mean()
    )
    system_bias = (
        round_bias.groupby(
            ["risk_budget", REGION, GROUP, "k"], as_index=False
        )["signed_error"]
        .mean()
        .rename(columns={"signed_error": "signed_bias"})
    )
    rows: list[dict[str, object]] = []
    for (k, budget), frame in system_bias.groupby(["k", "risk_budget"]):
        rows.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "k": int(k),
                "risk_budget": float(budget),
                "systems": int(len(frame)),
                **directional_metrics(frame["signed_bias"].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "risk_budget"])


def gate_directional(policy_systems: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, k), frame in policy_systems.groupby(["variant", "k"]):
        if variant not in POLICY_LABELS:
            continue
        rows.append(
            {
                "variant": variant,
                "series": POLICY_LABELS[variant],
                "k": int(k),
                "systems": int(len(frame)),
                **directional_metrics(frame["signed_bias"].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def assemble_risk_coverage_directional(
    existing: pd.DataFrame,
    budget_metrics: pd.DataFrame,
    gate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    output = existing.copy()
    directional_columns = [
        "equal_system_signed_bias",
        "system_underprediction_rate_gt_0",
        "system_underprediction_rate_gt_0_5",
        "system_underprediction_rate_gt_1_0",
        "mean_underprediction_magnitude",
        "worst_decile_underprediction",
    ]
    for column in directional_columns:
        output[column] = np.nan

    path_mask = output["series"].eq("SRCS budget path")
    path = output.loc[path_mask].drop(columns=directional_columns).merge(
        budget_metrics[["k", "risk_budget", *directional_columns]],
        on=["k", "risk_budget"],
        how="left",
        validate="one_to_one",
    )
    output.loc[path_mask, directional_columns] = path[directional_columns].to_numpy()

    for series in (
        "Coverage-matched capped History gate",
        "Conservative capped RawMean gate",
    ):
        mask = output["series"].eq(series)
        metrics = gate_metrics.loc[gate_metrics["series"].eq(series)]
        merged = output.loc[mask].drop(columns=directional_columns).merge(
            metrics[["k", *directional_columns]],
            on="k",
            how="left",
            validate="one_to_one",
        )
        output.loc[mask, directional_columns] = merged[directional_columns].to_numpy()

    if output[directional_columns].isna().any().any():
        raise AssertionError("Risk-coverage directional metrics are incomplete")
    return output


def directional_figure_data(directional: pd.DataFrame) -> pd.DataFrame:
    all_systems = directional.loc[
        directional["scope"].eq("all_evaluable_systems")
    ].copy()
    all_systems["display_method"] = all_systems["method"]
    corrected = directional.loc[
        directional["scope"].eq("srcs_corrected_systems")
        & directional["method"].eq("SRCS")
    ].copy()
    corrected["display_method"] = "SRCS corrected only"
    output = pd.concat([all_systems, corrected], ignore_index=True)
    order = {
        "SRCS": 1,
        "SRCS corrected only": 2,
        "History mean": 3,
        "Capped History mean": 4,
        "Raw residual": 5,
        "Capped Raw residual": 6,
        "Zero-shot": 7,
    }
    output["display_order"] = output["display_method"].map(order)
    return output[
        [
            "analysis_status",
            "confirmatory_status",
            "scope",
            "k",
            "display_method",
            "display_order",
            "systems",
            "equal_system_signed_bias",
            "system_underprediction_rate_gt_0",
            "system_underprediction_rate_gt_0_5",
            "system_underprediction_rate_gt_1_0",
            "mean_underprediction_magnitude",
            "worst_decile_underprediction",
        ]
    ].sort_values(["display_order", "k"])


def run(
    predictions_path: Path,
    coverage_path: Path,
    risk_systems_path: Path,
    risk_coverage_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    inputs = [
        predictions_path,
        coverage_path,
        risk_systems_path,
        risk_coverage_path,
    ]
    before_hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path, low_memory=False)
    coverage = pd.read_csv(coverage_path, low_memory=False)
    risk_systems = pd.read_csv(risk_systems_path, low_memory=False)
    risk_coverage = pd.read_csv(risk_coverage_path, low_memory=False)

    systems_by_k = {k: system_method_frame(predictions, k) for k in (1, 2, 3)}
    directional, directional_systems = summarize_directional(systems_by_k)
    policy_systems = derive_policy_system_bias(predictions, coverage)
    overlap, agreement, overlap_counts = overlap_tables(policy_systems)
    budget_metrics = risk_budget_directional(predictions, risk_systems)
    gates = gate_directional(policy_systems)
    risk_directional = assemble_risk_coverage_directional(
        risk_coverage,
        budget_metrics,
        gates,
    )
    directional_figure = directional_figure_data(directional)

    locked_bias = directional.loc[
        directional["scope"].eq("all_evaluable_systems")
        & directional["method"].eq("SRCS"),
        ["k", "equal_system_signed_bias"],
    ]
    frontier_bias = risk_directional.loc[
        risk_directional["series"].eq("SRCS budget path")
        & np.isclose(risk_directional["risk_budget"], 0.12),
        ["k", "equal_system_signed_bias"],
    ]
    bias_check = locked_bias.merge(frontier_bias, on="k", suffixes=("_locked", "_frontier"))
    max_bias_difference = float(
        np.max(
            np.abs(
                bias_check["equal_system_signed_bias_locked"]
                - bias_check["equal_system_signed_bias_frontier"]
            )
        )
    )
    if max_bias_difference > 1e-10:
        raise AssertionError(
            "The 12% frontier signed bias does not reproduce locked SRCS: "
            f"maximum difference={max_bias_difference:.3e}"
        )

    output_files = {
        "directional_underprediction_summary.csv": directional,
        "directional_underprediction_systems.csv": directional_systems,
        "strategy_overlap_policy_summary.csv": overlap,
        "strategy_overlap_agreement.csv": agreement,
        "strategy_overlap_counts.csv": overlap_counts,
        "policy_system_directional_outcomes.csv": policy_systems,
        "risk_budget_directional_summary.csv": budget_metrics,
        "risk_coverage_directional_figure_data.csv": risk_directional,
        "directional_underprediction_figure_data.csv": directional_figure,
    }
    for name, frame in output_files.items():
        write_csv(frame, output_dir / name)

    after_hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    if before_hashes != after_hashes:
        raise AssertionError("A protected input changed during final reviewer analysis")

    metadata = {
        "analysis_status": ANALYSIS_STATUS,
        "confirmatory_status": "not_confirmatory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "definitions": {
            "system_signed_bias": (
                "prediction minus observation, averaged within future monitoring "
                "round and then equally across future rounds within system"
            ),
            "system_underprediction_rate_gt_delta": (
                "fraction of systems with system signed bias below -delta ug/L"
            ),
            "worst_decile_underprediction": (
                "mean of the largest ceil(10% * n) values of max(-system signed bias, 0)"
            ),
            "overlap_groups": (
                "both correct, SRCS only corrects, coverage-matched capped-History "
                "gate only corrects, and both fall back"
            ),
        },
        "input_sha256": before_hashes,
        "output_sha256": {
            name: sha256_file(output_dir / name) for name in output_files
        },
        "validation": {
            "protected_inputs_unchanged": True,
            "policy_mae_reconstructed_within": 1e-10,
            "locked_12pct_srcs_bias_max_difference": max_bias_difference,
            "directional_summary_rows": int(len(directional)),
            "overlap_summary_rows": int(len(overlap)),
            "risk_coverage_rows": int(len(risk_directional)),
            "directional_figure_rows": int(len(directional_figure)),
        },
    }
    metadata_path = output_dir / "final_reviewer_revision_analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    optimized = PROJECT_DIR / "outputs" / "optimized_srcs_strict_v4_20260728"
    reviewer2 = (
        PROJECT_DIR
        / "outputs"
        / "stage4_revision_20260730"
        / "reviewer2_analyses"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=optimized / "tables" / "us_predictions.csv",
    )
    parser.add_argument(
        "--coverage-outcomes",
        type=Path,
        default=reviewer2 / "analysis" / "coverage_matched_system_outcomes.csv",
    )
    parser.add_argument(
        "--risk-systems",
        type=Path,
        default=optimized / "analysis" / "risk_budget_system_results.csv",
    )
    parser.add_argument(
        "--risk-coverage",
        type=Path,
        default=reviewer2 / "analysis" / "risk_coverage_figure_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "final_reviewer_revision_20260731",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run(
        predictions_path=args.predictions,
        coverage_path=args.coverage_outcomes,
        risk_systems_path=args.risk_systems,
        risk_coverage_path=args.risk_coverage,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": "PASS", **metadata["validation"]}, indent=2))


if __name__ == "__main__":
    main()
