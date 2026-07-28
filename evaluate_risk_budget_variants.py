from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_optimized_results as analysis
import run_optimized_experiments as optimized


GROUP = optimized.GROUP
REGION = optimized.REGION
POLICY = optimized.POLICY_NAME
RISK_BUDGETS = (0.04, 0.06, *optimized.RISK_BUDGETS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_outer_k(
    core: pd.DataFrame,
    candidate_tables: optimized.CandidateTableCache,
    outer_region: int,
    k: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    all_regions = tuple(sorted(int(value) for value in core[REGION].unique()))
    source_regions = tuple(value for value in all_regions if value != outer_region)

    def candidate(
        excluded: tuple[int, ...], predicted_region: int, with_samples: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return candidate_tables.get(excluded, predicted_region, k, with_samples)

    oof_frames = {alpha: [] for alpha in optimized.RIDGE_ALPHAS}
    source_system_frames = []
    source_sample_frames = []
    for held_region in source_regions:
        validation_systems, validation_samples = candidate(
            (outer_region, held_region), held_region, True
        )
        source_system_frames.append(validation_systems)
        source_sample_frames.append(validation_samples)
        training_frames = []
        for pseudo_region in source_regions:
            if pseudo_region == held_region:
                continue
            systems, _ = candidate(
                (outer_region, held_region, pseudo_region), pseudo_region, False
            )
            training_frames.append(systems)
        training = pd.concat(training_frames, ignore_index=True)
        for alpha in optimized.RIDGE_ALPHAS:
            models = optimized.fit_policy_models(training, alpha)
            predicted = optimized.predict_policy(models, validation_systems)
            predicted["policy_held_region"] = held_region
            predicted["outer_target_region"] = outer_region
            oof_frames[alpha].append(predicted)

    oof_by_alpha = {
        alpha: pd.concat(frames, ignore_index=True)
        for alpha, frames in oof_frames.items()
    }
    source_systems = pd.concat(source_system_frames, ignore_index=True)
    source_samples = pd.concat(source_sample_frames, ignore_index=True)
    target_systems, target_samples = candidate((outer_region,), outer_region, True)

    sample_outputs: list[pd.DataFrame] = []
    system_outputs: list[pd.DataFrame] = []
    spec_outputs: list[dict] = []
    final_models: dict[float, dict] = {}
    for budget in RISK_BUDGETS:
        spec, source_decisions, _ = optimized.tune_policy(
            oof_by_alpha, budget, "all"
        )
        if spec.alpha not in final_models:
            final_models[spec.alpha] = optimized.fit_policy_models(
                source_systems, spec.alpha
            )
        target_policy = optimized.predict_policy(
            final_models[spec.alpha], target_systems
        )
        target_decisions = optimized.decisions_from_predictions(
            target_policy, spec.margin, spec.action_set
        )
        selected_source = optimized.selected_sample_predictions(
            source_samples, source_decisions
        )
        q90, q90_sample, q90_cluster = optimized.clustered_interval_quantile(
            selected_source
        )
        selected_target = optimized.selected_sample_predictions(
            target_samples, target_decisions
        )
        selected_target["interval_low"] = np.maximum(
            0.0, selected_target[POLICY] - q90
        )
        selected_target["interval_high"] = selected_target[POLICY] + q90
        selected_target["risk_budget"] = budget
        sample_outputs.append(
            selected_target[
                [
                    "sample_id",
                    GROUP,
                    REGION,
                    "round_index",
                    "k",
                    "observed",
                    "Zero-shot",
                    POLICY,
                    "interval_low",
                    "interval_high",
                    "selected_action",
                    "selected_family",
                    "selected_shrink",
                    "adapted",
                    "risk_budget",
                ]
            ].copy()
        )
        systems = target_decisions.copy()
        systems["risk_budget"] = budget
        systems["method_mae"] = (
            systems["base_mae"] + systems["selected_actual_delta"]
        )
        systems["q90_source"] = q90
        system_outputs.append(systems)
        spec_outputs.append(
            {
                "outer_target_region": outer_region,
                "k": k,
                **spec.__dict__,
                "q90_source": q90,
                "q90_source_sample": q90_sample,
                "q90_source_system_clustered": q90_cluster,
                "source_metrics_role": (
                    "Source tuning constraints; held outer region not used"
                ),
            }
        )
    return sample_outputs, system_outputs, spec_outputs


def summarize_variants(
    samples: pd.DataFrame,
    systems: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    region_rows = []
    for (k, budget), sample_frame in samples.groupby(["k", "risk_budget"]):
        system_frame = systems.loc[
            (systems["k"] == k) & (systems["risk_budget"] == budget)
        ].copy()
        metrics = analysis.regression_metrics(
            sample_frame["observed"], sample_frame[POLICY]
        )
        zero_mae = float(
            np.mean(np.abs(sample_frame["observed"] - sample_frame["Zero-shot"]))
        )
        region_sample_mae = []
        region_system_mae = []
        for region, regional_samples in sample_frame.groupby(REGION):
            regional_systems = system_frame.loc[system_frame[REGION] == region]
            regional_metrics = analysis.regression_metrics(
                regional_samples["observed"], regional_samples[POLICY]
            )
            covered = (
                (regional_samples["observed"] >= regional_samples["interval_low"])
                & (regional_samples["observed"] <= regional_samples["interval_high"])
            )
            region_sample_mae.append(regional_metrics["mae"])
            region_system_mae.append(float(regional_systems["method_mae"].mean()))
            region_rows.append(
                {
                    "k": int(k),
                    "risk_budget": float(budget),
                    REGION: int(region),
                    "rows": int(len(regional_samples)),
                    "systems": int(len(regional_systems)),
                    **regional_metrics,
                    "system_round_balanced_mae": float(
                        regional_systems["method_mae"].mean()
                    ),
                    "negative_transfer_rate": float(
                        regional_systems["negative_transfer"].mean()
                    ),
                    "mean_regret": float(
                        regional_systems["selected_actual_delta"].mean()
                    ),
                    "strict_cvar90_regret": analysis.strict_cvar90(
                        regional_systems["selected_actual_delta"].to_numpy(float)
                    ),
                    "p95_regret": float(
                        np.quantile(
                            regional_systems["selected_actual_delta"], 0.95
                        )
                    ),
                    "max_regret": float(
                        regional_systems["selected_actual_delta"].max()
                    ),
                    "adaptation_rate": float(regional_systems["adapted"].mean()),
                    "coverage_90": float(covered.mean()),
                    "mean_interval_width": float(
                        np.mean(
                            regional_samples["interval_high"]
                            - regional_samples["interval_low"]
                        )
                    ),
                }
            )
        covered = (
            (sample_frame["observed"] >= sample_frame["interval_low"])
            & (sample_frame["observed"] <= sample_frame["interval_high"])
        )
        delta = system_frame["selected_actual_delta"].to_numpy(float)
        summary_rows.append(
            {
                "k": int(k),
                "risk_budget": float(budget),
                "rows": int(len(sample_frame)),
                "systems": int(len(system_frame)),
                **metrics,
                "relative_sample_mae_improvement": (zero_mae - metrics["mae"])
                / max(zero_mae, 1e-12),
                "system_round_balanced_mae": float(
                    system_frame["method_mae"].mean()
                ),
                "region_balanced_sample_mae": float(np.mean(region_sample_mae)),
                "region_balanced_system_round_mae": float(
                    np.mean(region_system_mae)
                ),
                "negative_transfer_rate": float(np.mean(delta > 1e-12)),
                "mean_regret": float(np.mean(delta)),
                "strict_cvar90_regret": analysis.strict_cvar90(delta),
                "p95_regret": float(np.quantile(delta, 0.95)),
                "max_regret": float(np.max(delta)),
                "adaptation_rate": float(system_frame["adapted"].mean()),
                "coverage_90": float(covered.mean()),
                "mean_interval_width": float(
                    np.mean(sample_frame["interval_high"] - sample_frame["interval_low"])
                ),
                "maximum_prediction_shift": float(
                    np.max(np.abs(sample_frame[POLICY] - sample_frame["Zero-shot"]))
                ),
                "evidence_role": (
                    "Post-hoc source-budget sensitivity on held outer-region predictions"
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(region_rows)


def reproduce_primary(
    reconstructed: pd.DataFrame,
    locked_path: Path,
) -> pd.DataFrame:
    locked = pd.read_csv(locked_path, low_memory=False)
    primary = reconstructed.loc[
        np.isclose(reconstructed["risk_budget"], optimized.PRIMARY_RISK_BUDGET)
    ].copy()
    primary.sort_values(["k", "sample_id"], inplace=True)
    locked.sort_values(["k", "sample_id"], inplace=True)
    if not np.array_equal(
        primary[["k", "sample_id"]].to_numpy(),
        locked[["k", "sample_id"]].to_numpy(),
    ):
        raise AssertionError("Reconstructed primary sample keys differ from locked output")
    action_equal = bool(
        np.array_equal(
            primary["selected_action"].to_numpy(),
            locked["selected_action"].to_numpy(),
        )
    )
    prediction_diff = float(
        np.max(np.abs(primary[POLICY].to_numpy(float) - locked[POLICY].to_numpy(float)))
    )
    interval_diff = float(
        max(
            np.max(
                np.abs(
                    primary["interval_low"].to_numpy(float)
                    - locked["interval_low"].to_numpy(float)
                )
            ),
            np.max(
                np.abs(
                    primary["interval_high"].to_numpy(float)
                    - locked["interval_high"].to_numpy(float)
                )
            ),
        )
    )
    passed = action_equal and prediction_diff <= 1e-12 and interval_diff <= 1e-12
    if not passed:
        raise AssertionError(
            "Risk-variant reconstruction did not reproduce the locked 12% primary"
        )
    return pd.DataFrame(
        [
            {
                "sample_keys_equal": True,
                "selected_actions_equal": action_equal,
                "maximum_prediction_difference": prediction_diff,
                "maximum_interval_difference": interval_diff,
                "passed": passed,
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct source-only SRCS risk-budget sensitivity"
    )
    parser.add_argument(
        "--optimized-output",
        type=Path,
        required=True,
        help="Output directory created by run_optimized_experiments.py",
    )
    parser.add_argument(
        "--data-package",
        type=Path,
        required=True,
        help="Path to the same cleaned haa6br_integrated_v1 package used for optimization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.optimized_output.resolve()
    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    data_package = optimized.validate_integrated_v1(
        args.data_package,
        optimized.INTEGRATED_V1_US_FILES,
    )
    paths = optimized.Paths(
        data_package,
        output,
        output / "tables",
        output / "figures",
        output / "locks",
    )
    core, _, _, feature_sets = optimized.load_data(paths)
    features = list(feature_sets["us_operational_core"])
    cache = optimized.RegionalPredictionCache(
        core,
        features,
        optimized.MODEL_NAME,
        output / "cache",
        "us_operational",
    )
    candidates = optimized.CandidateTableCache(cache)

    sample_frames = []
    system_frames = []
    specs = []
    regions = tuple(sorted(int(value) for value in core[REGION].unique()))
    for index, outer_region in enumerate(regions, start=1):
        for k in (1, 2, 3):
            print(
                f"[risk variants {index}/{len(regions)}] EPA {outer_region}, k={k}",
                flush=True,
            )
            samples, systems, outer_specs = reconstruct_outer_k(
                core, candidates, outer_region, k
            )
            sample_frames.extend(samples)
            system_frames.extend(systems)
            specs.extend(outer_specs)

    samples = pd.concat(sample_frames, ignore_index=True)
    systems = pd.concat(system_frames, ignore_index=True)
    summaries, region_summaries = summarize_variants(samples, systems)
    reproduction = reproduce_primary(
        samples, output / "tables" / "us_predictions.csv"
    )

    summaries.to_csv(
        analysis_dir / "risk_budget_full_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_summaries.to_csv(
        analysis_dir / "risk_budget_full_region_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    systems.to_csv(
        analysis_dir / "risk_budget_system_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(specs).to_csv(
        analysis_dir / "risk_budget_policy_specs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reproduction.to_csv(
        analysis_dir / "risk_budget_primary_reproduction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": (
            "Post-hoc risk-budget sensitivity, including exploratory 4% and 6% modes; "
            "each held target region remained excluded from its policy fitting and "
            "source budget selection"
        ),
        "risk_budgets": RISK_BUDGETS,
        "primary_budget": optimized.PRIMARY_RISK_BUDGET,
        "primary_reproduction_passed": bool(reproduction["passed"].iloc[0]),
        "main_script_sha256": sha256_file(Path(optimized.__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "protocol_lock_sha256": sha256_file(
            output / "locks" / "protocol_lock_before_optimized_run.json"
        ),
        "saved_sample_predictions": False,
        "system_rows": int(len(systems)),
    }
    (analysis_dir / "risk_budget_analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
