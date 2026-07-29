from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / ".deps"))

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import t as student_t

import run_optimized_experiments as optimized
from run_new_experiments import GROUP, REGION, Paths, clip_prediction, load_data, sha256_file
from stage4_revision_analyses import (
    ANALYSIS_STATUS,
    ensure_output_is_separate,
    exact_sign_flip_p,
    strict_cvar90,
    validate_integrated_package,
    validate_locked_output,
)


PRIMARY_BUDGET = 0.12
VARIANTS = (
    "full_srcs_reproduced",
    "zero_margin_gate",
    "forced_action_no_abstention",
    "source_utility_without_risk_constraints",
    "cap_removed_at_application_same_selector",
)


class ReadOnlyCandidateTableCache:
    """Read locked candidate tables without writing to the strict-v4 directory."""

    def __init__(
        self,
        prediction_cache: optimized.RegionalPredictionCache,
        locked_candidate_dir: Path,
        fallback_dir: Path,
    ) -> None:
        self.prediction_cache = prediction_cache
        self.locked_candidate_dir = locked_candidate_dir
        self.fallback_dir = fallback_dir
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        self.memory: dict[
            tuple[tuple[int, ...], int, int, bool],
            tuple[pd.DataFrame, pd.DataFrame],
        ] = {}

    @staticmethod
    def _stem(excluded: tuple[int, ...], predicted_region: int, k: int) -> str:
        token = "-".join(map(str, excluded))
        return f"exclude_{token}__predict_{predicted_region}__k{k}"

    def get(
        self,
        excluded_regions: Iterable[int],
        predicted_region: int,
        k: int,
        with_samples: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        excluded = tuple(sorted(int(value) for value in excluded_regions))
        key = (excluded, int(predicted_region), int(k), bool(with_samples))
        if key in self.memory:
            return self.memory[key]

        self.prediction_cache.ensure(excluded)
        stem = self._stem(excluded, int(predicted_region), int(k))
        source_system = self.locked_candidate_dir / f"{stem}__systems.pkl"
        source_sample = self.locked_candidate_dir / f"{stem}__samples.pkl"
        local_system = self.fallback_dir / f"{stem}__systems.pkl"
        local_sample = self.fallback_dir / f"{stem}__samples.pkl"

        system_path = source_system if source_system.exists() else local_system
        sample_path = source_sample if source_sample.exists() else local_sample
        if system_path.exists() and (not with_samples or sample_path.exists()):
            systems = pd.read_pickle(system_path)
            samples = pd.read_pickle(sample_path) if with_samples else pd.DataFrame()
        else:
            predicted = self.prediction_cache.predicted_frame(
                excluded,
                int(predicted_region),
            )
            systems, samples = optimized.build_candidate_dataset(
                predicted,
                int(k),
                with_samples,
            )
            systems.to_pickle(local_system)
            if with_samples:
                samples.to_pickle(local_sample)

        self.memory[key] = (systems, samples)
        if with_samples:
            self.memory[(excluded, int(predicted_region), int(k), False)] = (
                systems,
                pd.DataFrame(),
            )
        return systems, samples


def source_utility_spec(search: pd.DataFrame) -> dict[str, float]:
    work = search.loc[search["action_set"].eq("all")].copy()
    work = work.drop_duplicates(["alpha", "margin"])
    work["negative_adaptation_rate"] = -work["adaptation_rate"]
    work.sort_values(
        [
            "mean_delta",
            "cvar90",
            "negative_transfer",
            "negative_adaptation_rate",
            "alpha",
            "margin",
        ],
        inplace=True,
    )
    winner = work.iloc[0]
    return {
        "alpha": float(winner["alpha"]),
        "margin": float(winner["margin"]),
        "source_mean_delta": float(winner["mean_delta"]),
        "source_negative_transfer": float(winner["negative_transfer"]),
        "source_cvar90": float(winner["cvar90"]),
        "source_adaptation_rate": float(winner["adaptation_rate"]),
    }


def uncapped_same_selector(
    selected: pd.DataFrame,
    target_systems: pd.DataFrame,
) -> pd.DataFrame:
    output = selected.copy()
    output["SRCS"] = output["Zero-shot"].to_numpy(float)
    system_features = target_systems.set_index(GROUP)
    mean_residual = output[GROUP].map(system_features["mean_residual"]).to_numpy(float)
    median_residual = output[GROUP].map(system_features["median_residual"]).to_numpy(float)
    base = output["Zero-shot"].to_numpy(float)

    for action, meta in optimized.ACTION_META.items():
        mask = output["selected_action"].eq(action).to_numpy()
        if not mask.any():
            continue
        shrink = float(meta["shrink"])
        family = str(meta["family"])
        if family == "Persistence":
            anchor = output["Baseline__Persistence"].to_numpy(float)
            value = base + shrink * (anchor - base)
        elif family == "HistoryMean":
            anchor = output["Baseline__HistoryMean"].to_numpy(float)
            value = base + shrink * (anchor - base)
        elif family == "HistoryMedian":
            anchor = output["Baseline__HistoryMedian"].to_numpy(float)
            value = base + shrink * (anchor - base)
        elif family == "RawMean":
            value = base + shrink * mean_residual
        elif family == "RawMedian":
            value = base + shrink * median_residual
        else:
            raise AssertionError(f"Unknown action family: {family}")
        output.loc[mask, "SRCS"] = clip_prediction(value[mask])
    return output


def system_variant_frame(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    work = frame[
        [REGION, GROUP, "round_index", "observed", "Zero-shot", "SRCS"]
    ].copy()
    work["base_abs"] = np.abs(work["observed"] - work["Zero-shot"])
    work["variant_abs"] = np.abs(work["observed"] - work["SRCS"])
    system = (
        work.groupby([REGION, GROUP, "round_index"], as_index=False)[
            ["base_abs", "variant_abs"]
        ]
        .mean()
        .groupby([REGION, GROUP], as_index=False)[["base_abs", "variant_abs"]]
        .mean()
    )
    system["regret"] = system["variant_abs"] - system["base_abs"]
    decisions = frame.groupby([REGION, GROUP], as_index=False).agg(
        adapted=("adapted", "first"),
        selected_action=("selected_action", "first"),
    )
    system = system.merge(
        decisions,
        on=[REGION, GROUP],
        how="left",
        validate="one_to_one",
    )
    system["adapted"] = system["adapted"].astype(bool)
    system["variant"] = variant
    return system


def summarize_variant(system: pd.DataFrame, k: int, variant: str) -> dict[str, object]:
    regrets = system["regret"].to_numpy(float)
    return {
        "analysis_status": ANALYSIS_STATUS,
        "confirmatory_status": "not_confirmatory",
        "k": int(k),
        "variant": variant,
        "systems": int(len(system)),
        "regions": int(system[REGION].nunique()),
        "equal_system_mae": float(system["variant_abs"].mean()),
        "mean_regret_vs_zero_shot": float(np.mean(regrets)),
        "negative_transfer_rate": float(np.mean(regrets > 1e-12)),
        "strict_cvar90_regret": strict_cvar90(regrets),
        "p95_regret": float(np.quantile(regrets, 0.95)),
        "maximum_regret": float(np.max(regrets)),
        "adaptation_rate": float(system["adapted"].mean()),
    }


def summarize_contrasts(system_tables: dict[str, pd.DataFrame], k: int) -> pd.DataFrame:
    full = system_tables["full_srcs_reproduced"].set_index(GROUP)
    rows = []
    for variant, frame in system_tables.items():
        if variant == "full_srcs_reproduced":
            continue
        comparator = frame.set_index(GROUP)
        if set(full.index) != set(comparator.index):
            raise AssertionError(f"Mechanism variant cohort mismatch: {variant}, k={k}")
        mae_difference = full["variant_abs"] - comparator["variant_abs"]
        negative_difference = (
            (full["regret"] > 1e-12).astype(float)
            - (comparator["regret"] > 1e-12).astype(float)
        )
        rows.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "k": int(k),
                "method": "full_srcs_reproduced",
                "comparator": variant,
                "direction": "negative values favor full SRCS",
                "systems": int(len(full)),
                "equal_system_mae_difference": float(mae_difference.mean()),
                "negative_transfer_rate_difference": float(
                    negative_difference.mean()
                ),
                "strict_cvar90_regret_difference": strict_cvar90(
                    full["regret"].to_numpy(float)
                )
                - strict_cvar90(comparator["regret"].to_numpy(float)),
                "p95_regret_difference": float(
                    np.quantile(full["regret"], 0.95)
                    - np.quantile(comparator["regret"], 0.95)
                ),
                "maximum_regret_difference": float(
                    full["regret"].max() - comparator["regret"].max()
                ),
                "adaptation_rate_difference": float(
                    full["adapted"].mean() - comparator["adapted"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def small_cluster_mechanism_sensitivity(
    regional_contrasts: pd.DataFrame,
) -> pd.DataFrame:
    metrics = (
        "equal_system_mae_difference",
        "negative_transfer_rate_difference",
        "strict_cvar90_regret_difference",
        "p95_regret_difference",
        "maximum_regret_difference",
        "adaptation_rate_difference",
    )
    rows = []
    for (k, comparator), frame in regional_contrasts.groupby(["k", "comparator"]):
        for metric in metrics:
            values = frame[metric].to_numpy(float)
            n_regions = len(values)
            mean_effect = float(np.mean(values))
            if n_regions > 1:
                standard_error = float(np.std(values, ddof=1) / np.sqrt(n_regions))
                critical = float(student_t.ppf(0.975, df=n_regions - 1))
                ci_low = mean_effect - critical * standard_error
                ci_high = mean_effect + critical * standard_error
            else:
                ci_low = np.nan
                ci_high = np.nan
            rows.append(
                {
                    "analysis_status": ANALYSIS_STATUS,
                    "confirmatory_status": "not_confirmatory",
                    "target_population": "fixed ten-EPA-region frame",
                    "k": int(k),
                    "method": "full_srcs_reproduced",
                    "comparator": comparator,
                    "metric": metric,
                    "regions": n_regions,
                    "equal_region_mean_effect": mean_effect,
                    "student_t_ci_low": ci_low,
                    "student_t_ci_high": ci_high,
                    "exact_two_sided_sign_flip_p": exact_sign_flip_p(values),
                    "direction": "negative values favor full SRCS",
                    "interpretation": "small-cluster mechanism sensitivity, not confirmatory inference",
                }
            )
    return pd.DataFrame(rows)


def run_mechanism_attribution(
    optimized_output: Path,
    data_package: Path,
    output_dir: Path,
    outer_regions: tuple[int, ...],
    k_values: tuple[int, ...],
) -> dict[str, object]:
    started = time.time()
    optimized_output = optimized_output.resolve()
    data_package = data_package.resolve()
    output_dir = output_dir.resolve()
    ensure_output_is_separate(output_dir, (optimized_output, data_package))
    validate_integrated_package(data_package)
    validate_locked_output(optimized_output)

    locked_prediction_path = optimized_output / "tables" / "us_predictions.csv"
    locked_spec_path = optimized_output / "tables" / "policy_spec.csv"
    locked_search_path = optimized_output / "tables" / "policy_search_full.csv"
    locked_candidate_dir = (
        optimized_output
        / "cache"
        / "us_operational"
        / optimized.MODEL_NAME.replace(" ", "_")
        / f"candidate_tables_{optimized.CANDIDATE_CACHE_VERSION}"
    )
    locked_candidate_manifest = locked_candidate_dir / "manifest.json"
    input_paths = {
        "locked_us_predictions": locked_prediction_path,
        "locked_policy_spec": locked_spec_path,
        "locked_policy_search": locked_search_path,
        "locked_candidate_manifest": locked_candidate_manifest,
        "clean_us_core": data_package / "data" / "us_ucmr4_core.csv",
    }
    input_hashes_before = {name: sha256_file(path) for name, path in input_paths.items()}

    paths = Paths(
        data_package,
        output_dir,
        output_dir / "tables",
        output_dir / "figures",
        output_dir / "locks",
    )
    core, _, _, feature_sets = load_data(paths)
    operational_features = list(feature_sets["us_operational_core"])
    all_regions = tuple(sorted(int(value) for value in core[REGION].unique()))
    if not set(outer_regions).issubset(all_regions):
        raise ValueError(f"Unknown outer regions: {outer_regions}")

    locked_predictions = pd.read_csv(locked_prediction_path, low_memory=False)
    locked_specs = pd.read_csv(locked_spec_path, low_memory=False)
    locked_search = pd.read_csv(locked_search_path, low_memory=False)
    prediction_cache = optimized.RegionalPredictionCache(
        core,
        operational_features,
        optimized.MODEL_NAME,
        output_dir / "cache",
        "us_operational",
        optimized_output / "cache",
    )
    candidate_tables = ReadOnlyCandidateTableCache(
        prediction_cache,
        locked_candidate_dir,
        output_dir / "cache" / "candidate_fallback",
    )

    prediction_outputs = []
    system_outputs = []
    contrast_frames = []
    spec_rows = []
    reproduction_rows = []
    for outer_region in outer_regions:
        source_regions = tuple(region for region in all_regions if region != outer_region)
        for k in k_values:
            print(f"[mechanism] outer={outer_region}, k={k}", flush=True)
            source_frames = []
            for held_region in source_regions:
                systems, _ = candidate_tables.get(
                    (outer_region, held_region),
                    held_region,
                    k,
                    False,
                )
                if not systems.empty:
                    source_frames.append(systems)
            final_source_systems = pd.concat(source_frames, ignore_index=True)
            target_systems, target_samples = candidate_tables.get(
                (outer_region,),
                outer_region,
                k,
                True,
            )

            full_spec_rows = locked_specs.loc[
                locked_specs["outer_target_region"].eq(outer_region)
                & locked_specs["k"].eq(k)
            ]
            if len(full_spec_rows) != 1:
                raise AssertionError(
                    f"Expected one locked policy spec, found {len(full_spec_rows)}"
                )
            full_spec = full_spec_rows.iloc[0]
            search = locked_search.loc[
                locked_search["outer_target_region"].eq(outer_region)
                & locked_search["k"].eq(k)
            ]
            unconstrained = source_utility_spec(search)
            alphas = {float(full_spec["alpha"]), float(unconstrained["alpha"])}
            predictions_by_alpha = {}
            for alpha in sorted(alphas):
                models = optimized.fit_policy_models(final_source_systems, alpha)
                predictions_by_alpha[alpha] = optimized.predict_policy(
                    models,
                    target_systems,
                )

            full_policy_predictions = predictions_by_alpha[float(full_spec["alpha"])]
            decisions = {
                "full_srcs_reproduced": optimized.decisions_from_predictions(
                    full_policy_predictions,
                    float(full_spec["margin"]),
                    "all",
                ),
                "zero_margin_gate": optimized.decisions_from_predictions(
                    full_policy_predictions,
                    0.0,
                    "all",
                ),
                "forced_action_no_abstention": optimized.decisions_from_predictions(
                    full_policy_predictions,
                    -np.inf,
                    "all",
                ),
                "source_utility_without_risk_constraints": optimized.decisions_from_predictions(
                    predictions_by_alpha[float(unconstrained["alpha"])],
                    float(unconstrained["margin"]),
                    "all",
                ),
            }
            for name, decision in decisions.items():
                invariant = optimized.fixed_spec_decision_invariant_to_future_losses(
                    full_policy_predictions
                    if name != "source_utility_without_risk_constraints"
                    else predictions_by_alpha[float(unconstrained["alpha"])],
                    (
                        float(full_spec["margin"])
                        if name == "full_srcs_reproduced"
                        else 0.0
                        if name == "zero_margin_gate"
                        else -np.inf
                        if name == "forced_action_no_abstention"
                        else float(unconstrained["margin"])
                    ),
                    "all",
                    optimized.SEED + 10000 * outer_region + 100 * k + len(name),
                )
                if not invariant:
                    raise AssertionError(f"Future-loss invariance failed: {name}")

            selected_frames = {
                name: optimized.add_baseline_aliases(
                    optimized.selected_sample_predictions(target_samples, decision)
                )
                for name, decision in decisions.items()
            }
            selected_frames["cap_removed_at_application_same_selector"] = (
                uncapped_same_selector(
                    selected_frames["full_srcs_reproduced"],
                    target_systems,
                )
            )

            locked_target = locked_predictions.loc[
                locked_predictions[REGION].eq(outer_region)
                & locked_predictions["k"].eq(k),
                ["sample_id", "SRCS", "selected_action", "adapted"],
            ].copy()
            reproduced = selected_frames["full_srcs_reproduced"][
                ["sample_id", "SRCS", "selected_action", "adapted"]
            ].copy()
            check = locked_target.merge(
                reproduced,
                on="sample_id",
                how="outer",
                suffixes=("_locked", "_reproduced"),
                indicator=True,
                validate="one_to_one",
            )
            maximum_difference = float(
                np.nanmax(np.abs(check["SRCS_locked"] - check["SRCS_reproduced"]))
            )
            action_match = bool(
                check["selected_action_locked"].equals(
                    check["selected_action_reproduced"]
                )
            )
            adapted_match = bool(
                check["adapted_locked"].astype(str).equals(
                    check["adapted_reproduced"].astype(str)
                )
            )
            reproduction_pass = bool(
                check["_merge"].eq("both").all()
                and maximum_difference <= 1e-10
                and action_match
                and adapted_match
            )
            reproduction_rows.append(
                {
                    "outer_target_region": outer_region,
                    "k": k,
                    "rows": len(check),
                    "maximum_prediction_difference": maximum_difference,
                    "selected_action_exact_match": action_match,
                    "adapted_exact_match": adapted_match,
                    "status": "PASS" if reproduction_pass else "FAIL",
                }
            )
            if not reproduction_pass:
                raise AssertionError(
                    f"Locked full-selector reproduction failed: outer={outer_region}, k={k}"
                )

            system_tables = {}
            for variant, selected in selected_frames.items():
                selected = selected.copy()
                selected["variant"] = variant
                selected["outer_target_region"] = outer_region
                minimal_columns = [
                    "sample_id",
                    GROUP,
                    "group_site_id",
                    REGION,
                    "round_index",
                    "k",
                    "observed",
                    "Zero-shot",
                    "SRCS",
                    "selected_action",
                    "adapted",
                    "variant",
                    "outer_target_region",
                ]
                prediction_outputs.append(selected[minimal_columns])
                system = system_variant_frame(selected, variant)
                system["k"] = k
                system["outer_target_region"] = outer_region
                system_tables[variant] = system
                system_outputs.append(system)
            regional_contrasts = summarize_contrasts(system_tables, k)
            regional_contrasts.insert(2, "outer_target_region", outer_region)
            contrast_frames.append(regional_contrasts)
            spec_rows.extend(
                [
                    {
                        "outer_target_region": outer_region,
                        "k": k,
                        "variant": "full_srcs_reproduced",
                        "alpha": float(full_spec["alpha"]),
                        "margin": float(full_spec["margin"]),
                        "selection_evidence": "locked source-only risk-constrained policy specification",
                    },
                    {
                        "outer_target_region": outer_region,
                        "k": k,
                        "variant": "source_utility_without_risk_constraints",
                        "alpha": float(unconstrained["alpha"]),
                        "margin": float(unconstrained["margin"]),
                        "selection_evidence": "minimum source-OOF mean delta without feasibility constraints; target outcomes unused",
                        **{
                            key: value
                            for key, value in unconstrained.items()
                            if key not in {"alpha", "margin"}
                        },
                    },
                ]
            )

    prediction_table = pd.concat(prediction_outputs, ignore_index=True)
    system_table = pd.concat(system_outputs, ignore_index=True)
    # Recompute pooled summaries across all outer-region systems, not averages of regions.
    pooled_rows = []
    for (k, variant), frame in system_table.groupby(["k", "variant"]):
        pooled_rows.append(summarize_variant(frame, int(k), str(variant)))
    summary = pd.DataFrame(pooled_rows)
    regional_contrasts = pd.concat(contrast_frames, ignore_index=True)
    pooled_contrast_frames = []
    for k, frame in system_table.groupby("k"):
        tables = {
            variant: subset.copy()
            for variant, subset in frame.groupby("variant")
        }
        pooled_contrast_frames.append(summarize_contrasts(tables, int(k)))
    pooled_contrasts = pd.concat(pooled_contrast_frames, ignore_index=True)
    small_cluster = small_cluster_mechanism_sensitivity(regional_contrasts)
    specifications = pd.DataFrame(spec_rows)
    reproduction = pd.DataFrame(reproduction_rows)

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mechanism_variant_system_outcomes.csv": system_table,
        "mechanism_variant_summary.csv": summary,
        "mechanism_variant_pooled_contrasts.csv": pooled_contrasts,
        "mechanism_variant_region_contrasts.csv": regional_contrasts,
        "mechanism_variant_small_cluster_sensitivity.csv": small_cluster,
        "mechanism_variant_source_specs.csv": specifications,
        "locked_full_selector_reproduction.csv": reproduction,
        "mechanism_base_model_cache_audit.csv": pd.DataFrame(
            prediction_cache.audit_rows
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(analysis_dir / name, index=False, encoding="utf-8-sig")
    prediction_path = analysis_dir / "mechanism_variant_predictions.csv.gz"
    prediction_table.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    input_hashes_after = {name: sha256_file(path) for name, path in input_paths.items()}
    if input_hashes_after != input_hashes_before:
        raise AssertionError("Locked mechanism-attribution inputs changed during execution")
    metadata = {
        "status": "PASS",
        "analysis_status": ANALYSIS_STATUS,
        "confirmatory_status": "not_confirmatory",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "data_package": str(data_package),
        "raw_or_uncleaned_data_used": False,
        "outer_regions": list(outer_regions),
        "k_values": list(k_values),
        "variants": list(VARIANTS),
        "interpretation_boundaries": {
            "cap_removed_at_application_same_selector": "conditional application-layer attribution; the selector was trained on capped action losses and was not retuned without the cap",
            "zero_margin_gate": "removes the positive abstention margin but retains fallback for actions predicted not to improve error",
            "forced_action_no_abstention": "forces the source-trained selector's predicted-best action for every target system",
            "source_utility_without_risk_constraints": "source-only hyperparameter choice minimizing mean error delta without the empirical risk-feasibility constraints",
            "safety": "prediction regret outcomes are not drinking-water health-safety endpoints",
        },
        "inputs": {
            name: {"path": str(path), "sha256": input_hashes_before[name]}
            for name, path in input_paths.items()
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": {
            **{
                name: {
                    "rows": int(len(frame)),
                    "sha256": sha256_file(analysis_dir / name),
                }
                for name, frame in outputs.items()
            },
            prediction_path.name: {
                "rows": int(len(prediction_table)),
                "sha256": sha256_file(prediction_path),
            },
        },
        "locked_inputs_hashes_verified_before_and_after": True,
        "locked_strict_v4_outputs_modified": False,
    }
    metadata_path = output_dir / "stage4_mechanism_attribution_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc cap, gate, and risk-mechanism attribution"
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
        "--output-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "outputs"
            / "stage4_revision_20260730"
            / "mechanism_attribution"
        ),
    )
    parser.add_argument("--outer-region", type=int, action="append")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outer_regions = tuple(args.outer_region or range(1, 11))
    k_values = tuple(sorted(set(int(value) for value in args.k_values)))
    if not k_values or any(value not in (1, 2, 3) for value in k_values):
        raise ValueError("k-values must be selected from 1,2,3")
    if args.smoke:
        outer_regions = (7,)
        k_values = (1,)
    metadata = run_mechanism_attribution(
        optimized_output=args.optimized_output,
        data_package=args.data_package,
        output_dir=args.output_dir,
        outer_regions=outer_regions,
        k_values=k_values,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
