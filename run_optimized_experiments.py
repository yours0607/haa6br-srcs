from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_new_experiments import (
    GROUP,
    INTEGRATED_V1_US_FILES,
    REGION,
    SEED,
    TARGET,
    Paths,
    build_source_model,
    clip_prediction,
    finite_quantile,
    load_data,
    metrics,
    sha256_file,
    validate_integrated_v1,
)


OUTPUT_NAME = "optimized_srcs_20260728"
PRIMARY_RISK_BUDGET = 0.12
RISK_BUDGETS = (0.08, 0.10, 0.12, 0.15)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
SHRINK_LEVELS = (0.25, 0.50, 0.75, 1.00)
MODEL_NAME = "XGBoost CUDA MAE"
POLICY_NAME = "SRCS"
MAX_ADAPTATION_SHIFT = 12.0
DECISION_TOLERANCE = 1e-12
CANDIDATE_CACHE_VERSION = "v4_strict_tail"

POLICY_FEATURES = [
    "k",
    "calibration_samples",
    "calibration_sites",
    "mean_observed",
    "median_observed",
    "sd_observed",
    "mad_observed",
    "last_observed",
    "trend_observed",
    "mean_base",
    "median_base",
    "sd_base",
    "mean_residual",
    "median_residual",
    "sd_residual",
    "mad_residual",
    "last_residual",
    "trend_residual",
    "abs_mean_residual",
    "abs_median_residual",
    "max_abs_residual",
    "history_base_gap",
    "zero_observed_fraction",
    "mean_abs_calibration_error",
]


def action_name(family: str, shrink: float) -> str:
    return f"{family}_{int(round(100 * shrink)):03d}"


ACTION_META = {
    action_name(family, shrink): {"family": family, "shrink": shrink}
    for family in (
        "Persistence",
        "HistoryMean",
        "HistoryMedian",
        "RawMean",
        "RawMedian",
    )
    for shrink in SHRINK_LEVELS
}
ACTIONS = tuple(ACTION_META)
ALL_METHOD_ACTIONS = ("Zero-shot", *ACTIONS)

ACTION_SETS = {
    "all": ACTIONS,
    "history_only": tuple(a for a in ACTIONS if ACTION_META[a]["family"].startswith("History")),
    "residual_only": tuple(a for a in ACTIONS if ACTION_META[a]["family"].startswith("Raw")),
    "no_persistence": tuple(a for a in ACTIONS if ACTION_META[a]["family"] != "Persistence"),
    "unshrunk_only": tuple(a for a in ACTIONS if ACTION_META[a]["shrink"] == 1.0),
}


@dataclass
class PolicySpec:
    alpha: float
    margin: float
    risk_budget: float
    action_set: str
    source_mean_delta: float
    source_negative_transfer: float
    source_region_balanced_negative_transfer: float
    source_worst_region_negative_transfer: float
    source_cvar90: float
    source_p95_regret: float
    source_max_regret: float
    source_adaptation_rate: float
    source_wilson_upper: float
    feasible: bool


def stable_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(v) for v in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def ordered_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def robust_sd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def linear_trend(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(1, len(values) + 1, dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def cvar90(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    tail_count = max(1, int(math.ceil(0.10 * len(values))))
    return float(np.mean(np.sort(values)[-tail_count:]))


def wilson_upper(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return float((center + spread) / denominator)


class RegionalPredictionCache:
    """Cache base predictions indexed by explicitly excluded EPA regions."""

    def __init__(
        self,
        frame: pd.DataFrame,
        features: list[str],
        model_name: str,
        cache_dir: Path,
        track: str,
        source_cache_dir: Path | None = None,
    ) -> None:
        self.frame = frame
        self.features = features
        self.model_name = model_name
        self.cache_dir = cache_dir / track / model_name.replace(" ", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_cache_dir = (
            source_cache_dir / track / model_name.replace(" ", "_")
            if source_cache_dir is not None
            else None
        )
        self.memory: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
        self.audit_rows: list[dict] = []
        self.audited_exclusions: set[tuple[int, ...]] = set()

    def _path(self, excluded: tuple[int, ...], predicted_region: int) -> Path:
        token = "-".join(str(v) for v in excluded) or "none"
        return self.cache_dir / f"exclude_{token}__predict_{predicted_region}.npy"

    def _source_path(self, excluded: tuple[int, ...], predicted_region: int) -> Path | None:
        if self.source_cache_dir is None:
            return None
        token = "-".join(str(v) for v in excluded) or "none"
        return self.source_cache_dir / f"exclude_{token}__predict_{predicted_region}.npy"

    def ensure(self, excluded_regions: Iterable[int]) -> None:
        excluded = tuple(sorted(int(v) for v in excluded_regions))
        if not excluded:
            raise ValueError("At least one predicted/excluded EPA region is required")
        missing = []
        restored_any = False
        for region in excluded:
            key = (excluded, region)
            if key in self.memory:
                continue
            path = self._path(excluded, region)
            expected = int((self.frame[REGION] == region).sum())
            if path.exists():
                values = np.load(path)
                if len(values) == expected and np.isfinite(values).all():
                    self.memory[key] = values.astype(float, copy=False)
                    continue
            source_path = self._source_path(excluded, region)
            if source_path is not None and source_path.exists():
                values = np.load(source_path)
                if len(values) == expected and np.isfinite(values).all():
                    self.memory[key] = values.astype(float, copy=False)
                    np.save(path, self.memory[key])
                    restored_any = True
                    continue
            missing.append(region)
        train = self.frame.loc[~self.frame[REGION].isin(excluded)].copy()
        if train.empty:
            raise RuntimeError(f"No training rows after excluding {excluded}")
        seed = SEED + 10000 * len(excluded) + sum((i + 1) * r for i, r in enumerate(excluded))
        elapsed = 0.0
        cache_status = "restored" if restored_any else "loaded"
        if missing:
            started = time.time()
            model = build_source_model(self.model_name, self.features, seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*No visible GPU.*")
                warnings.filterwarnings("ignore", message=".*Device is changed.*")
                model.fit(train[self.features], train[TARGET])
            elapsed = time.time() - started
            cache_status = "fitted"
            for region in excluded:
                held = self.frame.loc[self.frame[REGION] == region]
                prediction = clip_prediction(model.predict(held[self.features]))
                key = (excluded, region)
                self.memory[key] = prediction
                np.save(self._path(excluded, region), prediction)
        if excluded not in self.audited_exclusions:
            self.audit_rows.append(
                {
                    "stage": "base_model",
                    "excluded_regions": "|".join(map(str, excluded)),
                    "train_regions": "|".join(
                        map(str, sorted(set(self.frame[REGION].unique()).difference(excluded)))
                    ),
                    "predicted_regions": "|".join(map(str, excluded)),
                    "train_rows": len(train),
                    "train_systems": train[GROUP].nunique(),
                    "excluded_systems": self.frame.loc[self.frame[REGION].isin(excluded), GROUP].nunique(),
                    "system_overlap": len(
                        set(train[GROUP]).intersection(
                            self.frame.loc[self.frame[REGION].isin(excluded), GROUP]
                        )
                    ),
                    "model": self.model_name,
                    "seed": seed,
                    "fit_seconds": elapsed,
                    "cache_status": cache_status,
                    "cache_source": (
                        "external validated cache" if self.source_cache_dir else "none"
                    ),
                    "prediction_hashes": "|".join(
                        f"{region}:{sha256_file(self._path(excluded, region))}"
                        for region in excluded
                    ),
                    "feature_hash": stable_hash(self.features),
                    "sample_order_hash": ordered_hash(self.frame["sample_id"].tolist()),
                    "train_system_hash": stable_hash(train[GROUP].unique()),
                }
            )
            self.audited_exclusions.add(excluded)

    def predicted_frame(self, excluded_regions: Iterable[int], predicted_region: int) -> pd.DataFrame:
        excluded = tuple(sorted(int(v) for v in excluded_regions))
        if int(predicted_region) not in excluded:
            raise AssertionError("Predicted region must be excluded from base-model training")
        self.ensure(excluded)
        output = self.frame.loc[self.frame[REGION] == int(predicted_region)].copy()
        output["base_prediction"] = self.memory[(excluded, int(predicted_region))]
        return output


class CandidateTableCache:
    """Persist expensive system/round aggregation for each exclusion design."""

    def __init__(self, prediction_cache: RegionalPredictionCache) -> None:
        self.prediction_cache = prediction_cache
        self.directory = (
            prediction_cache.cache_dir / f"candidate_tables_{CANDIDATE_CACHE_VERSION}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        fingerprint_columns = ["sample_id", GROUP, REGION, TARGET, "round_index"]
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(
                prediction_cache.frame[fingerprint_columns], index=False
            ).to_numpy(np.uint64).tobytes()
        ).hexdigest()
        manifest = {
            "cache_version": CANDIDATE_CACHE_VERSION,
            "script_sha256": sha256_file(Path(__file__)),
            "data_hash": data_hash,
            "feature_hash": stable_hash(prediction_cache.features),
            "model": prediction_cache.model_name,
            "maximum_prediction_shift_ug_l": MAX_ADAPTATION_SHIFT,
            "actions": ACTION_META,
            "policy_features": POLICY_FEATURES,
        }
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise RuntimeError(
                    f"Candidate-cache manifest mismatch: {manifest_path}"
                )
        else:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        self.memory: dict[tuple[tuple[int, ...], int, int, bool], tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _paths(
        self, excluded: tuple[int, ...], predicted_region: int, k: int
    ) -> tuple[Path, Path]:
        token = "-".join(map(str, excluded))
        stem = f"exclude_{token}__predict_{predicted_region}__k{k}"
        return self.directory / f"{stem}__systems.pkl", self.directory / f"{stem}__samples.pkl"

    def get(
        self,
        excluded_regions: Iterable[int],
        predicted_region: int,
        k: int,
        with_samples: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        excluded = tuple(sorted(int(v) for v in excluded_regions))
        key = (excluded, int(predicted_region), int(k), bool(with_samples))
        if key in self.memory:
            return self.memory[key]
        # Rehydrate the split audit even when the derived candidate table is cached.
        self.prediction_cache.ensure(excluded)
        system_path, sample_path = self._paths(excluded, int(predicted_region), int(k))
        if system_path.exists() and (not with_samples or sample_path.exists()):
            systems = pd.read_pickle(system_path)
            samples = pd.read_pickle(sample_path) if with_samples else pd.DataFrame()
        else:
            predicted = self.prediction_cache.predicted_frame(excluded, int(predicted_region))
            systems, samples = build_candidate_dataset(predicted, int(k), with_samples)
            systems.to_pickle(system_path)
            if with_samples:
                samples.to_pickle(sample_path)
        self.memory[key] = (systems, samples)
        if with_samples:
            self.memory[(excluded, int(predicted_region), int(k), False)] = (
                systems,
                pd.DataFrame(),
            )
        return systems, samples


def build_candidate_dataset(
    predicted: pd.DataFrame,
    k: int,
    with_samples: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rounds = (
        predicted.groupby([GROUP, REGION, "round_index"], as_index=False, sort=False)
        .agg(
            observed=(TARGET, "mean"),
            base=("base_prediction", "mean"),
            round_samples=("sample_id", "size"),
        )
        .sort_values([GROUP, "round_index"])
    )
    maximum_round = rounds.groupby(GROUP)["round_index"].max()
    eligible_systems = maximum_round.index[maximum_round > k]
    if len(eligible_systems) == 0:
        return pd.DataFrame(), pd.DataFrame()

    work = predicted.loc[predicted[GROUP].isin(eligible_systems)].copy()
    calibration = work.loc[work["round_index"] <= k].copy()
    evaluation = work.loc[work["round_index"] > k].copy()
    calibration_max = calibration.groupby(GROUP)["sample_date"].max()
    evaluation_min = evaluation.groupby(GROUP)["sample_date"].min()
    chronology = calibration_max.to_frame("calibration_max").join(
        evaluation_min.rename("evaluation_min"), how="inner"
    )
    if set(chronology.index) != set(eligible_systems):
        missing = sorted(set(eligible_systems).difference(chronology.index))[:5]
        raise AssertionError(
            f"Eligible systems missing calibration/evaluation dates: {missing}, k={k}"
        )
    if not (chronology["calibration_max"] < chronology["evaluation_min"]).all():
        bad = chronology.index[
            ~(chronology["calibration_max"] < chronology["evaluation_min"])
        ][:5]
        raise AssertionError(f"Chronology violation for systems {list(bad)}, k={k}")

    calibration_rounds = rounds.loc[
        rounds[GROUP].isin(eligible_systems) & (rounds["round_index"] <= k)
    ].copy()
    calibration_rounds["residual"] = (
        calibration_rounds["observed"] - calibration_rounds["base"]
    )
    calibration_rounds["abs_residual"] = calibration_rounds["residual"].abs()
    calibration_rounds["zero_observed"] = calibration_rounds["observed"] <= 1e-12
    grouped = calibration_rounds.groupby(GROUP, sort=False)
    features = grouped.agg(
        mean_observed=("observed", "mean"),
        median_observed=("observed", "median"),
        sd_observed=("observed", "std"),
        mean_base=("base", "mean"),
        median_base=("base", "median"),
        sd_base=("base", "std"),
        mean_residual=("residual", "mean"),
        median_residual=("residual", "median"),
        sd_residual=("residual", "std"),
        max_abs_residual=("abs_residual", "max"),
        zero_observed_fraction=("zero_observed", "mean"),
        mean_abs_calibration_error=("abs_residual", "mean"),
    )
    features[["sd_observed", "sd_base", "sd_residual"]] = features[
        ["sd_observed", "sd_base", "sd_residual"]
    ].fillna(0.0)

    centered = calibration_rounds.merge(
        features[["median_observed", "median_residual"]],
        left_on=GROUP,
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    centered["abs_observed_from_median"] = (
        centered["observed"] - centered["median_observed"]
    ).abs()
    centered["abs_residual_from_median"] = (
        centered["residual"] - centered["median_residual"]
    ).abs()
    median_deviation = centered.groupby(GROUP).agg(
        mad_observed=("abs_observed_from_median", "median"),
        mad_residual=("abs_residual_from_median", "median"),
    )
    features = features.join(median_deviation, how="left")

    ordered = calibration_rounds.sort_values([GROUP, "round_index"])
    first = ordered.groupby(GROUP, sort=False).first()
    last = ordered.groupby(GROUP, sort=False).last()
    features["last_observed"] = last["observed"]
    features["last_residual"] = last["residual"]
    denominator = max(k - 1, 1)
    features["trend_observed"] = (
        (last["observed"] - first["observed"]) / denominator if k > 1 else 0.0
    )
    features["trend_residual"] = (
        (last["residual"] - first["residual"]) / denominator if k > 1 else 0.0
    )
    features["abs_mean_residual"] = features["mean_residual"].abs()
    features["abs_median_residual"] = features["median_residual"].abs()
    features["history_base_gap"] = features["mean_observed"] - features["mean_base"]
    features["calibration_samples"] = calibration.groupby(GROUP).size()
    features["calibration_sites"] = (
        calibration.groupby(GROUP)["group_site_id"].nunique()
        if "group_site_id" in calibration.columns
        else 1
    )
    features["k"] = k

    evaluation_stats = evaluation.groupby(GROUP, as_index=False).agg(
        **{
            REGION: (REGION, "first"),
            "evaluation_rounds": ("round_index", "nunique"),
            "evaluation_samples": ("sample_id", "size"),
        }
    )
    systems = features.reset_index().merge(
        evaluation_stats, on=GROUP, how="inner", validate="one_to_one"
    )
    anchor_columns = [
        GROUP,
        "last_observed",
        "mean_observed",
        "median_observed",
        "mean_residual",
        "median_residual",
    ]
    detail = evaluation.merge(
        systems[anchor_columns], on=GROUP, how="inner", validate="many_to_one"
    )
    base = detail["base_prediction"].to_numpy(float)
    y = detail[TARGET].to_numpy(float)
    action_predictions: dict[str, np.ndarray] = {}
    for action, meta in ACTION_META.items():
        shrink = float(meta["shrink"])
        family = str(meta["family"])
        if family == "Persistence":
            value = base + shrink * (detail["last_observed"].to_numpy(float) - base)
        elif family == "HistoryMean":
            value = base + shrink * (detail["mean_observed"].to_numpy(float) - base)
        elif family == "HistoryMedian":
            value = base + shrink * (detail["median_observed"].to_numpy(float) - base)
        elif family == "RawMean":
            value = base + shrink * detail["mean_residual"].to_numpy(float)
        elif family == "RawMedian":
            value = base + shrink * detail["median_residual"].to_numpy(float)
        else:
            raise AssertionError(f"Unknown action family: {family}")
        value = np.minimum(
            np.maximum(value, base - MAX_ADAPTATION_SHIFT),
            base + MAX_ADAPTATION_SHIFT,
        )
        action_predictions[action] = clip_prediction(value)

    error_frame = detail[[GROUP, "round_index"]].copy()
    error_frame["Zero-shot"] = np.abs(y - base)
    for action in ACTIONS:
        error_frame[action] = np.abs(y - action_predictions[action])
    error_columns = ["Zero-shot", *ACTIONS]
    round_mae = error_frame.groupby([GROUP, "round_index"], sort=False)[
        error_columns
    ].mean()
    system_mae = round_mae.groupby(GROUP, sort=False)[error_columns].mean()
    renamed = {"Zero-shot": "base_mae", **{a: f"mae__{a}" for a in ACTIONS}}
    systems = systems.merge(
        system_mae.rename(columns=renamed).reset_index(),
        on=GROUP,
        how="inner",
        validate="one_to_one",
    )
    for action in ACTIONS:
        systems[f"actual__{action}"] = systems[f"mae__{action}"] - systems["base_mae"]

    if not with_samples:
        return systems, pd.DataFrame()
    id_columns = [
        column
        for column in (
            "sample_id",
            GROUP,
            "group_site_id",
            REGION,
            "sample_date",
            "round_index",
        )
        if column in detail.columns
    ]
    samples = detail[id_columns].copy()
    samples["k"] = k
    samples["observed"] = y
    samples["Zero-shot"] = base
    for action in ACTIONS:
        samples[action] = action_predictions[action]
    samples["Baseline__Persistence"] = clip_prediction(
        detail["last_observed"].to_numpy(float)
    )
    samples["Baseline__HistoryMean"] = clip_prediction(
        detail["mean_observed"].to_numpy(float)
    )
    samples["Baseline__HistoryMedian"] = clip_prediction(
        detail["median_observed"].to_numpy(float)
    )
    samples["Baseline__RawMean"] = clip_prediction(
        base + detail["mean_residual"].to_numpy(float)
    )
    samples["Baseline__RawMedian"] = clip_prediction(
        base + detail["median_residual"].to_numpy(float)
    )
    return systems, samples


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def fit_policy_models(train: pd.DataFrame, alpha: float) -> dict[str, Pipeline]:
    if train.empty:
        raise RuntimeError("Policy training table is empty")
    models: dict[str, Pipeline] = {}
    for action in ACTIONS:
        model = ridge_pipeline(alpha)
        model.fit(train[POLICY_FEATURES], train[f"actual__{action}"])
        models[action] = model
    return models


def predict_policy(models: dict[str, Pipeline], frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[[GROUP, REGION, "k", "base_mae", "evaluation_rounds", "evaluation_samples"]].copy()
    for action in ACTIONS:
        output[f"pred__{action}"] = models[action].predict(frame[POLICY_FEATURES])
        output[f"actual__{action}"] = frame[f"actual__{action}"].to_numpy(float)
    return output


def decisions_from_predictions(
    predictions: pd.DataFrame,
    margin: float,
    action_set: str,
) -> pd.DataFrame:
    actions = ACTION_SETS[action_set]
    pred = predictions[[f"pred__{a}" for a in actions]].to_numpy(float)
    best_index = np.argmin(np.round(pred, decimals=12), axis=1)
    best_pred = pred[np.arange(len(pred)), best_index]
    best_action = np.asarray(actions, dtype=object)[best_index]
    adapted = best_pred <= -float(margin) + DECISION_TOLERANCE
    selected_action = np.where(adapted, best_action, "Zero-shot")
    selected_actual = np.zeros(len(predictions), dtype=float)
    for action in actions:
        mask = selected_action == action
        if mask.any():
            selected_actual[mask] = predictions.loc[mask, f"actual__{action}"].to_numpy(float)
    output = predictions[[GROUP, REGION, "k", "base_mae", "evaluation_rounds", "evaluation_samples"]].copy()
    output["selected_action"] = selected_action
    output["selected_predicted_delta"] = np.where(adapted, best_pred, 0.0)
    output["selected_actual_delta"] = selected_actual
    output["adapted"] = adapted
    output["negative_transfer"] = selected_actual > 1e-12
    return output


def fixed_spec_decision_invariant_to_future_losses(
    predictions: pd.DataFrame,
    margin: float,
    action_set: str,
    seed: int,
) -> bool:
    reference = decisions_from_predictions(predictions, margin, action_set)
    perturbed = predictions.copy()
    rng = np.random.default_rng(seed)
    for action in ACTIONS:
        perturbed[f"actual__{action}"] = rng.normal(0.0, 1000.0, len(perturbed))
    repeated = decisions_from_predictions(perturbed, margin, action_set)
    return bool(
        reference["selected_action"].equals(repeated["selected_action"])
        and np.array_equal(reference["adapted"].to_numpy(), repeated["adapted"].to_numpy())
        and np.allclose(
            reference["selected_predicted_delta"],
            repeated["selected_predicted_delta"],
            rtol=0.0,
            atol=0.0,
        )
    )


def score_decisions(decisions: pd.DataFrame, risk_budget: float) -> dict[str, float | bool]:
    delta = decisions["selected_actual_delta"].to_numpy(float)
    negative = delta > 1e-12
    by_region = decisions.groupby(REGION)["negative_transfer"].mean()
    pooled = float(np.mean(negative)) if len(delta) else 0.0
    region_balanced = float(by_region.mean()) if len(by_region) else 0.0
    worst_region = float(by_region.max()) if len(by_region) else 0.0
    upper = wilson_upper(int(negative.sum()), len(negative))
    tail = cvar90(delta)
    p95 = float(np.quantile(delta, 0.95)) if len(delta) else 0.0
    max_regret = float(np.max(delta)) if len(delta) else 0.0
    feasible = bool(
        pooled <= risk_budget + 1e-12
        and region_balanced <= risk_budget + 0.01
        and worst_region <= min(0.25, risk_budget + 0.07)
        and upper <= risk_budget + 0.02
        and float(np.mean(delta)) <= 0.0
        and tail <= 2.0
        and p95 <= 2.5
        and max_regret <= MAX_ADAPTATION_SHIFT + 1e-9
    )
    return {
        "mean_delta": float(np.mean(delta)) if len(delta) else 0.0,
        "negative_transfer": pooled,
        "region_balanced_negative_transfer": region_balanced,
        "worst_region_negative_transfer": worst_region,
        "cvar90": tail,
        "p95_regret": p95,
        "max_regret": max_regret,
        "adaptation_rate": float(decisions["adapted"].mean()) if len(delta) else 0.0,
        "wilson_upper": upper,
        "feasible": feasible,
    }


def margin_candidates(predictions: pd.DataFrame, action_set: str) -> np.ndarray:
    actions = ACTION_SETS[action_set]
    best = predictions[[f"pred__{a}" for a in actions]].min(axis=1).to_numpy(float)
    benefit = np.maximum(-best, 0.0)
    candidates = np.unique(
        np.round(
            np.concatenate(
            [
                np.array([0.0]),
                np.quantile(benefit, np.linspace(0.0, 1.0, 51)),
                np.array([float(np.max(benefit) + 1e-9)]),
            ]
            ),
            decimals=12,
        )
    )
    return candidates


def tune_policy(
    oof_by_alpha: dict[float, pd.DataFrame],
    risk_budget: float,
    action_set: str = "all",
    allowed_alphas: Iterable[float] | None = None,
) -> tuple[PolicySpec, pd.DataFrame, pd.DataFrame]:
    allowed = tuple(allowed_alphas) if allowed_alphas is not None else tuple(oof_by_alpha)
    scored_rows: list[dict] = []
    best: tuple | None = None
    best_decisions: pd.DataFrame | None = None
    for alpha in allowed:
        predictions = oof_by_alpha[float(alpha)]
        actions = ACTION_SETS[action_set]
        predicted_matrix = predictions[[f"pred__{a}" for a in actions]].to_numpy(float)
        actual_matrix = predictions[[f"actual__{a}" for a in actions]].to_numpy(float)
        best_index = np.argmin(np.round(predicted_matrix, decimals=12), axis=1)
        best_predicted = predicted_matrix[np.arange(len(predictions)), best_index]
        best_actual = actual_matrix[np.arange(len(predictions)), best_index]
        regions = predictions[REGION].to_numpy()
        for margin in margin_candidates(predictions, action_set):
            adapted = best_predicted <= -float(margin) + DECISION_TOLERANCE
            selected_actual = np.where(adapted, best_actual, 0.0)
            negative = selected_actual > 1e-12
            regional_rates = [
                float(np.mean(negative[regions == region])) for region in np.unique(regions)
            ]
            pooled = float(np.mean(negative))
            upper = wilson_upper(int(negative.sum()), len(negative))
            tail = cvar90(selected_actual)
            p95 = float(np.quantile(selected_actual, 0.95))
            max_regret = float(np.max(selected_actual))
            mean_delta = float(np.mean(selected_actual))
            region_balanced = float(np.mean(regional_rates))
            worst_region = float(np.max(regional_rates))
            feasible = bool(
                pooled <= risk_budget + 1e-12
                and region_balanced <= risk_budget + 0.01
                and worst_region <= min(0.25, risk_budget + 0.07)
                and upper <= risk_budget + 0.02
                and mean_delta <= 0.0
                and tail <= 2.0
                and p95 <= 2.5
                and max_regret <= MAX_ADAPTATION_SHIFT + 1e-9
            )
            score = {
                "mean_delta": mean_delta,
                "negative_transfer": pooled,
                "region_balanced_negative_transfer": region_balanced,
                "worst_region_negative_transfer": worst_region,
                "cvar90": tail,
                "p95_regret": p95,
                "max_regret": max_regret,
                "adaptation_rate": float(np.mean(adapted)),
                "wilson_upper": upper,
                "feasible": feasible,
            }
            row = {
                "alpha": float(alpha),
                "margin": float(margin),
                "risk_budget": risk_budget,
                "action_set": action_set,
                **score,
            }
            scored_rows.append(row)
            if score["feasible"]:
                key = (
                    float(score["mean_delta"]),
                    float(score["cvar90"]),
                    float(score["negative_transfer"]),
                    -float(score["adaptation_rate"]),
                    float(alpha),
                    float(margin),
                )
                if best is None or key < best:
                    best = key
                    best_row = row
    if best is None:
        alpha = float(min(allowed))
        predictions = oof_by_alpha[alpha]
        margin = float(np.max(margin_candidates(predictions, action_set)))
        best_decisions = decisions_from_predictions(predictions, margin, action_set)
        fallback_score = score_decisions(best_decisions, risk_budget)
        best_row = {
            "alpha": alpha,
            "margin": margin,
            "risk_budget": risk_budget,
            "action_set": action_set,
            **fallback_score,
        }
    else:
        best_predictions = oof_by_alpha[float(best_row["alpha"])]
        best_decisions = decisions_from_predictions(
            best_predictions, float(best_row["margin"]), action_set
        )
    spec = PolicySpec(
        alpha=float(best_row["alpha"]),
        margin=float(best_row["margin"]),
        risk_budget=float(risk_budget),
        action_set=action_set,
        source_mean_delta=float(best_row["mean_delta"]),
        source_negative_transfer=float(best_row["negative_transfer"]),
        source_region_balanced_negative_transfer=float(best_row["region_balanced_negative_transfer"]),
        source_worst_region_negative_transfer=float(best_row["worst_region_negative_transfer"]),
        source_cvar90=float(best_row["cvar90"]),
        source_p95_regret=float(best_row["p95_regret"]),
        source_max_regret=float(best_row["max_regret"]),
        source_adaptation_rate=float(best_row["adaptation_rate"]),
        source_wilson_upper=float(best_row["wilson_upper"]),
        feasible=bool(best_row["feasible"]),
    )
    best_decisions = best_decisions.copy()
    best_decisions["alpha"] = spec.alpha
    best_decisions["margin"] = spec.margin
    best_decisions["risk_budget"] = spec.risk_budget
    best_decisions["action_set"] = spec.action_set
    return spec, best_decisions, pd.DataFrame(scored_rows)


def selected_sample_predictions(samples: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions[GROUP].duplicated().any():
        duplicates = decisions.loc[decisions[GROUP].duplicated(keep=False), GROUP].head().tolist()
        raise AssertionError(f"Multiple selector decisions for systems: {duplicates}")
    mapping = decisions[[GROUP, "selected_action", "selected_predicted_delta", "adapted"]]
    output = samples.merge(mapping, on=GROUP, how="inner", validate="many_to_one")
    output[POLICY_NAME] = output["Zero-shot"].to_numpy(float)
    for action in ACTIONS:
        mask = output["selected_action"] == action
        if mask.any():
            output.loc[mask, POLICY_NAME] = output.loc[mask, action].to_numpy(float)
    output["selected_family"] = output["selected_action"].map(
        lambda value: ACTION_META.get(value, {"family": "Fallback"})["family"]
    )
    output["selected_shrink"] = output["selected_action"].map(
        lambda value: ACTION_META.get(value, {"shrink": 0.0})["shrink"]
    )
    return output


def clustered_interval_quantile(selected_samples: pd.DataFrame) -> tuple[float, float, float]:
    errors = np.abs(
        selected_samples["observed"].to_numpy(float)
        - selected_samples[POLICY_NAME].to_numpy(float)
    )
    sample_q = finite_quantile(errors, 0.90)
    work = selected_samples[[GROUP]].copy()
    work["error"] = errors
    system_q90 = work.groupby(GROUP)["error"].quantile(0.90).to_numpy(float)
    cluster_q = finite_quantile(system_q90, 0.90)
    return max(sample_q, cluster_q), sample_q, cluster_q


def build_outer_policy(
    core: pd.DataFrame,
    cache: RegionalPredictionCache,
    candidate_tables: CandidateTableCache,
    outer_region: int,
    k: int,
    risk_budgets: tuple[float, ...],
) -> dict[str, object]:
    all_regions = tuple(sorted(int(v) for v in core[REGION].unique()))
    source_regions = tuple(v for v in all_regions if v != outer_region)
    def candidate(excluded: Iterable[int], predicted_region: int, samples: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        return candidate_tables.get(excluded, int(predicted_region), k, samples)

    oof_frames = {alpha: [] for alpha in RIDGE_ALPHAS}
    source_sample_frames = []
    source_final_system_frames = []
    policy_audit_rows = []
    for held_region in source_regions:
        validation_systems, validation_samples = candidate(
            (outer_region, held_region), held_region, True
        )
        if validation_systems.empty:
            continue
        source_sample_frames.append(validation_samples)
        source_final_system_frames.append(validation_systems)
        training_frames = []
        for pseudo_region in source_regions:
            if pseudo_region == held_region:
                continue
            systems, _ = candidate(
                (outer_region, held_region, pseudo_region), pseudo_region, False
            )
            if not systems.empty:
                training_frames.append(systems)
        training = pd.concat(training_frames, ignore_index=True)
        overlap = set(training[GROUP]).intersection(validation_systems[GROUP])
        if overlap:
            raise AssertionError(
                f"Policy system leakage outer={outer_region}, held={held_region}: {len(overlap)}"
            )
        for alpha in RIDGE_ALPHAS:
            models = fit_policy_models(training, alpha)
            predicted = predict_policy(models, validation_systems)
            predicted["policy_held_region"] = held_region
            predicted["outer_target_region"] = outer_region
            oof_frames[alpha].append(predicted)
        policy_audit_rows.append(
            {
                "stage": "policy_crossfit",
                "outer_target_region": outer_region,
                "policy_held_region": held_region,
                "training_regions": "|".join(map(str, sorted(set(training[REGION].unique())))),
                "validation_regions": str(held_region),
                "training_systems": training[GROUP].nunique(),
                "validation_systems": validation_systems[GROUP].nunique(),
                "system_overlap": len(overlap),
                "base_training_excludes_policy_held": True,
                "policy_features_calibration_only": True,
            }
        )

    oof_by_alpha = {
        alpha: pd.concat(frames, ignore_index=True) for alpha, frames in oof_frames.items()
    }
    print(
        f"  source policy OOF ready: {len(next(iter(oof_by_alpha.values()))):,} systems",
        flush=True,
    )
    final_source_systems = pd.concat(source_final_system_frames, ignore_index=True)
    final_source_samples = pd.concat(source_sample_frames, ignore_index=True)
    target_systems, target_samples = candidate((outer_region,), outer_region, True)

    specs: dict[float, PolicySpec] = {}
    source_decisions: dict[float, pd.DataFrame] = {}
    search_frames = []
    for budget in risk_budgets:
        spec, decisions, search = tune_policy(oof_by_alpha, budget, "all")
        specs[budget] = spec
        source_decisions[budget] = decisions
        search["outer_target_region"] = outer_region
        search["k"] = k
        search_frames.append(search)
    print(
        f"  primary source policy tuned: alpha={specs[PRIMARY_RISK_BUDGET].alpha:g}, "
        f"margin={specs[PRIMARY_RISK_BUDGET].margin:.4f}",
        flush=True,
    )

    primary_spec = specs[PRIMARY_RISK_BUDGET]
    final_models_by_alpha: dict[float, dict[str, Pipeline]] = {}

    def target_policy_predictions(alpha: float) -> pd.DataFrame:
        if alpha not in final_models_by_alpha:
            final_models_by_alpha[alpha] = fit_policy_models(final_source_systems, alpha)
        return predict_policy(final_models_by_alpha[alpha], target_systems)

    target_sensitivity_rows = []
    target_decisions_by_budget: dict[float, pd.DataFrame] = {}
    for budget, spec in specs.items():
        predicted = target_policy_predictions(spec.alpha)
        decisions = decisions_from_predictions(predicted, spec.margin, spec.action_set)
        target_decisions_by_budget[budget] = decisions
        score = score_decisions(decisions, budget)
        target_sensitivity_rows.append(
            {
                "outer_target_region": outer_region,
                "k": k,
                "risk_budget": budget,
                "selected_alpha": spec.alpha,
                "selected_margin": spec.margin,
                "source_feasible": spec.feasible,
                "source_mean_delta": spec.source_mean_delta,
                "source_negative_transfer": spec.source_negative_transfer,
                "source_cvar90": spec.source_cvar90,
                "target_mean_delta": score["mean_delta"],
                "target_negative_transfer": score["negative_transfer"],
                "target_cvar90": score["cvar90"],
                "target_p95_regret": score["p95_regret"],
                "target_max_regret": score["max_regret"],
                "target_adaptation_rate": score["adaptation_rate"],
            }
        )

    ablation_rows = []
    for action_set in ACTION_SETS:
        spec, _, _ = tune_policy(oof_by_alpha, PRIMARY_RISK_BUDGET, action_set)
        predicted = target_policy_predictions(spec.alpha)
        decisions = decisions_from_predictions(predicted, spec.margin, action_set)
        score = score_decisions(decisions, PRIMARY_RISK_BUDGET)
        ablation_rows.append(
            {
                "outer_target_region": outer_region,
                "k": k,
                "ablation_type": "action_set",
                "setting": action_set,
                "selected_alpha": spec.alpha,
                "selected_margin": spec.margin,
                "source_mean_delta": spec.source_mean_delta,
                "source_negative_transfer": spec.source_negative_transfer,
                "target_mean_delta": score["mean_delta"],
                "target_negative_transfer": score["negative_transfer"],
                "target_cvar90": score["cvar90"],
                "target_adaptation_rate": score["adaptation_rate"],
            }
        )
    for alpha in RIDGE_ALPHAS:
        spec, _, _ = tune_policy(
            oof_by_alpha, PRIMARY_RISK_BUDGET, "all", allowed_alphas=(alpha,)
        )
        predicted = target_policy_predictions(alpha)
        decisions = decisions_from_predictions(predicted, spec.margin, "all")
        score = score_decisions(decisions, PRIMARY_RISK_BUDGET)
        ablation_rows.append(
            {
                "outer_target_region": outer_region,
                "k": k,
                "ablation_type": "ridge_alpha",
                "setting": str(alpha),
                "selected_alpha": alpha,
                "selected_margin": spec.margin,
                "source_mean_delta": spec.source_mean_delta,
                "source_negative_transfer": spec.source_negative_transfer,
                "target_mean_delta": score["mean_delta"],
                "target_negative_transfer": score["negative_transfer"],
                "target_cvar90": score["cvar90"],
                "target_adaptation_rate": score["adaptation_rate"],
            }
        )

    source_selected_samples = selected_sample_predictions(
        final_source_samples, source_decisions[PRIMARY_RISK_BUDGET]
    )
    q90, q90_sample, q90_cluster = clustered_interval_quantile(source_selected_samples)
    target_selected_samples = selected_sample_predictions(
        target_samples, target_decisions_by_budget[PRIMARY_RISK_BUDGET]
    )
    target_selected_samples["interval_low"] = np.maximum(
        0.0, target_selected_samples[POLICY_NAME] - q90
    )
    target_selected_samples["interval_high"] = target_selected_samples[POLICY_NAME] + q90
    target_selected_samples["q90_source"] = q90
    target_selected_samples["outer_label"] = f"EPA-{outer_region}"
    target_selected_samples["evidence_class"] = "Post-hoc nested internal EPA-region validation"

    source_invariance = fixed_spec_decision_invariant_to_future_losses(
        oof_by_alpha[primary_spec.alpha],
        primary_spec.margin,
        primary_spec.action_set,
        SEED + 100 * outer_region + k,
    )
    target_invariance = fixed_spec_decision_invariant_to_future_losses(
        target_policy_predictions(primary_spec.alpha),
        primary_spec.margin,
        primary_spec.action_set,
        SEED + 1000 + 100 * outer_region + k,
    )
    if not source_invariance or not target_invariance:
        raise AssertionError("Fixed-spec selector changed after perturbing future loss labels")

    spec_row = {
        "outer_target_region": outer_region,
        "k": k,
        **primary_spec.__dict__,
        "q90_source": q90,
        "q90_source_sample": q90_sample,
        "q90_source_system_clustered": q90_cluster,
        "source_calibration_samples": len(source_selected_samples),
        "source_calibration_systems": source_selected_samples[GROUP].nunique(),
        "source_fixed_spec_future_loss_invariance": source_invariance,
        "target_fixed_spec_future_loss_invariance": target_invariance,
        "source_metrics_role": "Policy tuning constraints; not independent validation evidence",
    }
    primary_source_decisions = source_decisions[PRIMARY_RISK_BUDGET].copy()
    primary_source_decisions["outer_target_region"] = outer_region
    primary_source_decisions["evidence_role"] = (
        "Cross-fitted source action predictions used for global policy tuning"
    )
    return {
        "predictions": target_selected_samples,
        "policy_spec": pd.DataFrame([spec_row]),
        "source_decisions": primary_source_decisions,
        "risk_sensitivity": pd.DataFrame(target_sensitivity_rows),
        "ablation": pd.DataFrame(ablation_rows),
        "policy_search": pd.concat(search_frames, ignore_index=True),
        "policy_audit": pd.DataFrame(policy_audit_rows),
        "invariant_audit": pd.DataFrame(
            [
                {
                    "outer_target_region": outer_region,
                    "k": k,
                    "source_fixed_spec_future_loss_invariance": source_invariance,
                    "target_fixed_spec_future_loss_invariance": target_invariance,
                    "invariance_scope": (
                        "Fixed alpha, margin, and action set; source tuning scores are not "
                        "independent validation evidence"
                    ),
                    "policy_features_exclude_future_fields": set(POLICY_FEATURES).isdisjoint(
                        {"evaluation_rounds", "evaluation_samples", "base_mae"}
                    ),
                }
            ]
        ),
    }


def add_baseline_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["Persistence"] = output["Baseline__Persistence"]
    output["History mean"] = output["Baseline__HistoryMean"]
    output["History median"] = output["Baseline__HistoryMedian"]
    output["Raw residual"] = output["Baseline__RawMean"]
    output["Raw median residual"] = output["Baseline__RawMedian"]
    output["Capped Persistence"] = output[action_name("Persistence", 1.0)]
    output["Capped History mean"] = output[action_name("HistoryMean", 1.0)]
    output["Capped History median"] = output[action_name("HistoryMedian", 1.0)]
    output["Capped Raw residual"] = output[action_name("RawMean", 1.0)]
    output["Capped Raw median residual"] = output[action_name("RawMedian", 1.0)]
    return output


def round_system_errors(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    columns = [REGION, GROUP, "round_index", "observed", "Zero-shot"]
    if method != "Zero-shot":
        columns.append(method)
    work = frame[columns].copy()
    work["base_abs"] = np.abs(work["observed"] - work["Zero-shot"])
    work["method_abs"] = (
        work["base_abs"]
        if method == "Zero-shot"
        else np.abs(work["observed"] - work[method])
    )
    rounds = (
        work.groupby([REGION, GROUP, "round_index"], as_index=False)[["base_abs", "method_abs"]]
        .mean()
    )
    systems = rounds.groupby([REGION, GROUP], as_index=False)[["base_abs", "method_abs"]].mean()
    systems["delta"] = systems["method_abs"] - systems["base_abs"]
    return systems


def result_summaries(frame: pd.DataFrame, analysis: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = [
        "Zero-shot",
        "Persistence",
        "History mean",
        "History median",
        "Raw residual",
        "Raw median residual",
        "Capped Persistence",
        "Capped History mean",
        "Capped History median",
        "Capped Raw residual",
        "Capped Raw median residual",
        POLICY_NAME,
    ]
    summary_rows = []
    region_rows = []
    system_frames = []
    for k in sorted(frame["k"].unique()):
        subset = frame.loc[frame["k"] == k].copy()
        history_system = round_system_errors(subset, "History mean").set_index(GROUP)
        for method in methods:
            values = metrics(subset["observed"].to_numpy(float), subset[method].to_numpy(float))
            systems = round_system_errors(subset, method)
            sample_balanced = (
                subset.assign(
                    base_abs=np.abs(subset["observed"] - subset["Zero-shot"]),
                    method_abs=np.abs(subset["observed"] - subset[method]),
                )
                .groupby([REGION, GROUP], as_index=False)[["base_abs", "method_abs"]]
                .mean()
            )
            sample_balanced["delta"] = (
                sample_balanced["method_abs"] - sample_balanced["base_abs"]
            )
            systems["analysis"] = analysis
            systems["k"] = k
            systems["method"] = method
            system_frames.append(systems)
            regional_mae = []
            for region, regional in subset.groupby(REGION):
                regional_metrics = metrics(
                    regional["observed"].to_numpy(float), regional[method].to_numpy(float)
                )
                regional_systems = systems.loc[systems[REGION] == region]
                regional_sample_balanced = sample_balanced.loc[
                    sample_balanced[REGION] == region
                ]
                region_rows.append(
                    {
                        "analysis": analysis,
                        "k": k,
                        REGION: region,
                        "method": method,
                        "rows": len(regional),
                        "systems": regional[GROUP].nunique(),
                        **regional_metrics,
                        "system_round_balanced_mae": float(regional_systems["method_abs"].mean()),
                        "negative_transfer_rate": float(np.mean(regional_systems["delta"] > 1e-12)),
                        "negative_transfer_rate_sample_balanced_legacy": float(
                            np.mean(regional_sample_balanced["delta"] > 1e-12)
                        ),
                        "mean_regret": float(regional_systems["delta"].mean()),
                        "cvar90_regret": cvar90(regional_systems["delta"].to_numpy(float)),
                    }
                )
                regional_mae.append(regional_metrics["mae"])
            delta = systems["delta"].to_numpy(float)
            history_delta = (
                systems.set_index(GROUP)["method_abs"] - history_system["method_abs"]
            ).dropna()
            coverage = np.nan
            width = np.nan
            adaptation_rate = np.nan
            adaptation_rate_sample_weighted = np.nan
            if method == POLICY_NAME and {"interval_low", "interval_high"}.issubset(subset):
                coverage = float(
                    np.mean(
                        (subset["observed"] >= subset["interval_low"])
                        & (subset["observed"] <= subset["interval_high"])
                    )
                )
                width = float(np.mean(subset["interval_high"] - subset["interval_low"]))
                system_adaptation = subset[[GROUP, "adapted"]].drop_duplicates()
                if len(system_adaptation) != subset[GROUP].nunique():
                    raise AssertionError("Adaptation status is not unique within system")
                adaptation_rate = float(system_adaptation["adapted"].mean())
                adaptation_rate_sample_weighted = float(subset["adapted"].mean())
            summary_rows.append(
                {
                    "analysis": analysis,
                    "k": k,
                    "method": method,
                    "rows": len(subset),
                    "systems": subset[GROUP].nunique(),
                    **values,
                    "system_round_balanced_mae": float(systems["method_abs"].mean()),
                    "region_balanced_sample_mae": float(np.mean(regional_mae)),
                    "relative_sample_mae_improvement": (
                        float(np.mean(np.abs(subset["observed"] - subset["Zero-shot"]))) - values["mae"]
                    )
                    / max(float(np.mean(np.abs(subset["observed"] - subset["Zero-shot"]))), 1e-12),
                    "negative_transfer_rate": float(np.mean(delta > 1e-12)),
                    "negative_transfer_rate_sample_balanced_legacy": float(
                        np.mean(sample_balanced["delta"] > 1e-12)
                    ),
                    "risk_estimand": "Equal-system, equal-future-round delta MAE",
                    "mean_regret": float(np.mean(delta)),
                    "p95_regret": float(np.quantile(delta, 0.95)),
                    "max_regret": float(np.max(delta)),
                    "cvar90_regret": cvar90(delta),
                    "system_mae_delta_vs_history": float(history_delta.mean()),
                    "coverage_90": coverage,
                    "mean_interval_width": width,
                    "adaptation_rate": adaptation_rate,
                    "adaptation_rate_sample_weighted": adaptation_rate_sample_weighted,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(region_rows), pd.concat(system_frames, ignore_index=True)


def hierarchical_bootstrap(
    systems: pd.DataFrame,
    value_column: str,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    region_values = {
        region: group[value_column].to_numpy(float)
        for region, group in systems.groupby(REGION)
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
    return {
        "estimate": float(systems[value_column].mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "probability_below_zero": float(np.mean(estimates < 0.0)),
        "bootstrap_replicates": n_boot,
    }


def bootstrap_table(predictions: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for k in sorted(predictions["k"].unique()):
        subset = predictions.loc[predictions["k"] == k]
        for method in ("History mean", "Raw residual", POLICY_NAME):
            systems = round_system_errors(subset, method)
            result = hierarchical_bootstrap(
                systems, "delta", n_boot, SEED + 100 * int(k) + len(method)
            )
            rows.append({"k": k, "method": method, "contrast": "method - Zero-shot", **result})
        srcs = round_system_errors(subset, POLICY_NAME).set_index([REGION, GROUP])
        history = round_system_errors(subset, "History mean").set_index([REGION, GROUP])
        paired = srcs[["method_abs"]].rename(columns={"method_abs": "srcs"}).join(
            history[["method_abs"]].rename(columns={"method_abs": "history"}), how="inner"
        )
        paired["delta"] = paired["srcs"] - paired["history"]
        paired = paired.reset_index()
        result = hierarchical_bootstrap(
            paired, "delta", n_boot, SEED + 1000 + int(k)
        )
        rows.append({"k": k, "method": POLICY_NAME, "contrast": "SRCS - History mean", **result})
    return pd.DataFrame(rows)


def coverage_tables(predictions: pd.DataFrame, analysis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    for k, subset in predictions.groupby("k"):
        covered = (
            (subset["observed"] >= subset["interval_low"])
            & (subset["observed"] <= subset["interval_high"])
        )
        work = subset.copy()
        work["covered"] = covered
        levels = {
            "overall": ["k"],
            "region": [REGION],
            "system": [GROUP],
        }
        if "group_site_id" in work.columns:
            levels["site"] = ["group_site_id"]
        for level, columns in levels.items():
            if level == "overall":
                grouped = pd.DataFrame(
                    [{"unit": "all", "rows": len(work), "coverage": float(work["covered"].mean())}]
                )
            else:
                grouped = (
                    work.groupby(columns, dropna=False)["covered"]
                    .agg(rows="size", coverage="mean")
                    .reset_index()
                )
                grouped["unit"] = grouped[columns].astype(str).agg("|".join, axis=1)
                grouped = grouped[["unit", "rows", "coverage"]]
            grouped["analysis"] = analysis
            grouped["k"] = k
            grouped["level"] = level
            detail_rows.extend(grouped.to_dict("records"))
            summary_rows.append(
                {
                    "analysis": analysis,
                    "k": k,
                    "level": level,
                    "units": len(grouped),
                    "mean_coverage": float(grouped["coverage"].mean()),
                    "min_coverage": float(grouped["coverage"].min()),
                    "fraction_below_75": float(np.mean(grouped["coverage"] < 0.75)),
                    "fraction_zero": float(np.mean(grouped["coverage"] <= 0.0)),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def eligibility_bias(core: pd.DataFrame, k_values: tuple[int, ...]) -> pd.DataFrame:
    system = (
        core.groupby([REGION, GROUP], as_index=False)
        .agg(
            rows=("sample_id", "size"),
            sites=("group_site_id", "nunique") if "group_site_id" in core.columns else ("sample_id", "size"),
            rounds=("round_index", "max"),
            mean_haa6br=(TARGET, "mean"),
            median_haa6br=(TARGET, "median"),
        )
    )
    rows = []
    for k in k_values:
        system["eligible"] = system["rounds"] > k
        eligible = system.loc[system["eligible"]]
        ineligible = system.loc[~system["eligible"]]
        pooled_sd = float(system["mean_haa6br"].std(ddof=1))
        for region, subset in [("All", system), *list(system.groupby(REGION))]:
            rows.append(
                {
                    "k": k,
                    REGION: region,
                    "all_systems": len(subset),
                    "eligible_systems": int(subset["eligible"].sum()),
                    "eligibility_rate": float(subset["eligible"].mean()),
                    "eligible_mean_haa6br": float(subset.loc[subset["eligible"], "mean_haa6br"].mean()),
                    "ineligible_mean_haa6br": float(subset.loc[~subset["eligible"], "mean_haa6br"].mean()),
                    "eligible_median_rows": float(subset.loc[subset["eligible"], "rows"].median()),
                    "ineligible_median_rows": float(subset.loc[~subset["eligible"], "rows"].median()),
                }
            )
        rows[-(len(system[REGION].unique()) + 1)]["standardized_mean_difference_all"] = (
            float(eligible["mean_haa6br"].mean() - ineligible["mean_haa6br"].mean())
            / max(pooled_sd, 1e-12)
        )
    return pd.DataFrame(rows)


def before_after_table(
    new_summary: pd.DataFrame,
    old_summary_path: Path | None,
) -> pd.DataFrame:
    rows = []
    if old_summary_path is not None and old_summary_path.exists():
        old = pd.read_csv(old_summary_path)
        for _, row in old.loc[old["method"].isin(["Zero-shot", "History mean", "Raw residual", "HRC", "SafeShrink HRC"])].iterrows():
            rows.append(
                {
                    "version": "Current protocol v2",
                    "evidence": row["analysis"],
                    "k": int(row["k"]),
                    "method": row["method"],
                    "sample_mae": row["mae"],
                    "rmse": row["rmse"],
                    "r2": row["r2"],
                    "negative_transfer_rate": row["negative_transfer_rate"],
                    "negative_transfer_rate_sample_balanced": row["negative_transfer_rate"],
                    "risk_estimand": "Equal-system; samples equally weighted within system",
                    "coverage_90": row["coverage_90"],
                    "note": "Historical output; source safety gate was not fully nested",
                }
            )
    for _, row in new_summary.loc[
        new_summary["method"].isin(
            [
                "Zero-shot",
                "History mean",
                "Raw residual",
                "Capped History mean",
                "Capped Raw residual",
                POLICY_NAME,
            ]
        )
    ].iterrows():
        rows.append(
            {
                "version": "Optimized SRCS v4",
                "evidence": row["analysis"],
                "k": int(row["k"]),
                "method": row["method"],
                "sample_mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "negative_transfer_rate": row["negative_transfer_rate"],
                "negative_transfer_rate_sample_balanced": row[
                    "negative_transfer_rate_sample_balanced_legacy"
                ],
                "risk_estimand": row["risk_estimand"],
                "coverage_90": row["coverage_90"],
                "note": "Post-hoc nested internal validation; target region isolated",
            }
        )
    return pd.DataFrame(rows)


def run_us(
    core: pd.DataFrame,
    features: list[str],
    cache_root: Path,
    prediction_cache_source: Path | None,
    model_name: str,
    outer_regions: tuple[int, ...],
    k_values: tuple[int, ...],
    risk_budgets: tuple[float, ...],
) -> dict[str, pd.DataFrame]:
    cache = RegionalPredictionCache(
        core,
        features,
        model_name,
        cache_root,
        "us_operational",
        prediction_cache_source,
    )
    candidate_tables = CandidateTableCache(cache)
    collected: dict[str, list[pd.DataFrame]] = {
        "predictions": [],
        "policy_spec": [],
        "source_decisions": [],
        "risk_sensitivity": [],
        "ablation": [],
        "policy_search": [],
        "policy_audit": [],
        "invariant_audit": [],
    }
    total = len(outer_regions) * len(k_values)
    step = 0
    for outer_region in outer_regions:
        for k in k_values:
            step += 1
            print(
                f"[US {step}/{total}] outer EPA region {outer_region}, k={k}",
                flush=True,
            )
            result = build_outer_policy(
                core, cache, candidate_tables, outer_region, k, risk_budgets
            )
            for key in collected:
                collected[key].append(result[key])
    output_tables = {
        key: pd.concat(frames, ignore_index=True) for key, frames in collected.items()
    }
    output_tables["model_audit"] = pd.DataFrame(cache.audit_rows)
    output_tables["predictions"] = add_baseline_aliases(output_tables["predictions"])
    return output_tables


def external_source_policy(
    core: pd.DataFrame,
    features: list[str],
    cache_root: Path,
    prediction_cache_source: Path | None,
    model_name: str,
    k_values: tuple[int, ...],
    risk_budgets: tuple[float, ...],
) -> tuple[dict[int, dict[str, object]], RegionalPredictionCache]:
    cache = RegionalPredictionCache(
        core,
        features,
        model_name,
        cache_root,
        "us_transport",
        prediction_cache_source,
    )
    candidate_tables = CandidateTableCache(cache)
    regions = tuple(sorted(int(v) for v in core[REGION].unique()))
    results: dict[int, dict[str, object]] = {}
    for k in k_values:
        print(f"[UK source policy] k={k}", flush=True)
        def candidate(excluded: Iterable[int], predicted_region: int, samples: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
            return candidate_tables.get(excluded, predicted_region, k, samples)

        oof_frames = {alpha: [] for alpha in RIDGE_ALPHAS}
        final_system_frames = []
        final_sample_frames = []
        for held in regions:
            validation, validation_samples = candidate((held,), held, True)
            final_system_frames.append(validation)
            final_sample_frames.append(validation_samples)
            training_frames = []
            for pseudo in regions:
                if pseudo == held:
                    continue
                systems, _ = candidate((held, pseudo), pseudo, False)
                training_frames.append(systems)
            training = pd.concat(training_frames, ignore_index=True)
            for alpha in RIDGE_ALPHAS:
                models = fit_policy_models(training, alpha)
                predicted = predict_policy(models, validation)
                predicted["policy_held_region"] = held
                oof_frames[alpha].append(predicted)
        oof_by_alpha = {
            alpha: pd.concat(frames, ignore_index=True) for alpha, frames in oof_frames.items()
        }
        specs = {}
        decisions = {}
        for budget in risk_budgets:
            specs[budget], decisions[budget], _ = tune_policy(oof_by_alpha, budget, "all")
        final_systems = pd.concat(final_system_frames, ignore_index=True)
        final_samples = pd.concat(final_sample_frames, ignore_index=True)
        primary = specs[PRIMARY_RISK_BUDGET]
        final_models = fit_policy_models(final_systems, primary.alpha)
        source_samples = selected_sample_predictions(final_samples, decisions[PRIMARY_RISK_BUDGET])
        q90, q90_sample, q90_cluster = clustered_interval_quantile(source_samples)
        results[k] = {
            "specs": specs,
            "primary": primary,
            "models": final_models,
            "q90": q90,
            "q90_sample": q90_sample,
            "q90_cluster": q90_cluster,
            "source_decisions": decisions[PRIMARY_RISK_BUDGET],
        }
    return results, cache


def apply_uk_after_lock(
    core: pd.DataFrame,
    features: list[str],
    data_package: Path,
    source_policy: dict[int, dict[str, object]],
    model_name: str,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    model = build_source_model(model_name, features, SEED + 99000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*No visible GPU.*")
        warnings.filterwarnings("ignore", message=".*Device is changed.*")
        model.fit(core[features], core[TARGET])
    uk = pd.read_csv(data_package / "data" / "uk_dwi242_locked_external.csv", low_memory=False)
    uk[REGION] = "DWI242"
    uk["sample_date"] = pd.to_datetime(uk["sample_date"], errors="coerce")
    uk = uk.loc[uk["sample_date"].notna()].copy()
    uk.sort_values([GROUP, "sample_date", "sample_id"], inplace=True)
    uk["round_index"] = uk.groupby(GROUP)["sample_date"].rank(method="dense").astype(int)
    uk["system_rounds"] = uk.groupby(GROUP)["round_index"].transform("max").astype(int)
    uk["base_prediction"] = clip_prediction(model.predict(uk[features]))
    prediction_frames = []
    for k in k_values:
        systems, samples = build_candidate_dataset(uk, k, True)
        if systems.empty:
            continue
        policy = source_policy[k]
        predicted = predict_policy(policy["models"], systems)
        spec: PolicySpec = policy["primary"]
        decisions = decisions_from_predictions(predicted, spec.margin, spec.action_set)
        selected = selected_sample_predictions(samples, decisions)
        selected["interval_low"] = np.maximum(0.0, selected[POLICY_NAME] - float(policy["q90"]))
        selected["interval_high"] = selected[POLICY_NAME] + float(policy["q90"])
        selected["q90_source"] = float(policy["q90"])
        selected["outer_label"] = "UK-DWI242"
        selected["evidence_class"] = "Retrospective U.K. stress test; outcomes previously viewed"
        prediction_frames.append(selected)
    return add_baseline_aliases(pd.concat(prediction_frames, ignore_index=True))


def save_table(frame: pd.DataFrame, tables: Path, name: str) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict post-hoc SRCS optimization for HAA6Br")
    parser.add_argument(
        "--data-package",
        type=Path,
        required=True,
        help="Path to a cleaned haa6br_integrated_v1 package; no raw data are accepted",
    )
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory (overrides --output-name)",
    )
    parser.add_argument(
        "--historical-output",
        type=Path,
        help="Optional run_new_experiments.py output directory for before/after comparison",
    )
    parser.add_argument("--base-model", default=MODEL_NAME, choices=("Median", "ElasticNet", MODEL_NAME))
    parser.add_argument("--held-region", type=int, action="append")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--skip-uk", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--prediction-cache-source",
        type=Path,
        help="Optional read-only source of validated base-prediction .npy caches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.k_values or any(int(k) not in (1, 2, 3) for k in args.k_values):
        raise ValueError("This locked protocol supports k-values 1, 2, and 3 only")
    started = time.time()
    required_files = INTEGRATED_V1_US_FILES
    if not args.skip_uk:
        required_files = (*required_files, Path("data/uk_dwi242_locked_external.csv"))
    data_package = validate_integrated_v1(args.data_package, required_files)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (PROJECT_DIR / "outputs" / args.output_name).resolve()
    )
    tables = output / "tables"
    locks = output / "locks"
    for directory in (output, tables, locks, output / "cache"):
        directory.mkdir(parents=True, exist_ok=True)
    cache_root = output / "cache"
    prediction_cache_source = (
        args.prediction_cache_source.resolve()
        if args.prediction_cache_source is not None
        else None
    )
    if prediction_cache_source is not None and not prediction_cache_source.is_dir():
        raise FileNotFoundError(
            f"Prediction cache source does not exist: {prediction_cache_source}"
        )
    paths = Paths(data_package, output, tables, output / "figures", locks)

    core_path = data_package / "data" / "us_ucmr4_core.csv"
    protocol = {
        "protocol": "SRCS post-hoc nested internal validation v4",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "seed": SEED,
        "data_contract": "haa6br_integrated_v1",
        "core_sha256": sha256_file(core_path),
        "raw_or_archived_inputs_permitted": False,
        "base_model": args.base_model,
        "base_model_status": "Fixed after reviewing prior U.S. results; post-hoc, not confirmatory",
        "outer_split": "Leave-one-EPA-region-out",
        "policy_split": "For target T and policy-held B, policy labels for C use base models excluding T,B,C",
        "policy_features": POLICY_FEATURES,
        "action_formulas": {
            "history_or_persistence": "clip_to_base_plus_minus_12(base + lambda*(anchor-base))",
            "residual": "clip_to_base_plus_minus_12(base + lambda*local_residual)",
        },
        "benchmark_formulas": "Legacy uncapped Persistence, History mean/median, and Raw residual mean/median",
        "maximum_prediction_shift_ug_l": MAX_ADAPTATION_SHIFT,
        "decision_tolerance": DECISION_TOLERANCE,
        "candidate_cache_version": CANDIDATE_CACHE_VERSION,
        "prediction_cache_source": "external validated cache" if prediction_cache_source else "none",
        "actions": ACTION_META,
        "risk_constraints": {
            "pooled_negative_transfer": "<= risk budget",
            "region_balanced_negative_transfer": "<= risk budget + 0.01",
            "worst_region_negative_transfer": "<= min(0.25, risk budget + 0.07)",
            "wilson_upper": "<= risk budget + 0.02",
            "mean_delta": "<= 0",
            "strict_worst_decile_cvar": "<= 2.0 ug/L",
            "p95_regret": "<= 2.5 ug/L",
            "maximum_regret": "<= 12.0 ug/L",
        },
        "cvar90_definition": "Mean of the largest ceil(10% * n_systems) system regrets",
        "source_policy_metrics_role": "Training/tuning constraints, not independent OOF validation",
        "interval_interpretation": "Source-derived empirical prediction band; not formal cluster conformal coverage",
        "shrink_levels": SHRINK_LEVELS,
        "ridge_alphas": RIDGE_ALPHAS,
        "risk_budgets": RISK_BUDGETS,
        "primary_risk_budget": PRIMARY_RISK_BUDGET,
        "risk_interpretation": "Empirical algorithmic negative-transfer budget; not a health-safety guarantee",
        "estimand": "Equal-system, equal-future-round delta MAE",
        "uk_role": "Retrospective stress test; outcomes were viewed before this optimization",
    }
    protocol_path = locks / "protocol_lock_before_optimized_run.json"
    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected = {k: v for k, v in protocol.items() if k != "locked_at_utc"}
        observed = {k: v for k, v in existing_protocol.items() if k != "locked_at_utc"}
        if observed != expected:
            raise RuntimeError(
                f"Existing protocol lock does not match the running code/configuration: {protocol_path}"
            )
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    core, _, _, feature_sets = load_data(paths)
    operational_features = list(feature_sets["us_operational_core"])
    transport_features = list(feature_sets["transportable_core"])
    all_regions = tuple(sorted(int(v) for v in core[REGION].unique()))
    outer_regions = tuple(args.held_region) if args.held_region else all_regions
    k_values = tuple(sorted(set(int(v) for v in args.k_values)))
    if args.smoke:
        outer_regions = outer_regions[:1] if args.held_region else (7,)
        k_values = k_values[:1]
        if args.bootstrap == 5000:
            args.bootstrap = 20

    us = run_us(
        core,
        operational_features,
        cache_root,
        prediction_cache_source,
        args.base_model,
        outer_regions,
        k_values,
        RISK_BUDGETS,
    )
    us_predictions = us["predictions"]
    us_summary, us_regions, us_systems = result_summaries(
        us_predictions, "Post-hoc nested internal EPA-region validation"
    )
    us_bootstrap = bootstrap_table(us_predictions, args.bootstrap)
    us_coverage_detail, us_coverage_summary = coverage_tables(
        us_predictions, "Post-hoc nested internal EPA-region validation"
    )
    eligibility = eligibility_bias(core, k_values)
    old_summary_path = (
        args.historical_output.expanduser().resolve() / "tables" / "core_summary.csv"
        if args.historical_output is not None
        else None
    )
    before_after = before_after_table(us_summary, old_summary_path)

    for name, frame in {
        "us_predictions": us_predictions,
        "us_summary": us_summary,
        "us_region_summary": us_regions,
        "us_system_summary": us_systems,
        "us_hierarchical_bootstrap": us_bootstrap,
        "us_coverage_detail": us_coverage_detail,
        "us_coverage_summary": us_coverage_summary,
        "eligibility_selection_bias": eligibility,
        "before_after_comparison": before_after,
        "policy_spec": us["policy_spec"],
        "source_policy_oof": us["source_decisions"],
        "risk_budget_sensitivity": us["risk_sensitivity"],
        "policy_ablation": us["ablation"],
        "policy_search_full": us["policy_search"],
        "policy_split_audit": us["policy_audit"],
        "selector_invariant_audit": us["invariant_audit"],
        "base_model_split_audit": us["model_audit"],
    }.items():
        save_table(frame, tables, name)

    uk_metadata = {"status": "SKIPPED"}
    if not args.skip_uk and set(outer_regions) == set(all_regions):
        source_policy, transport_cache = external_source_policy(
            core,
            transport_features,
            cache_root,
            prediction_cache_source,
            args.base_model,
            k_values,
            RISK_BUDGETS,
        )
        method_lock = {
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(protocol_path),
            "uk_outcomes_loaded_before_lock": False,
            "base_model": args.base_model,
            "transport_features": transport_features,
            "policy": {
                str(k): {
                    **source_policy[k]["primary"].__dict__,
                    "q90": source_policy[k]["q90"],
                }
                for k in k_values
            },
            "evidence_warning": "UK outcomes had been viewed in earlier work; this lock prevents only run-time tuning",
        }
        method_lock_path = locks / "method_lock_before_uk_runtime_load.json"
        method_lock_path.write_text(
            json.dumps(method_lock, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        uk_predictions = apply_uk_after_lock(
            core,
            transport_features,
            data_package,
            source_policy,
            args.base_model,
            k_values,
        )
        uk_summary, uk_regions, uk_systems = result_summaries(
            uk_predictions, "Retrospective U.K. stress test"
        )
        uk_coverage_detail, uk_coverage_summary = coverage_tables(
            uk_predictions, "Retrospective U.K. stress test"
        )
        for name, frame in {
            "uk_predictions": uk_predictions,
            "uk_summary": uk_summary,
            "uk_region_summary": uk_regions,
            "uk_system_summary": uk_systems,
            "uk_coverage_detail": uk_coverage_detail,
            "uk_coverage_summary": uk_coverage_summary,
            "transport_base_model_split_audit": pd.DataFrame(transport_cache.audit_rows),
        }.items():
            save_table(frame, tables, name)
        uk_metadata = {
            "status": "COMPLETE_RETROSPECTIVE",
            "rows": len(uk_predictions),
            "systems": uk_predictions[GROUP].nunique(),
            "method_lock": "locks/method_lock_before_uk_runtime_load.json",
        }

    base_audit = us["model_audit"]
    policy_audit = us["policy_audit"]
    invariant_audit = us["invariant_audit"]
    leakage_checks = {
        "base_system_overlap_zero": bool((base_audit["system_overlap"] == 0).all()),
        "policy_system_overlap_zero": bool((policy_audit["system_overlap"] == 0).all()),
        "policy_base_excludes_held": bool(policy_audit["base_training_excludes_policy_held"].all()),
        "policy_features_calibration_only": bool(policy_audit["policy_features_calibration_only"].all()),
        "source_fixed_spec_future_loss_invariance": bool(
            invariant_audit["source_fixed_spec_future_loss_invariance"].all()
        ),
        "target_fixed_spec_future_loss_invariance": bool(
            invariant_audit["target_fixed_spec_future_loss_invariance"].all()
        ),
        "policy_features_exclude_future_fields": bool(
            invariant_audit["policy_features_exclude_future_fields"].all()
        ),
    }
    if not all(leakage_checks.values()):
        raise AssertionError(f"Leakage audit failed: {leakage_checks}")
    metadata = {
        "status": "PASS_EXECUTION_AND_AUDIT",
        "scientific_status": "Post-hoc nested internal validation; judge from result tables",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "base_model": args.base_model,
        "outer_regions": outer_regions,
        "k_values": k_values,
        "bootstrap_replicates": args.bootstrap,
        "protocol_lock": "locks/protocol_lock_before_optimized_run.json",
        "leakage_checks": leakage_checks,
        "uk": uk_metadata,
        "tables": sorted(path.stem for path in tables.glob("*.csv")),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
