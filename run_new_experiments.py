from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


SEED = 20260728
TARGET = "haa6br_ug_l"
GROUP = "group_system_id"
REGION = "epa_region"
N_BOOT = 500
OUTPUT_NAME = "new_plan_20260728"
INTEGRATED_V1_US_FILES = (
    Path("metadata/model_feature_sets.json"),
    Path("data/us_ucmr4_core.csv"),
    Path("data/us_ucmr4_enriched_strict.csv"),
)
INTEGRATED_V1_REQUIRED_FILES = (
    *INTEGRATED_V1_US_FILES,
    Path("data/uk_dwi242_locked_external.csv"),
    Path("data/uk_dbp2009_field_external.csv"),
)

FORBIDDEN = {
    "haa5_ug_l_audit_only",
    "haa9_ug_l_audit_only",
    "haa6br_component_lower_bound_ug_l",
    "haa6br_target_abs_difference_ug_l",
    "target_quality_flag",
    "target_method",
}
NUMERIC_FEATURES = {
    "month",
    "toc_is_censored",
    "bromide_is_censored",
    "toc_mg_l_half_mrl",
    "bromide_ug_l_half_mrl",
    "toc_mg_l_mrl_sqrt2",
    "bromide_ug_l_mrl_sqrt2",
    "water_temperature_c",
    "free_chlorine_mg_l",
}
GATE_FEATURES = [
    "abs_local_mean",
    "local_sd",
    "abs_prior_mean",
    "abs_correction",
    "weight",
    "posterior_sd",
    "calibration_sample_count",
]


@dataclass(frozen=True)
class Paths:
    data_package: Path
    output: Path
    tables: Path
    figures: Path
    locks: Path


@dataclass
class PriorFit:
    model: Pipeline | None
    features: list[str]
    global_mean: float
    tau2: float
    sigma2: float

    def predict_mean(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None or not self.features:
            return np.full(len(frame), self.global_mean, dtype=float)
        return np.asarray(self.model.predict(frame[self.features]), dtype=float)

    def weight(self, k: int) -> float:
        return float((k * self.tau2) / max(k * self.tau2 + self.sigma2, 1e-12))

    def posterior_sd(self, k: int) -> float:
        variance = (self.tau2 * self.sigma2) / max(k * self.tau2 + self.sigma2, 1e-12)
        return float(np.sqrt(max(variance, 1e-12)))


@dataclass
class GateFit:
    model: Pipeline | None
    threshold: float
    source_cv_probability: pd.Series
    source_negative_transfer_rate: float
    source_mean_delta: float
    source_adaptation_rate: float

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(frame), dtype=float)
        return self.model.predict_proba(frame[GATE_FEATURES])[:, 1]


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_integrated_v1(
    data_package: Path,
    required_files: tuple[Path, ...] = INTEGRATED_V1_REQUIRED_FILES,
) -> Path:
    root = data_package.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Integrated-v1 data package does not exist: {root}")
    missing = [str(path) for path in required_files if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Integrated-v1 data package is incomplete; missing relative paths: "
            f"{missing}"
        )
    return root


def split_features(features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    categorical = [feature for feature in features if feature not in numeric]
    return categorical, numeric


def feature_preprocessor(features: list[str], min_frequency: int = 10) -> ColumnTransformer:
    categorical, numeric = split_features(features)
    transformers = []
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    return ColumnTransformer(transformers, remainder="drop")


def build_source_model(name: str, features: list[str], seed: int) -> Pipeline:
    preprocessor = feature_preprocessor(features)
    if name == "Median":
        return Pipeline([("preprocess", preprocessor), ("model", DummyRegressor(strategy="median"))])
    if name == "ElasticNet":
        return Pipeline(
            [
                ("preprocess", preprocessor),
                ("scale", StandardScaler(with_mean=False)),
                (
                    "model",
                    ElasticNet(
                        alpha=0.02,
                        l1_ratio=0.1,
                        max_iter=800,
                        tol=1e-3,
                        selection="random",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name in {"XGBoost CUDA MAE", "XGBoost CUDA log1p"}:
        regressor = XGBRegressor(
            objective="reg:absoluteerror" if name.endswith("MAE") else "reg:squarederror",
            n_estimators=350,
            learning_rate=0.045,
            max_depth=7,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            tree_method="hist",
            device="cuda",
            random_state=seed,
            n_jobs=-1,
        )
        model = (
            regressor
            if name.endswith("MAE")
            else TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
        )
        return Pipeline([("preprocess", preprocessor), ("model", model)])
    raise ValueError(f"Unknown source model: {name}")


def clip_prediction(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, None)


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    error = pred - y
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(r2_score(y, pred)),
        "medae": float(np.median(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def finite_quantile(values: np.ndarray, coverage: float = 0.90) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    level = min(1.0, np.ceil((len(clean) + 1) * coverage) / len(clean))
    return float(np.quantile(clean, level, method="higher"))


def assign_rounds(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["sample_date"] = pd.to_datetime(output["sample_date"], errors="coerce")
    output = output.loc[output["sample_date"].notna()].copy()
    output.sort_values([GROUP, "sample_date", "sample_id"], inplace=True)
    output["round_index"] = (
        output.groupby(GROUP, sort=False)["sample_date"].rank(method="dense").astype(int)
    )
    output["system_rounds"] = output.groupby(GROUP)["round_index"].transform("max").astype(int)
    return output.reset_index(drop=True)


def make_round_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"base_prediction", TARGET, GROUP, "sample_date", "round_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Round table missing columns: {sorted(missing)}")
    work = frame.copy()
    work["residual"] = work[TARGET].astype(float) - work["base_prediction"].astype(float)
    grouped = (
        work.groupby([GROUP, REGION, "sample_date", "round_index"], as_index=False, dropna=False)
        .agg(
            round_residual=("residual", "mean"),
            round_observed=(TARGET, "mean"),
            round_base=("base_prediction", "mean"),
            round_samples=("sample_id", "size"),
        )
        .sort_values([GROUP, "round_index"])
    )
    return grouped.reset_index(drop=True)


def system_feature_rows(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    columns = [GROUP, *features]
    return (
        frame.sort_values([GROUP, "sample_date", "sample_id"])
        .drop_duplicates(GROUP, keep="first")[columns]
        .reset_index(drop=True)
    )


def prior_pipeline(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", feature_preprocessor(features, min_frequency=5)),
            ("scale", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=50.0)),
        ]
    )


def fit_prior(frame: pd.DataFrame, features: list[str]) -> PriorFit:
    rounds = make_round_table(frame)
    system_means = rounds.groupby(GROUP, as_index=False).agg(
        residual_mean=("round_residual", "mean"),
        n_rounds=("round_index", "size"),
    )
    feature_rows = system_feature_rows(frame, features)
    system_means = system_means.merge(feature_rows, on=GROUP, how="left", validate="one_to_one")
    global_mean = float(system_means["residual_mean"].mean())
    model = None
    predicted_mean = np.full(len(system_means), global_mean, dtype=float)
    if features:
        model = prior_pipeline(features)
        model.fit(system_means[features], system_means["residual_mean"])
        predicted_mean = np.asarray(model.predict(system_means[features]), dtype=float)

    round_with_mean = rounds.merge(
        system_means[[GROUP, "residual_mean"]], on=GROUP, how="left", validate="many_to_one"
    )
    within_ss = np.square(
        round_with_mean["round_residual"].to_numpy(float)
        - round_with_mean["residual_mean"].to_numpy(float)
    ).sum()
    within_df = int((system_means["n_rounds"] - 1).clip(lower=0).sum())
    sigma2 = float(within_ss / max(within_df, 1))
    centered = system_means["residual_mean"].to_numpy(float) - predicted_mean
    observed_between = float(np.var(centered, ddof=1)) if len(centered) > 1 else 0.0
    noise_in_means = float(sigma2 * np.mean(1.0 / system_means["n_rounds"].clip(lower=1)))
    tau2 = float(max(observed_between - noise_in_means, 1e-6))
    return PriorFit(model=model, features=features, global_mean=global_mean, tau2=tau2, sigma2=sigma2)


def target_system_inputs(
    frame: pd.DataFrame,
    prior: PriorFit,
    global_prior: PriorFit,
    k: int,
) -> pd.DataFrame:
    rows = []
    feature_lookup = system_feature_rows(frame, prior.features).set_index(GROUP)
    prior_mean_lookup = pd.Series(
        prior.predict_mean(feature_lookup.reset_index()), index=feature_lookup.index
    )
    for system, subset in frame.groupby(GROUP, sort=False):
        ordered_rounds = make_round_table(subset)
        if ordered_rounds["round_index"].nunique() <= k:
            continue
        calibration = ordered_rounds.loc[ordered_rounds["round_index"] <= k]
        local_mean = float(calibration["round_residual"].mean())
        local_sd = float(calibration["round_residual"].std(ddof=1)) if k > 1 else 0.0
        conditional_mean = float(prior_mean_lookup.loc[system])
        hrc_weight = prior.weight(k)
        global_weight = global_prior.weight(k)
        hrc_correction = conditional_mean + hrc_weight * (local_mean - conditional_mean)
        blup_correction = global_prior.global_mean + global_weight * (
            local_mean - global_prior.global_mean
        )
        rows.append(
            {
                GROUP: system,
                "k": k,
                "local_mean": local_mean,
                "local_sd": local_sd,
                "prior_mean": conditional_mean,
                "hrc_correction": hrc_correction,
                "blup_correction": blup_correction,
                "weight": hrc_weight,
                "posterior_sd": prior.posterior_sd(k),
                "calibration_sample_count": int(
                    subset.loc[subset["round_index"] <= k].shape[0]
                ),
                "abs_local_mean": abs(local_mean),
                "abs_prior_mean": abs(conditional_mean),
                "abs_correction": abs(hrc_correction),
            }
        )
    return pd.DataFrame(rows)


def simulate_held_out_systems(
    held_frame: pd.DataFrame,
    prior: PriorFit,
    global_prior: PriorFit,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = target_system_inputs(held_frame, prior, global_prior, k)
    if inputs.empty:
        return pd.DataFrame(), pd.DataFrame()
    system_rows = []
    sample_rows = []
    for input_row in inputs.to_dict("records"):
        system = input_row[GROUP]
        subset = held_frame.loc[held_frame[GROUP] == system].copy()
        evaluation = subset.loc[subset["round_index"] > k].copy()
        base = evaluation["base_prediction"].to_numpy(float)
        y = evaluation[TARGET].to_numpy(float)
        hrc = clip_prediction(base + input_row["hrc_correction"])
        base_mae = float(np.mean(np.abs(y - base)))
        hrc_mae = float(np.mean(np.abs(y - hrc)))
        system_rows.append(
            {
                **input_row,
                REGION: evaluation[REGION].iloc[0],
                "base_mae": base_mae,
                "hrc_mae": hrc_mae,
                "hrc_delta": hrc_mae - base_mae,
                "hrc_improved": int(hrc_mae < base_mae),
                "evaluation_samples": len(evaluation),
            }
        )
        for sample, base_value, hrc_value, observed in zip(
            evaluation["sample_id"], base, hrc, y, strict=True
        ):
            sample_rows.append(
                {
                    "sample_id": sample,
                    GROUP: system,
                    REGION: evaluation[REGION].iloc[0],
                    "k": k,
                    "base_abs_error": abs(observed - base_value),
                    "hrc_abs_error": abs(observed - hrc_value),
                }
            )
    return pd.DataFrame(system_rows), pd.DataFrame(sample_rows)


def crossfit_source_simulations(
    source: pd.DataFrame,
    prior_features: list[str],
    k_values: tuple[int, ...] = (1, 2, 3),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    systems = source[GROUP].astype(str).to_numpy()
    n_splits = min(3, source[GROUP].nunique())
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    system_frames = []
    sample_frames = []
    for train_index, held_index in splitter.split(source, groups=systems):
        training = source.iloc[train_index].copy()
        held = source.iloc[held_index].copy()
        prior = fit_prior(training, prior_features)
        global_prior = fit_prior(training, [])
        for k in k_values:
            system_result, sample_result = simulate_held_out_systems(
                held, prior, global_prior, k
            )
            if not system_result.empty:
                system_frames.append(system_result)
                sample_frames.append(sample_result)
    return pd.concat(system_frames, ignore_index=True), pd.concat(sample_frames, ignore_index=True)


def gate_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_gate(simulations: pd.DataFrame, k: int) -> GateFit:
    subset = simulations.loc[simulations["k"] == k].copy().reset_index(drop=True)
    if subset.empty or subset["hrc_improved"].nunique() < 2:
        probabilities = pd.Series(np.zeros(len(subset)), index=subset.index, dtype=float)
        return GateFit(None, 1.01, probabilities, 0.0, 0.0, 0.0)

    cv_probability = np.full(len(subset), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=SEED + k)
    for train_index, test_index in splitter.split(subset, groups=subset[GROUP]):
        model = gate_pipeline()
        model.fit(subset.iloc[train_index][GATE_FEATURES], subset.iloc[train_index]["hrc_improved"])
        cv_probability[test_index] = model.predict_proba(subset.iloc[test_index][GATE_FEATURES])[:, 1]

    candidates = [*np.linspace(0.10, 0.90, 17), 1.01]
    scored = []
    delta = subset["hrc_delta"].to_numpy(float)
    for threshold in candidates:
        adapt = cv_probability >= threshold
        gated_delta = np.where(adapt, delta, 0.0)
        negative_rate = float(np.mean(gated_delta > 1e-12))
        mean_delta = float(np.mean(gated_delta))
        adaptation_rate = float(np.mean(adapt))
        if negative_rate <= 0.15 and mean_delta <= 0.0:
            scored.append((mean_delta, negative_rate, -adaptation_rate, threshold))
    if not scored:
        threshold = 1.01
    else:
        threshold = float(min(scored)[3])
    adapt = cv_probability >= threshold
    gated_delta = np.where(adapt, delta, 0.0)
    final_model = gate_pipeline()
    final_model.fit(subset[GATE_FEATURES], subset["hrc_improved"])
    return GateFit(
        model=final_model,
        threshold=threshold,
        source_cv_probability=pd.Series(cv_probability, index=subset.index),
        source_negative_transfer_rate=float(np.mean(gated_delta > 1e-12)),
        source_mean_delta=float(np.mean(gated_delta)),
        source_adaptation_rate=float(np.mean(adapt)),
    )


def select_source_model(
    source: pd.DataFrame,
    features: list[str],
    outer_label: str,
    candidates: tuple[str, ...] = ("Median", "ElasticNet", "XGBoost CUDA MAE"),
) -> tuple[str, np.ndarray, pd.DataFrame]:
    groups = source[GROUP].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    predictions = {name: np.full(len(source), np.nan, dtype=float) for name in candidates}
    fold_rows = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(source[features], source[TARGET], groups), start=1
    ):
        for candidate in candidates:
            model = build_source_model(candidate, features, SEED + fold)
            model.fit(source.iloc[train_index][features], source.iloc[train_index][TARGET])
            prediction = clip_prediction(model.predict(source.iloc[validation_index][features]))
            predictions[candidate][validation_index] = prediction
            fold_rows.append(
                {
                    "outer_label": outer_label,
                    "inner_fold": fold,
                    "model": candidate,
                    "validation_rows": len(validation_index),
                    **metrics(source.iloc[validation_index][TARGET].to_numpy(float), prediction),
                }
            )
    summaries = []
    for candidate in candidates:
        if np.isnan(predictions[candidate]).any():
            raise RuntimeError(f"Missing inner OOF predictions for {outer_label} / {candidate}")
        summaries.append(
            {
                "outer_label": outer_label,
                "model": candidate,
                **metrics(source[TARGET].to_numpy(float), predictions[candidate]),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(["mae", "rmse", "model"])
    selected = str(summary.iloc[0]["model"])
    fold_table = pd.DataFrame(fold_rows).merge(
        summary[["outer_label", "model", "mae"]].rename(columns={"mae": "overall_inner_mae"}),
        on=["outer_label", "model"],
        how="left",
    )
    fold_table["selected"] = fold_table["model"] == selected
    return selected, predictions[selected], fold_table


def add_prediction(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["base_prediction"] = clip_prediction(prediction)
    return output


def apply_methods_to_target(
    target: pd.DataFrame,
    source: pd.DataFrame,
    prior_features: list[str],
    source_simulations: pd.DataFrame,
    source_simulation_samples: pd.DataFrame,
    k: int,
    outer_label: str,
) -> tuple[pd.DataFrame, dict]:
    prior = fit_prior(source, prior_features)
    global_prior = fit_prior(source, [])
    inputs = target_system_inputs(target, prior, global_prior, k)
    if inputs.empty:
        return pd.DataFrame(), {}

    gate = fit_gate(source_simulations, k)
    gate_subset = source_simulations.loc[source_simulations["k"] == k].copy().reset_index(drop=True)
    gate_subset["gate_probability"] = gate.source_cv_probability.to_numpy(float)
    gate_subset["safe_adapt"] = gate_subset["gate_probability"] >= gate.threshold
    sample_calibration = source_simulation_samples.loc[
        source_simulation_samples["k"] == k
    ].merge(
        gate_subset[[GROUP, "safe_adapt"]], on=GROUP, how="left", validate="many_to_one"
    )
    sample_calibration["safe_abs_error"] = np.where(
        sample_calibration["safe_adapt"],
        sample_calibration["hrc_abs_error"],
        sample_calibration["base_abs_error"],
    )
    q_zero = finite_quantile(sample_calibration["base_abs_error"].to_numpy(float))
    q_safe = finite_quantile(sample_calibration["safe_abs_error"].to_numpy(float))

    inputs["gate_probability"] = gate.probability(inputs)
    inputs["safe_adapt"] = inputs["gate_probability"] >= gate.threshold
    input_lookup = inputs.set_index(GROUP)
    rows = []
    for system, subset in target.groupby(GROUP, sort=False):
        if system not in input_lookup.index:
            continue
        info = input_lookup.loc[system]
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        calibration = subset.loc[subset["round_index"] <= k]
        evaluation = subset.loc[subset["round_index"] > k].copy()
        if evaluation.empty:
            continue
        if not calibration["sample_date"].max() < evaluation["sample_date"].min():
            raise AssertionError(f"Calibration/evaluation date leakage for {system} k={k}")
        last_round = int(calibration["round_index"].max())
        last_value = float(
            calibration.loc[calibration["round_index"] == last_round, TARGET].mean()
        )
        history_mean = float(
            calibration.groupby("round_index")[TARGET].mean().mean()
        )
        base = evaluation["base_prediction"].to_numpy(float)
        raw = clip_prediction(base + float(info["local_mean"]))
        blup = clip_prediction(base + float(info["blup_correction"]))
        hrc = clip_prediction(base + float(info["hrc_correction"]))
        safe = hrc if bool(info["safe_adapt"]) else base
        for index, (_, sample) in enumerate(evaluation.iterrows()):
            rows.append(
                {
                    "outer_label": outer_label,
                    REGION: sample[REGION],
                    GROUP: system,
                    "sample_id": sample["sample_id"],
                    "sample_date": sample["sample_date"],
                    "round_index": int(sample["round_index"]),
                    "k": k,
                    "observed": float(sample[TARGET]),
                    "Zero-shot": float(base[index]),
                    "Persistence": last_value,
                    "History mean": history_mean,
                    "Raw residual": float(raw[index]),
                    "BLUP": float(blup[index]),
                    "HRC": float(hrc[index]),
                    "SafeShrink HRC": float(safe[index]),
                    "gate_probability": float(info["gate_probability"]),
                    "safe_adapt": bool(info["safe_adapt"]),
                    "hrc_correction": float(info["hrc_correction"]),
                    "q_zero": q_zero,
                    "q_safe": q_safe,
                }
            )
    diagnostics = {
        "outer_label": outer_label,
        "k": k,
        "prior_tau2": prior.tau2,
        "prior_sigma2": prior.sigma2,
        "prior_weight": prior.weight(k),
        "gate_threshold": gate.threshold,
        "gate_source_negative_transfer_rate": gate.source_negative_transfer_rate,
        "gate_source_mean_delta": gate.source_mean_delta,
        "gate_source_adaptation_rate": gate.source_adaptation_rate,
        "q_zero": q_zero,
        "q_safe": q_safe,
    }
    return pd.DataFrame(rows), diagnostics


def paired_bootstrap(
    frame: pd.DataFrame,
    method: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    work = frame[[GROUP, "observed", "Zero-shot", method]].copy()
    work["base_abs"] = np.abs(work["observed"] - work["Zero-shot"])
    work["method_abs"] = np.abs(work["observed"] - work[method])
    grouped = work.groupby(GROUP).agg(
        n=("observed", "size"),
        base_abs=("base_abs", "sum"),
        method_abs=("method_abs", "sum"),
    )
    rng = np.random.default_rng(seed)
    group_count = len(grouped)
    counts = rng.multinomial(
        group_count, np.full(group_count, 1.0 / group_count), size=n_boot
    )
    denominator = counts @ grouped["n"].to_numpy(float)
    base_mae = (counts @ grouped["base_abs"].to_numpy(float)) / denominator
    method_mae = (counts @ grouped["method_abs"].to_numpy(float)) / denominator
    delta = method_mae - base_mae
    return float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))


def method_summary(frame: pd.DataFrame, evidence_class: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = [
        "Zero-shot",
        "Persistence",
        "History mean",
        "Raw residual",
        "BLUP",
        "HRC",
        "SafeShrink HRC",
    ]
    rows = []
    region_rows = []
    for k in sorted(frame["k"].unique()):
        subset = frame.loc[frame["k"] == k].copy()
        base_mae = float(np.mean(np.abs(subset["observed"] - subset["Zero-shot"])))
        for method in methods:
            values = metrics(subset["observed"].to_numpy(float), subset[method].to_numpy(float))
            system_errors = (
                subset.assign(
                    base_abs=np.abs(subset["observed"] - subset["Zero-shot"]),
                    method_abs=np.abs(subset["observed"] - subset[method]),
                )
                .groupby(GROUP)[["base_abs", "method_abs"]]
                .mean()
            )
            negative_rate = float(
                np.mean(system_errors["method_abs"] > system_errors["base_abs"] + 1e-12)
            )
            ci_low, ci_high = (0.0, 0.0) if method == "Zero-shot" else paired_bootstrap(subset, method)
            if method == "SafeShrink HRC":
                lower = subset[method] - subset["q_safe"]
                upper = subset[method] + subset["q_safe"]
                coverage = float(np.mean((subset["observed"] >= lower) & (subset["observed"] <= upper)))
                width = float(np.mean(2.0 * subset["q_safe"]))
            elif method == "Zero-shot":
                lower = subset[method] - subset["q_zero"]
                upper = subset[method] + subset["q_zero"]
                coverage = float(np.mean((subset["observed"] >= lower) & (subset["observed"] <= upper)))
                width = float(np.mean(2.0 * subset["q_zero"]))
            else:
                coverage = np.nan
                width = np.nan
            rows.append(
                {
                    "analysis": evidence_class,
                    "k": int(k),
                    "method": method,
                    "rows": len(subset),
                    "systems": subset[GROUP].nunique(),
                    "zero_shot_mae": base_mae,
                    **values,
                    "relative_mae_improvement": (base_mae - values["mae"]) / base_mae,
                    "paired_delta_mae": values["mae"] - base_mae,
                    "paired_delta_ci_low": ci_low,
                    "paired_delta_ci_high": ci_high,
                    "negative_transfer_rate": negative_rate,
                    "coverage_90": coverage,
                    "mean_interval_width": width,
                    "adaptation_rate": float(subset["safe_adapt"].mean())
                    if method == "SafeShrink HRC"
                    else np.nan,
                }
            )
            for region, regional in subset.groupby(REGION):
                regional_base = float(
                    np.mean(np.abs(regional["observed"] - regional["Zero-shot"]))
                )
                regional_method = float(
                    np.mean(np.abs(regional["observed"] - regional[method]))
                )
                if method == "SafeShrink HRC":
                    regional_coverage = float(
                        np.mean(
                            (regional["observed"] >= regional[method] - regional["q_safe"])
                            & (regional["observed"] <= regional[method] + regional["q_safe"])
                        )
                    )
                elif method == "Zero-shot":
                    regional_coverage = float(
                        np.mean(
                            (regional["observed"] >= regional[method] - regional["q_zero"])
                            & (regional["observed"] <= regional[method] + regional["q_zero"])
                        )
                    )
                else:
                    regional_coverage = np.nan
                region_rows.append(
                    {
                        "analysis": evidence_class,
                        "k": int(k),
                        REGION: region,
                        "method": method,
                        "rows": len(regional),
                        "systems": regional[GROUP].nunique(),
                        "zero_shot_mae": regional_base,
                        "method_mae": regional_method,
                        "delta_mae": regional_method - regional_base,
                        "relative_improvement": (regional_base - regional_method) / regional_base,
                        "coverage_90": regional_coverage,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(region_rows)


def anova_icc(frame: pd.DataFrame, group_column: str, value_column: str) -> float:
    grouped = frame.groupby(group_column)[value_column]
    sizes = grouped.size().to_numpy(float)
    means = grouped.mean().to_numpy(float)
    if len(means) < 2:
        return np.nan
    grand = float(np.average(means, weights=sizes))
    ss_between = float(np.sum(sizes * np.square(means - grand)))
    ss_within = float(grouped.apply(lambda x: np.square(x - x.mean()).sum()).sum())
    df_between = len(means) - 1
    df_within = int(sizes.sum() - len(means))
    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)
    n0 = (sizes.sum() - np.square(sizes).sum() / sizes.sum()) / max(df_between, 1)
    return float((ms_between - ms_within) / max(ms_between + (n0 - 1) * ms_within, 1e-12))


def residual_mechanism_table(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rounds = make_round_table(oof)
    system_icc = anova_icc(rounds, GROUP, "round_residual")
    region_icc = anova_icc(rounds, REGION, "round_residual")
    pairs = rounds[[GROUP, "round_index", "round_residual"]].copy()
    pairs["next_residual"] = pairs.groupby(GROUP)["round_residual"].shift(-1)
    pairs = pairs.dropna(subset=["next_residual"])
    pearson = pearsonr(pairs["round_residual"], pairs["next_residual"])
    spearman = spearmanr(pairs["round_residual"], pairs["next_residual"])
    summary = pd.DataFrame(
        [
            {
                "system_icc": system_icc,
                "region_icc": region_icc,
                "consecutive_round_pairs": len(pairs),
                "pearson_consecutive_residual": float(pearson.statistic),
                "pearson_p_value": float(pearson.pvalue),
                "spearman_consecutive_residual": float(spearman.statistic),
                "spearman_p_value": float(spearman.pvalue),
            }
        ]
    )
    return summary, pairs


def load_data(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metadata_path = paths.data_package / "metadata" / "model_feature_sets.json"
    feature_sets = json.loads(metadata_path.read_text(encoding="utf-8"))
    core_all = pd.read_csv(paths.data_package / "data" / "us_ucmr4_core.csv", low_memory=False)
    enriched_all = pd.read_csv(
        paths.data_package / "data" / "us_ucmr4_enriched_strict.csv", low_memory=False
    )
    core_all["primary_analysis_eligible"] = parse_bool(core_all["primary_analysis_eligible"])
    enriched_all["primary_analysis_eligible"] = parse_bool(
        enriched_all["primary_analysis_eligible"]
    )
    core = core_all.loc[core_all["primary_analysis_eligible"]].copy()
    enriched = enriched_all.loc[enriched_all["primary_analysis_eligible"]].copy()
    for frame in (core, enriched):
        frame[REGION] = pd.to_numeric(frame[REGION], errors="raise").astype(int)
        if frame[GROUP].isna().any() or frame[TARGET].isna().any():
            raise AssertionError("Primary U.S. data have missing group or target")
    all_features = set(feature_sets["transportable_core"]) | set(
        feature_sets["us_operational_core"]
    ) | set(feature_sets["us_enriched_primary"])
    if not all_features.isdisjoint(FORBIDDEN):
        raise AssertionError("Feature whitelist contains forbidden target-derived variables")
    return assign_rounds(core), assign_rounds(enriched), core_all, feature_sets


def eligibility_table(core: pd.DataFrame, dwi_path: Path) -> pd.DataFrame:
    rows = []
    for k in (0, 1, 2, 3):
        eligible = core[GROUP].nunique() if k == 0 else core.loc[core["system_rounds"] > k, GROUP].nunique()
        evaluation_rows = len(core) if k == 0 else int((core["round_index"] > k).sum())
        rows.append(
            {
                "dataset": "UCMR4 primary 2018-2020",
                "k": k,
                "eligible_systems": int(eligible),
                "evaluation_rows": evaluation_rows,
                "date_missing_rate": 0.0,
            }
        )
    dwi = assign_rounds(
        pd.read_csv(dwi_path, usecols=["sample_id", GROUP, "sample_date"], low_memory=False)
    )
    for k in (0, 1, 2, 3):
        eligible = dwi[GROUP].nunique() if k == 0 else dwi.loc[dwi["system_rounds"] > k, GROUP].nunique()
        evaluation_rows = len(dwi) if k == 0 else int((dwi["round_index"] > k).sum())
        rows.append(
            {
                "dataset": "DWI242 retrospective",
                "k": k,
                "eligible_systems": int(eligible),
                "evaluation_rows": evaluation_rows,
                "date_missing_rate": 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_core_loro(
    core: pd.DataFrame,
    features: list[str],
    prior_features: list[str],
) -> dict[str, pd.DataFrame]:
    target_frames = []
    full_oof_frames = []
    selection_frames = []
    diagnostic_rows = []
    split_rows = []
    regions = sorted(core[REGION].unique())
    for region_index, held_region in enumerate(regions, start=1):
        print(f"[core {region_index}/{len(regions)}] held EPA region {held_region}", flush=True)
        source = core.loc[core[REGION] != held_region].copy().reset_index(drop=True)
        target = core.loc[core[REGION] == held_region].copy().reset_index(drop=True)
        overlap = set(source[GROUP]).intersection(target[GROUP])
        if overlap:
            raise AssertionError(f"Source/target system overlap in region {held_region}: {len(overlap)}")
        outer_label = f"EPA-{held_region}"
        selected, source_oof_prediction, selection = select_source_model(
            source, features, outer_label
        )
        selection_frames.append(selection)
        source_with_prediction = add_prediction(source, source_oof_prediction)
        model = build_source_model(selected, features, SEED + 100 + int(held_region))
        model.fit(source[features], source[TARGET])
        target_prediction = clip_prediction(model.predict(target[features]))
        target_with_prediction = add_prediction(target, target_prediction)
        full_oof_frames.append(target_with_prediction)

        simulations, simulation_samples = crossfit_source_simulations(
            source_with_prediction, prior_features
        )
        for k in (1, 2, 3):
            result, diagnostics = apply_methods_to_target(
                target_with_prediction,
                source_with_prediction,
                prior_features,
                simulations,
                simulation_samples,
                k,
                outer_label,
            )
            if not result.empty:
                target_frames.append(result)
                diagnostic_rows.append({"selected_source_model": selected, **diagnostics})

        for row in target_with_prediction[["sample_id", GROUP, REGION, "sample_date", "round_index"]].to_dict("records"):
            split_rows.append(
                {
                    **row,
                    "outer_label": outer_label,
                    "role": "target",
                    "source_target_system_overlap": 0,
                }
            )

    prediction_detail = pd.concat(target_frames, ignore_index=True)
    full_oof = pd.concat(full_oof_frames, ignore_index=True).sort_index()
    summary, regions = method_summary(prediction_detail, "Confirmatory U.S. EPA-region LORO")
    return {
        "predictions": prediction_detail,
        "full_oof": full_oof,
        "summary": summary,
        "regions": regions,
        "selection": pd.concat(selection_frames, ignore_index=True),
        "diagnostics": pd.DataFrame(diagnostic_rows),
        "split_audit": pd.DataFrame(split_rows),
    }


def group_oof_fixed_model(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str = "XGBoost CUDA MAE",
    n_splits: int = 5,
) -> np.ndarray:
    prediction = np.full(len(frame), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame[features], frame[TARGET], frame[GROUP]), start=1
    ):
        model = build_source_model(model_name, features, SEED + 500 + fold)
        model.fit(frame.iloc[train_index][features], frame.iloc[train_index][TARGET])
        prediction[test_index] = clip_prediction(model.predict(frame.iloc[test_index][features]))
    if np.isnan(prediction).any():
        raise RuntimeError("Fixed source OOF prediction is incomplete")
    return prediction


def run_uk_pressure_test(
    core: pd.DataFrame,
    transport_features: list[str],
    method_lock_path: Path,
    data_package: Path,
) -> dict[str, pd.DataFrame]:
    if not method_lock_path.exists():
        raise AssertionError("Method lock must exist before loading U.K. outcomes")
    source_oof = group_oof_fixed_model(core, transport_features)
    source = add_prediction(core, source_oof)
    prior_features = [
        "source_water_type_std",
        "sample_point_type_std",
        "disinfectant_type_codes",
    ]
    simulations, simulation_samples = crossfit_source_simulations(source, prior_features, (1, 2))

    dwi = pd.read_csv(data_package / "data" / "uk_dwi242_locked_external.csv", low_memory=False)
    dwi[REGION] = "DWI242"
    dwi = assign_rounds(dwi)
    final_model = build_source_model("XGBoost CUDA MAE", transport_features, SEED + 900)
    final_model.fit(core[transport_features], core[TARGET])
    dwi = add_prediction(dwi, final_model.predict(dwi[transport_features]))
    result_frames = []
    diagnostic_rows = []
    for k in (1, 2):
        result, diagnostics = apply_methods_to_target(
            dwi,
            source,
            prior_features,
            simulations,
            simulation_samples,
            k,
            "UK-DWI242",
        )
        result_frames.append(result)
        diagnostic_rows.append(diagnostics)
    predictions = pd.concat(result_frames, ignore_index=True)
    summary, systems = method_summary(predictions.rename(columns={REGION: "_region"}).assign(**{REGION: "DWI242"}), "Retrospective U.K. pressure test")
    system_rows = []
    for k in (1, 2):
        subset = predictions.loc[predictions["k"] == k]
        for system, system_frame in subset.groupby(GROUP):
            base_mae = float(np.mean(np.abs(system_frame["observed"] - system_frame["Zero-shot"])))
            safe_mae = float(
                np.mean(np.abs(system_frame["observed"] - system_frame["SafeShrink HRC"]))
            )
            system_rows.append(
                {
                    "k": k,
                    GROUP: system,
                    "zero_shot_mae": base_mae,
                    "safe_mae": safe_mae,
                    "delta_mae": safe_mae - base_mae,
                    "improved_or_equal": int(safe_mae <= base_mae + 1e-12),
                }
            )

    dbp = pd.read_csv(data_package / "data" / "uk_dbp2009_field_external.csv", low_memory=False)
    dbp_prediction = clip_prediction(final_model.predict(dbp[transport_features]))
    dbp_summary = pd.DataFrame(
        [
            {
                "dataset": "DBP2009 field",
                "evidence_class": "External zero-shot sensitivity; dates unavailable",
                "rows": len(dbp),
                "systems": dbp[GROUP].nunique(),
                **metrics(dbp[TARGET].to_numpy(float), dbp_prediction),
            }
        ]
    )
    return {
        "predictions": predictions,
        "summary": summary,
        "systems": pd.DataFrame(system_rows),
        "diagnostics": pd.DataFrame(diagnostic_rows),
        "dbp_summary": dbp_summary,
    }


def apply_two_priors(
    target: pd.DataFrame,
    source: pd.DataFrame,
    base_prior_features: list[str],
    chemistry_prior_features: list[str],
    k: int,
    outer_label: str,
) -> pd.DataFrame:
    base_prior = fit_prior(source, base_prior_features)
    chemistry_prior = fit_prior(source, chemistry_prior_features)
    global_prior = fit_prior(source, [])
    base_inputs = target_system_inputs(target, base_prior, global_prior, k).set_index(GROUP)
    chemistry_inputs = target_system_inputs(target, chemistry_prior, global_prior, k).set_index(GROUP)
    rows = []
    common_systems = base_inputs.index.intersection(chemistry_inputs.index)
    for system in common_systems:
        subset = target.loc[(target[GROUP] == system) & (target["round_index"] > k)].copy()
        if subset.empty:
            continue
        base = subset["base_prediction"].to_numpy(float)
        hrc = clip_prediction(base + float(base_inputs.loc[system, "hrc_correction"]))
        cg = clip_prediction(base + float(chemistry_inputs.loc[system, "hrc_correction"]))
        for index, (_, sample) in enumerate(subset.iterrows()):
            rows.append(
                {
                    "outer_label": outer_label,
                    REGION: sample[REGION],
                    GROUP: system,
                    "sample_id": sample["sample_id"],
                    "k": k,
                    "observed": float(sample[TARGET]),
                    "Zero-shot": float(base[index]),
                    "HRC": float(hrc[index]),
                    "CG-HRC": float(cg[index]),
                }
            )
    return pd.DataFrame(rows)


def run_enriched_track(
    enriched: pd.DataFrame,
    features: list[str],
    base_prior_features: list[str],
    chemistry_prior_features: list[str],
) -> dict[str, pd.DataFrame]:
    all_predictions = []
    selection_frames = []
    for region_index, held_region in enumerate(sorted(enriched[REGION].unique()), start=1):
        print(f"[enriched {region_index}/10] held EPA region {held_region}", flush=True)
        source = enriched.loc[enriched[REGION] != held_region].copy().reset_index(drop=True)
        target = enriched.loc[enriched[REGION] == held_region].copy().reset_index(drop=True)
        _, source_oof, selection = select_source_model(
            source,
            features,
            f"Enriched-EPA-{held_region}",
            candidates=("XGBoost CUDA MAE",),
        )
        selection_frames.append(selection)
        source = add_prediction(source, source_oof)
        model = build_source_model("XGBoost CUDA MAE", features, SEED + 1200 + int(held_region))
        model.fit(source[features], source[TARGET])
        target = add_prediction(target, model.predict(target[features]))
        for k in (1, 2):
            result = apply_two_priors(
                target,
                source,
                base_prior_features,
                chemistry_prior_features,
                k,
                f"Enriched-EPA-{held_region}",
            )
            if not result.empty:
                all_predictions.append(result)
    predictions = pd.concat(all_predictions, ignore_index=True)
    rows = []
    region_rows = []
    for k in (1, 2):
        subset = predictions.loc[predictions["k"] == k]
        for method in ("Zero-shot", "HRC", "CG-HRC"):
            values = metrics(subset["observed"], subset[method])
            rows.append(
                {
                    "k": k,
                    "method": method,
                    "rows": len(subset),
                    "systems": subset[GROUP].nunique(),
                    **values,
                }
            )
        for region, regional in subset.groupby(REGION):
            hrc_mae = float(np.mean(np.abs(regional["observed"] - regional["HRC"])))
            cg_mae = float(np.mean(np.abs(regional["observed"] - regional["CG-HRC"])))
            region_rows.append(
                {
                    "k": k,
                    REGION: region,
                    "hrc_mae": hrc_mae,
                    "cg_hrc_mae": cg_mae,
                    "delta_cg_vs_hrc": cg_mae - hrc_mae,
                    "cg_better": int(cg_mae < hrc_mae),
                }
            )
    summary = pd.DataFrame(rows)
    hrc_k2 = float(summary.loc[(summary["k"] == 2) & (summary["method"] == "HRC"), "mae"].iloc[0])
    summary["relative_improvement_vs_hrc_k2"] = np.where(
        summary["k"] == 2, (hrc_k2 - summary["mae"]) / hrc_k2, np.nan
    )
    return {
        "predictions": predictions,
        "summary": summary,
        "regions": pd.DataFrame(region_rows),
        "selection": pd.concat(selection_frames, ignore_index=True),
    }


def temporal_2021_sensitivity(
    core: pd.DataFrame,
    core_all: pd.DataFrame,
    features: list[str],
    prior_features: list[str],
) -> pd.DataFrame:
    sensitivity_flag = parse_bool(core_all["is_2021_sensitivity"])
    future = core_all.loc[sensitivity_flag].copy()
    future[REGION] = pd.to_numeric(future[REGION], errors="coerce")
    model = build_source_model("XGBoost CUDA MAE", features, SEED + 1500)
    model.fit(core[features], core[TARGET])
    prediction = clip_prediction(model.predict(future[features]))
    rows = [
        {
            "analysis": "2021 zero-shot temporal sensitivity",
            "rows": len(future),
            "systems": future[GROUP].nunique(),
            **metrics(future[TARGET].to_numpy(float), prediction),
        }
    ]
    historical_systems = set(core[GROUP]).intersection(future[GROUP])
    if historical_systems:
        history = core.loc[core[GROUP].isin(historical_systems)].copy()
        history_prediction = clip_prediction(model.predict(history[features]))
        history = add_prediction(history, history_prediction)
        prior = fit_prior(history, prior_features)
        global_prior = fit_prior(history, [])
        inputs = target_system_inputs(history, prior, global_prior, 2).set_index(GROUP)
        future_subset = future.loc[future[GROUP].isin(inputs.index)].copy()
        future_base = clip_prediction(model.predict(future_subset[features]))
        future_hrc = future_base.copy()
        for system, indices in future_subset.groupby(GROUP).groups.items():
            correction = float(inputs.loc[system, "hrc_correction"])
            positions = future_subset.index.get_indexer(indices)
            future_hrc[positions] = clip_prediction(future_base[positions] + correction)
        rows.append(
            {
                "analysis": "2021 fixed k=2 HRC on systems with prior history",
                "rows": len(future_subset),
                "systems": future_subset[GROUP].nunique(),
                **metrics(future_subset[TARGET].to_numpy(float), future_hrc),
            }
        )
    return pd.DataFrame(rows)


def robustness_table(core_predictions: pd.DataFrame) -> pd.DataFrame:
    subset = core_predictions.loc[core_predictions["k"] == 2].copy()
    settings = {
        "Raw residual correction": "Raw residual",
        "Global-prior BLUP": "BLUP",
        "Conditional HRC": "HRC",
        "SafeShrink HRC": "SafeShrink HRC",
        "Persistence monitoring baseline": "Persistence",
        "Historical-mean monitoring baseline": "History mean",
    }
    base_mae = float(np.mean(np.abs(subset["observed"] - subset["Zero-shot"])))
    rows = []
    for setting, column in settings.items():
        score = float(np.mean(np.abs(subset["observed"] - subset[column])))
        rows.append(
            {
                "setting": setting,
                "zero_shot_mae": base_mae,
                "method_mae": score,
                "relative_improvement": (base_mae - score) / base_mae,
                "improvement_direction": int(score < base_mae),
            }
        )
    return pd.DataFrame(rows)


def acceptance_table(
    core_summary: pd.DataFrame,
    core_regions: pd.DataFrame,
    uk_summary: pd.DataFrame,
    uk_systems: pd.DataFrame,
    robustness: pd.DataFrame,
    leakage_passed: bool,
) -> pd.DataFrame:
    main = core_summary.loc[core_summary["method"] == "SafeShrink HRC"].set_index("k")
    zero = core_summary.loc[core_summary["method"] == "Zero-shot"].set_index("k")
    region_k2 = core_regions.loc[
        (core_regions["k"] == 2) & (core_regions["method"] == "SafeShrink HRC")
    ]
    uk_k2 = uk_summary.loc[
        (uk_summary["k"] == 2) & (uk_summary["method"] == "SafeShrink HRC")
    ].iloc[0]
    uk_nonworse = int(uk_systems.loc[uk_systems["k"] == 2, "improved_or_equal"].sum())
    robustness_direction = float(robustness["improvement_direction"].mean())
    width_inflation = float(
        (main.loc[2, "mean_interval_width"] - zero.loc[2, "mean_interval_width"])
        / zero.loc[2, "mean_interval_width"]
    )
    gates = [
        ("G01", "Leakage audit passed", ">=", 1.0, None, float(leakage_passed), "Confirmatory", True),
        ("G02", "U.S. k=1 relative MAE improvement", ">=", 0.10, None, float(main.loc[1, "relative_mae_improvement"]), "Confirmatory", True),
        ("G03", "U.S. k=2 relative MAE improvement", ">=", 0.15, None, float(main.loc[2, "relative_mae_improvement"]), "Confirmatory", True),
        ("G04", "EPA regions improved at k=2", ">=", 8.0, None, float((region_k2["delta_mae"] < 0).sum()), "Confirmatory", True),
        ("G05", "Target-system negative transfer rate", "<=", None, 0.15, float(main.loc[2, "negative_transfer_rate"]), "Confirmatory", True),
        ("G06", "k=2 paired delta MAE CI upper", "<", None, 0.0, float(main.loc[2, "paired_delta_ci_high"]), "Confirmatory", True),
        ("G07", "90% interval overall coverage", "range", 0.85, 0.95, float(main.loc[2, "coverage_90"]), "Confirmatory", True),
        ("G08", "Worst EPA-region coverage", ">=", 0.75, None, float(region_k2["coverage_90"].min()), "Confirmatory", True),
        ("G09", "Interval width inflation vs zero-shot", "<=", None, 0.20, width_inflation, "Confirmatory", True),
        ("G10", "DWI242 k=2 MAE improvement", ">=", 0.10, None, float(uk_k2["relative_mae_improvement"]), "Retrospective support", False),
        ("G11", "DWI242 improved/non-worse plants", ">=", 14.0, None, float(uk_nonworse), "Retrospective support", False),
        ("G12", "Robustness settings preserving direction", ">=", 0.80, None, robustness_direction, "Confirmatory", True),
        ("G13", "New independent locked external data", ">=", 1.0, None, 0.0, "Top-journal bonus", False),
    ]
    rows = []
    for gate, metric_name, direction, lower, upper, actual, evidence, core_gate in gates:
        if direction == ">=":
            passed = actual >= float(lower)
        elif direction == "<=":
            passed = actual <= float(upper)
        elif direction == "<":
            passed = actual < float(upper)
        else:
            passed = float(lower) <= actual <= float(upper)
        rows.append(
            {
                "gate": gate,
                "evidence_class": evidence,
                "metric": metric_name,
                "direction": direction,
                "lower": lower,
                "upper": upper,
                "actual": actual,
                "status": "Passed" if passed else "Not passed",
                "is_core": core_gate,
            }
        )
    return pd.DataFrame(rows)


def publication_scenario(acceptance: pd.DataFrame, core_summary: pd.DataFrame, uk_summary: pd.DataFrame) -> pd.DataFrame:
    core_gates = acceptance.loc[acceptance["is_core"]]
    k1 = core_summary.loc[
        (core_summary["k"] == 1) & (core_summary["method"] == "SafeShrink HRC")
    ].iloc[0]
    k2 = core_summary.loc[
        (core_summary["k"] == 2) & (core_summary["method"] == "SafeShrink HRC")
    ].iloc[0]
    uk_k2 = uk_summary.loc[
        (uk_summary["k"] == 2) & (uk_summary["method"] == "SafeShrink HRC")
    ].iloc[0]
    if (core_gates["status"] == "Passed").all():
        if (
            k1["relative_mae_improvement"] >= 0.15
            and k2["relative_mae_improvement"] >= 0.20
            and k2["negative_transfer_rate"] < 0.10
            and uk_k2["relative_mae_improvement"] >= 0.15
        ):
            scenario = "A: exceptionally strong internal effect"
            level = "Top Q1 competitive, but still limited by retrospective external evidence"
        else:
            scenario = "B: core thresholds met"
            level = "Strong Q2; Q1 can be attempted"
    elif k2["relative_mae_improvement"] >= 0.08:
        scenario = "C: partial U.S. improvement"
        level = "Q2/Q3 depending on consistency and method contribution"
    else:
        scenario = "D: calibration still unsuccessful"
        level = "Q3 or transportability/uncertainty boundary paper"
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "realistic_level": level,
                "core_gates_passed": int((core_gates["status"] == "Passed").sum()),
                "core_gates_total": len(core_gates),
                "us_k1_improvement": float(k1["relative_mae_improvement"]),
                "us_k2_improvement": float(k2["relative_mae_improvement"]),
                "us_k2_negative_transfer": float(k2["negative_transfer_rate"]),
                "uk_k2_improvement": float(uk_k2["relative_mae_improvement"]),
                "external_evidence_note": "DWI242 is retrospective because its outcomes were viewed previously; G13 remains not passed.",
            }
        ]
    )


def save_table(frame: pd.DataFrame, paths: Paths, name: str) -> None:
    frame.to_csv(paths.tables / f"{name}.csv", index=False, encoding="utf-8-sig")


def plot_results(
    paths: Paths,
    eligibility: pd.DataFrame,
    mechanism: pd.DataFrame,
    residual_pairs: pd.DataFrame,
    core_summary: pd.DataFrame,
    core_regions: pd.DataFrame,
    core_predictions: pd.DataFrame,
    uk_systems: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    us_eligibility = eligibility.loc[eligibility["dataset"].str.startswith("UCMR4")]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(us_eligibility["k"].astype(str), us_eligibility["eligible_systems"], color="#2878B5")
    ax.set(title="U.S. systems eligible for chronological calibration", xlabel="Calibration rounds (k)", ylabel="Systems")
    for x, value in enumerate(us_eligibility["eligible_systems"]):
        ax.text(x, value + 60, f"{int(value):,}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig02_eligibility.png", dpi=180)
    plt.close(fig)

    sample_pairs = residual_pairs.sample(min(8000, len(residual_pairs)), random_state=SEED)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(sample_pairs["round_residual"], sample_pairs["next_residual"], s=7, alpha=0.18, color="#2878B5")
    limit = float(np.nanquantile(np.abs(sample_pairs[["round_residual", "next_residual"]]), 0.98))
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Residual at round t (ug/L)", ylabel="Residual at next round (ug/L)")
    ax.set_title(
        f"Residual persistence: system ICC={mechanism.iloc[0]['system_icc']:.3f}, "
        f"Spearman={mechanism.iloc[0]['spearman_consecutive_residual']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(paths.figures / "fig03_residual_persistence.png", dpi=180)
    plt.close(fig)

    selected_methods = ["Zero-shot", "Persistence", "Raw residual", "BLUP", "HRC", "SafeShrink HRC"]
    learning = core_summary.loc[core_summary["method"].isin(selected_methods)].copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for method, subset in learning.groupby("method"):
        subset = subset.sort_values("k")
        ax.plot(subset["k"], subset["mae"], marker="o", linewidth=2, label=method)
    ax.set(title="Chronological few-shot learning curve", xlabel="Calibration rounds (k)", ylabel="MAE (ug/L)", xticks=[1, 2, 3])
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig04_learning_curve.png", dpi=180)
    plt.close(fig)

    region = core_regions.loc[
        (core_regions["k"] == 2) & (core_regions["method"] == "SafeShrink HRC")
    ].sort_values("delta_mae")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    colors = np.where(region["delta_mae"] < 0, "#2E8B57", "#C44E52")
    ax.barh(region[REGION].astype(str), region["delta_mae"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set(title="SafeShrink HRC k=2 change versus zero-shot", xlabel="Delta MAE (method - zero-shot, ug/L)", ylabel="Held-out EPA region")
    fig.tight_layout()
    fig.savefig(paths.figures / "fig05_region_delta.png", dpi=180)
    plt.close(fig)

    k2 = core_predictions.loc[core_predictions["k"] == 2].copy()
    system_delta = (
        k2.assign(
            base_abs=np.abs(k2["observed"] - k2["Zero-shot"]),
            safe_abs=np.abs(k2["observed"] - k2["SafeShrink HRC"]),
            hrc_abs=np.abs(k2["observed"] - k2["HRC"]),
        )
        .groupby(GROUP)[["base_abs", "safe_abs", "hrc_abs"]]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for column, label, color in (
        ("hrc_abs", "Always HRC", "#C44E52"),
        ("safe_abs", "SafeShrink HRC", "#2878B5"),
    ):
        values = np.sort(system_delta[column] - system_delta["base_abs"])
        ax.plot(values, np.linspace(0, 1, len(values), endpoint=True), label=label, color=color)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set(title="System-level negative-transfer distribution", xlabel="Delta system MAE (ug/L)", ylabel="Cumulative fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "fig06_negative_transfer_cdf.png", dpi=180)
    plt.close(fig)

    coverage = region.sort_values(REGION)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(coverage[REGION].astype(str), coverage["coverage_90"], color="#2878B5")
    ax.axhspan(0.85, 0.95, color="#9ACD9A", alpha=0.28, label="Overall target range")
    ax.axhline(0.75, color="#C44E52", linestyle="--", label="Worst-region gate")
    ax.set(title="Local 90% interval coverage after k=2 calibration", xlabel="Held-out EPA region", ylabel="Coverage", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig07_region_coverage.png", dpi=180)
    plt.close(fig)

    uk = uk_systems.loc[uk_systems["k"] == 2].sort_values("delta_mae")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.barh(np.arange(len(uk)), uk["delta_mae"], color=np.where(uk["delta_mae"] <= 0, "#2E8B57", "#C44E52"))
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set(yticks=np.arange(len(uk)), yticklabels=[str(value).replace("UK::", "") for value in uk[GROUP]], title="DWI242 retrospective k=2 plant changes", xlabel="Delta MAE (ug/L)", ylabel="Plant")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig08_dwi242_plants.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the historical HAA6Br few-shot calibration benchmark"
    )
    parser.add_argument(
        "--data-package",
        type=Path,
        required=True,
        help="Path to a cleaned haa6br_integrated_v1 package; no raw data are accepted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / OUTPUT_NAME,
        help="Directory for generated tables, figures, and protocol locks",
    )
    parser.add_argument(
        "--protocol-plan",
        type=Path,
        help="Optional protocol-plan file to hash into the run lock",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("The new protocol enables CUDA, but no CUDA device is available")

    data_package = validate_integrated_v1(args.data_package)
    output = args.output_dir.expanduser().resolve()
    paths = Paths(
        data_package=data_package,
        output=output,
        tables=output / "tables",
        figures=output / "figures",
        locks=output / "locks",
    )
    for directory in (paths.output, paths.tables, paths.figures, paths.locks):
        directory.mkdir(parents=True, exist_ok=True)

    plan_path = args.protocol_plan.expanduser().resolve() if args.protocol_plan else None
    if plan_path is not None and not plan_path.is_file():
        raise FileNotFoundError(f"Protocol plan does not exist: {plan_path}")
    core_path = data_package / "data" / "us_ucmr4_core.csv"
    enriched_path = data_package / "data" / "us_ucmr4_enriched_strict.csv"
    protocol_lock = {
        "protocol": "HAA6Br chronological few-shot calibration SAP v2",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "plan_sha256": sha256_file(plan_path) if plan_path is not None else None,
        "plan_status": "author-supplied" if plan_path is not None else "not supplied",
        "core_data_sha256": sha256_file(core_path),
        "enriched_data_sha256": sha256_file(enriched_path),
        "primary_k": [1, 2],
        "sensitivity_k": [3],
        "outer_split": "Leave-one-EPA-region-out",
        "inner_split": "3-fold GroupKFold by system",
        "forbidden_predictors": sorted(FORBIDDEN),
        "uk_role": "Retrospective pressure test; not a newly locked confirmation",
    }
    (paths.locks / "protocol_lock.json").write_text(
        json.dumps(protocol_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    core, enriched, core_all, feature_sets = load_data(paths)
    operational_features = list(feature_sets["us_operational_core"])
    transport_features = list(feature_sets["transportable_core"])
    enriched_features = list(feature_sets["us_enriched_primary"])
    core_prior_features = [
        "system_size_code",
        "source_water_type_std",
        "treatment_information_codes",
        "disinfectant_type_codes",
        "disinfectant_residual_codes",
    ]
    chemistry_prior_features = [
        *core_prior_features,
        "toc_mg_l_half_mrl",
        "bromide_ug_l_half_mrl",
        "toc_is_censored",
        "bromide_is_censored",
    ]

    dwi_path = data_package / "data" / "uk_dwi242_locked_external.csv"
    eligibility = eligibility_table(core, dwi_path)
    us_k1 = int(eligibility.loc[(eligibility["dataset"].str.startswith("UCMR4")) & (eligibility["k"] == 1), "eligible_systems"].iloc[0])
    us_k2 = int(eligibility.loc[(eligibility["dataset"].str.startswith("UCMR4")) & (eligibility["k"] == 2), "eligible_systems"].iloc[0])
    dwi_k2 = int(eligibility.loc[(eligibility["dataset"].str.startswith("DWI")) & (eligibility["k"] == 2), "eligible_systems"].iloc[0])
    if us_k1 < 4700 or us_k2 < 3000 or dwi_k2 != 20:
        raise AssertionError(f"Eligibility gate failed: U.S. k1={us_k1}, k2={us_k2}, DWI k2={dwi_k2}")

    core_results = run_core_loro(core, operational_features, core_prior_features)
    mechanism, residual_pairs = residual_mechanism_table(core_results["full_oof"])

    enriched_results = run_enriched_track(
        enriched,
        enriched_features,
        core_prior_features,
        chemistry_prior_features,
    )

    method_lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_lock_sha256": sha256_file(paths.locks / "protocol_lock.json"),
        "transport_features": transport_features,
        "uk_source_model": "XGBoost CUDA MAE",
        "prior_features": [
            "source_water_type_std",
            "sample_point_type_std",
            "disinfectant_type_codes",
        ],
        "hrc_formula": "m(z)+w*(local_round_mean_residual-m(z))",
        "gate_selection": "Source-only cross-fitted logistic gate; negative-transfer constraint <=15%",
        "uk_outcomes_used_for_selection": False,
        "us_core_result_hash": hashlib.sha256(
            core_results["summary"].to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    method_lock_path = paths.locks / "method_lock_before_uk.json"
    method_lock_path.write_text(
        json.dumps(method_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    uk_results = run_uk_pressure_test(
        core, transport_features, method_lock_path, data_package
    )
    temporal = temporal_2021_sensitivity(
        core, core_all, operational_features, core_prior_features
    )
    robustness = robustness_table(core_results["predictions"])
    acceptance = acceptance_table(
        core_results["summary"],
        core_results["regions"],
        uk_results["summary"],
        uk_results["systems"],
        robustness,
        leakage_passed=True,
    )
    scenario = publication_scenario(
        acceptance, core_results["summary"], uk_results["summary"]
    )

    tables = {
        "eligibility": eligibility,
        "source_model_selection": core_results["selection"],
        "core_predictions": core_results["predictions"],
        "core_full_oof": core_results["full_oof"],
        "core_summary": core_results["summary"],
        "core_region_summary": core_results["regions"],
        "core_method_diagnostics": core_results["diagnostics"],
        "split_audit": core_results["split_audit"],
        "residual_mechanism": mechanism,
        "residual_round_pairs": residual_pairs,
        "enriched_predictions": enriched_results["predictions"],
        "enriched_summary": enriched_results["summary"],
        "enriched_region_summary": enriched_results["regions"],
        "enriched_model_selection": enriched_results["selection"],
        "uk_predictions": uk_results["predictions"],
        "uk_summary": uk_results["summary"],
        "uk_system_summary": uk_results["systems"],
        "uk_method_diagnostics": uk_results["diagnostics"],
        "dbp2009_zero_shot": uk_results["dbp_summary"],
        "temporal_2021": temporal,
        "robustness": robustness,
        "acceptance_gates": acceptance,
        "publication_scenario": scenario,
    }
    for name, frame in tables.items():
        save_table(frame, paths, name)

    plot_results(
        paths,
        eligibility,
        mechanism,
        residual_pairs,
        core_results["summary"],
        core_results["regions"],
        core_results["predictions"],
        uk_results["systems"],
    )

    metadata = {
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - run_start,
        "seed": SEED,
        "cuda_device": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": __import__("xgboost").__version__,
        "torch": torch.__version__,
        "protocol_lock": "locks/protocol_lock.json",
        "method_lock_before_uk": "locks/method_lock_before_uk.json",
        "tables": sorted(tables),
        "figures": sorted(path.name for path in paths.figures.glob("*.png")),
        "core_gates_passed": int(
            ((acceptance["is_core"]) & (acceptance["status"] == "Passed")).sum()
        ),
        "core_gates_total": int(acceptance["is_core"].sum()),
        "publication_scenario": scenario.iloc[0]["scenario"],
    }
    (paths.output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
