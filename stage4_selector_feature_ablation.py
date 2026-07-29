from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / ".deps"))

import numpy as np
import pandas as pd
import scipy
import sklearn

from run_new_experiments import (
    FORBIDDEN,
    GROUP,
    REGION,
    SEED,
    TARGET,
    Paths,
    load_data,
    parse_bool,
    sha256_file,
)
from run_optimized_experiments import (
    ACTIONS,
    ACTION_META,
    CANDIDATE_CACHE_VERSION,
    DECISION_TOLERANCE,
    MAX_ADAPTATION_SHIFT,
    MODEL_NAME,
    POLICY_FEATURES,
    POLICY_NAME,
    PRIMARY_RISK_BUDGET,
    RIDGE_ALPHAS,
    cvar90,
    decisions_from_predictions,
    fixed_spec_decision_invariant_to_future_losses,
    ridge_pipeline,
    selected_sample_predictions,
    stable_hash,
    tune_policy,
)


ANALYSIS_STATUS = "posthoc_revision_sensitivity_nonconfirmatory"
REMOVED_SELECTOR_FEATURES = ("calibration_samples", "calibration_sites")
ABLATED_POLICY_FEATURES = tuple(
    feature for feature in POLICY_FEATURES if feature not in REMOVED_SELECTOR_FEATURES
)
FUTURE_OUTCOME_FIELDS = {
    "base_mae",
    "evaluation_rounds",
    "evaluation_samples",
    *[f"actual__{action}" for action in ACTIONS],
}
DEFAULT_DATA_PACKAGE = PROJECT_DIR.parent / "haa6br_data" / "haa6br_integrated_v1"
DEFAULT_LOCKED_ROOT = (
    PROJECT_DIR / "outputs" / "optimized_srcs_strict_v4_20260728"
)
TABLE_PREFIX = "selector_feature_ablation"


def ordered_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_feature_contract() -> None:
    full = tuple(POLICY_FEATURES)
    ablated = tuple(ABLATED_POLICY_FEATURES)
    removed = tuple(feature for feature in full if feature not in ablated)
    if removed != REMOVED_SELECTOR_FEATURES:
        raise AssertionError(
            f"Selector ablation must jointly remove {REMOVED_SELECTOR_FEATURES}; got {removed}"
        )
    if any(feature in ablated for feature in REMOVED_SELECTOR_FEATURES):
        raise AssertionError("Sampling-intensity shortcut features remain in the ablation")
    if set(ablated).intersection(FUTURE_OUTCOME_FIELDS):
        raise AssertionError("Ablated selector contains future-outcome fields")


def ensure_output_is_separate(output_root: Path, locked_root: Path) -> None:
    output = output_root.resolve()
    locked = locked_root.resolve()
    if output == locked or locked in output.parents:
        raise ValueError("Stage 4 output root must be outside the locked strict-v4 root")


def _directory_state(root: Path) -> dict[str, object]:
    records = []
    total_bytes = 0
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        total_bytes += stat.st_size
        records.append(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}")
    return {
        "root": str(root.resolve()),
        "files": len(records),
        "bytes": int(total_bytes),
        "name_size_mtime_sha256": ordered_digest(records),
    }


def _load_checksum_map(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        checksums[relative.strip().replace("\\", "/")] = digest.lower()
    return checksums


def validate_cleaned_package(data_package: Path) -> dict[str, object]:
    data_package = data_package.resolve()
    if data_package.name != "haa6br_integrated_v1":
        raise ValueError(
            "This analysis is locked to the cleaned haa6br_integrated_v1 package"
        )
    report_path = data_package / "metadata" / "validation_report.json"
    sums_path = data_package / "metadata" / "SHA256SUMS.txt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or int(report.get("checks_failed", 1)) != 0:
        raise RuntimeError("Cleaned data package validation report is not PASS")
    expected = _load_checksum_map(sums_path)
    audited = {}
    for relative in (
        "data/us_ucmr4_core.csv",
        "metadata/model_feature_sets.json",
        "metadata/validation_report.json",
    ):
        path = data_package / Path(relative)
        observed = sha256_file(path).lower()
        if expected.get(relative) != observed:
            raise RuntimeError(f"Cleaned package checksum mismatch: {relative}")
        audited[relative] = observed
    return {
        "package": str(data_package),
        "validation_status": report["status"],
        "checks_passed": int(report["checks_passed"]),
        "checks_failed": int(report["checks_failed"]),
        "validation_report_sha256": sha256_file(report_path),
        "sha256sums_sha256": sha256_file(sums_path),
        "audited_files": audited,
    }


class ReadOnlyLockedCandidateStore:
    """Read strict-v4 base predictions and derived candidate tables without writes."""

    def __init__(
        self,
        core: pd.DataFrame,
        operational_features: Sequence[str],
        locked_root: Path,
    ) -> None:
        self.core = core
        self.operational_features = tuple(operational_features)
        self.locked_root = locked_root.resolve()
        self.base_root = (
            self.locked_root
            / "cache"
            / "us_operational"
            / MODEL_NAME.replace(" ", "_")
        )
        self.candidate_root = (
            self.base_root / f"candidate_tables_{CANDIDATE_CACHE_VERSION}"
        )
        self.manifest_path = self.candidate_root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Locked candidate-cache manifest is missing: {self.manifest_path}"
            )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.protocol_path = (
            self.locked_root / "locks" / "protocol_lock_before_optimized_run.json"
        )
        if not self.protocol_path.is_file():
            raise FileNotFoundError(f"Locked strict-v4 protocol is missing: {self.protocol_path}")
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self._validate_manifest()
        self.base_audit: dict[Path, dict[str, object]] = {}
        self.candidate_audit: dict[Path, dict[str, object]] = {}
        self.path_roles: dict[Path, set[str]] = defaultdict(set)

    def _validate_manifest(self) -> None:
        required = {
            "cache_version": CANDIDATE_CACHE_VERSION,
            "model": MODEL_NAME,
            "maximum_prediction_shift_ug_l": MAX_ADAPTATION_SHIFT,
            "policy_features": list(POLICY_FEATURES),
            "actions": ACTION_META,
            "feature_hash": stable_hash(self.operational_features),
        }
        for key, expected in required.items():
            if self.manifest.get(key) != expected:
                raise RuntimeError(
                    f"Locked candidate-cache manifest mismatch for {key}"
                )
        fingerprint_columns = ["sample_id", GROUP, REGION, TARGET, "round_index"]
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(
                self.core[fingerprint_columns], index=False
            ).to_numpy(np.uint64).tobytes()
        ).hexdigest()
        if self.manifest.get("data_hash") != data_hash:
            raise RuntimeError("Locked candidate cache does not match cleaned core data")
        if self.manifest.get("script_sha256") != self.protocol.get("script_sha256"):
            raise RuntimeError(
                "Locked candidate cache producer hash differs from the strict-v4 protocol lock"
            )
        if float(self.protocol.get("primary_risk_budget", -1.0)) != PRIMARY_RISK_BUDGET:
            raise RuntimeError("Locked protocol risk budget differs from this ablation")
        if tuple(float(value) for value in self.protocol.get("ridge_alphas", [])) != tuple(
            float(value) for value in RIDGE_ALPHAS
        ):
            raise RuntimeError("Locked protocol ridge-alpha grid differs from this ablation")
        self.current_runner_sha256 = sha256_file(
            PROJECT_DIR / "run_optimized_experiments.py"
        )
        self.current_runner_matches_locked_producer = bool(
            self.current_runner_sha256 == self.manifest.get("script_sha256")
        )

    @staticmethod
    def _token(excluded: Iterable[int]) -> tuple[tuple[int, ...], str]:
        normalized = tuple(sorted(int(value) for value in excluded))
        if not normalized:
            raise ValueError("At least one excluded EPA region is required")
        return normalized, "-".join(map(str, normalized))

    def _base_path(self, excluded: Iterable[int], predicted_region: int) -> Path:
        normalized, token = self._token(excluded)
        if int(predicted_region) not in normalized:
            raise AssertionError("Predicted region must be excluded from base training")
        return self.base_root / f"exclude_{token}__predict_{int(predicted_region)}.npy"

    def _candidate_paths(
        self, excluded: Iterable[int], predicted_region: int, k: int
    ) -> tuple[Path, Path]:
        normalized, token = self._token(excluded)
        if int(predicted_region) not in normalized:
            raise AssertionError("Predicted region must be excluded from base training")
        stem = f"exclude_{token}__predict_{int(predicted_region)}__k{int(k)}"
        return (
            self.candidate_root / f"{stem}__systems.pkl",
            self.candidate_root / f"{stem}__samples.pkl",
        )

    def _verify_base(
        self, excluded: tuple[int, ...], predicted_region: int, role: str
    ) -> None:
        path = self._base_path(excluded, predicted_region)
        if not path.is_file():
            raise FileNotFoundError(
                "Required locked strict-v4 base-prediction cache is missing; "
                f"refitting is forbidden: {path}"
            )
        self.path_roles[path].add(role)
        if path in self.base_audit:
            return
        values = np.load(path, allow_pickle=False)
        expected_rows = int((self.core[REGION] == int(predicted_region)).sum())
        if len(values) != expected_rows or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid locked base-prediction cache: {path}")
        self.base_audit[path] = {
            "path": str(path),
            "relative_path": path.relative_to(self.locked_root).as_posix(),
            "excluded_regions": "|".join(map(str, excluded)),
            "predicted_region": int(predicted_region),
            "rows": int(len(values)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "read_only_input": True,
        }

    def _record_candidate(self, path: Path, role: str, rows: int) -> None:
        self.path_roles[path].add(role)
        if path in self.candidate_audit:
            return
        self.candidate_audit[path] = {
            "path": str(path),
            "relative_path": path.relative_to(self.locked_root).as_posix(),
            "rows": int(rows),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "read_only_input": True,
        }

    def get(
        self,
        excluded_regions: Iterable[int],
        predicted_region: int,
        k: int,
        with_samples: bool,
        role: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        excluded, _ = self._token(excluded_regions)
        if int(k) not in (1, 2, 3):
            raise ValueError(f"Unsupported monitoring depth: {k}")
        self._verify_base(excluded, int(predicted_region), role)
        system_path, sample_path = self._candidate_paths(
            excluded, int(predicted_region), int(k)
        )
        required_paths = [system_path, *([sample_path] if with_samples else [])]
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required locked strict-v4 candidate cache is missing; rebuilding "
                f"inside the protected root is forbidden: {missing}"
            )
        systems = pd.read_pickle(system_path)
        samples = pd.read_pickle(sample_path) if with_samples else pd.DataFrame()
        self._record_candidate(system_path, role, len(systems))
        if with_samples:
            self._record_candidate(sample_path, role, len(samples))
        if systems.empty:
            return systems, samples
        required_columns = {
            GROUP,
            REGION,
            "k",
            "base_mae",
            "evaluation_rounds",
            "evaluation_samples",
            *POLICY_FEATURES,
            *[f"actual__{action}" for action in ACTIONS],
        }
        missing_columns = sorted(required_columns.difference(systems.columns))
        if missing_columns:
            raise RuntimeError(
                f"Locked candidate table is incomplete: {missing_columns[:5]}"
            )
        if systems[GROUP].duplicated().any():
            raise RuntimeError(f"Duplicate systems in locked candidate table: {system_path}")
        if not systems[REGION].eq(int(predicted_region)).all():
            raise RuntimeError(f"Candidate table mixes EPA regions: {system_path}")
        if not systems["k"].eq(int(k)).all():
            raise RuntimeError(f"Candidate table mixes monitoring depths: {system_path}")
        if with_samples and not samples.empty:
            if not set(samples[GROUP]).issubset(set(systems[GROUP])):
                raise RuntimeError("Candidate sample systems are absent from system table")
            if not samples[REGION].eq(int(predicted_region)).all():
                raise RuntimeError(f"Candidate sample table mixes regions: {sample_path}")
            if not samples["k"].eq(int(k)).all():
                raise RuntimeError(f"Candidate sample table mixes depths: {sample_path}")
        return systems, samples

    def base_audit_frame(self) -> pd.DataFrame:
        rows = []
        for path, row in self.base_audit.items():
            rows.append({**row, "access_roles": "|".join(sorted(self.path_roles[path]))})
        return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)

    def candidate_audit_frame(self) -> pd.DataFrame:
        rows = []
        for path, row in self.candidate_audit.items():
            rows.append({**row, "access_roles": "|".join(sorted(self.path_roles[path]))})
        return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)


def fit_selector_models(
    train: pd.DataFrame,
    alpha: float,
    feature_names: Sequence[str] = ABLATED_POLICY_FEATURES,
) -> dict[str, object]:
    if train.empty:
        raise RuntimeError("Policy training table is empty")
    missing = sorted(set(feature_names).difference(train.columns))
    if missing:
        raise ValueError(f"Selector training features are missing: {missing}")
    models: dict[str, object] = {}
    for action in ACTIONS:
        model = ridge_pipeline(float(alpha))
        model.fit(train[list(feature_names)], train[f"actual__{action}"])
        models[action] = model
    return models


def predict_selector(
    models: dict[str, object],
    frame: pd.DataFrame,
    feature_names: Sequence[str] = ABLATED_POLICY_FEATURES,
) -> pd.DataFrame:
    columns = [
        GROUP,
        REGION,
        "k",
        "base_mae",
        "evaluation_rounds",
        "evaluation_samples",
    ]
    output = frame[columns].copy()
    for action in ACTIONS:
        output[f"pred__{action}"] = models[action].predict(frame[list(feature_names)])
        output[f"actual__{action}"] = frame[f"actual__{action}"].to_numpy(float)
    return output


def build_outer_ablation(
    core: pd.DataFrame,
    store: ReadOnlyLockedCandidateStore,
    outer_region: int,
    k: int,
) -> dict[str, pd.DataFrame]:
    all_regions = tuple(sorted(int(value) for value in core[REGION].unique()))
    if int(outer_region) not in all_regions:
        raise ValueError(f"Unknown held EPA region: {outer_region}")
    source_regions = tuple(value for value in all_regions if value != int(outer_region))
    oof_frames: dict[float, list[pd.DataFrame]] = {
        float(alpha): [] for alpha in RIDGE_ALPHAS
    }
    final_source_frames = []
    policy_audit_rows = []

    for held_region in source_regions:
        validation, _ = store.get(
            (outer_region, held_region),
            held_region,
            k,
            False,
            "source_policy_validation",
        )
        if validation.empty:
            continue
        final_source_frames.append(validation)
        training_frames = []
        for pseudo_region in source_regions:
            if pseudo_region == held_region:
                continue
            systems, _ = store.get(
                (outer_region, held_region, pseudo_region),
                pseudo_region,
                k,
                False,
                "source_policy_training",
            )
            if not systems.empty:
                training_frames.append(systems)
        if not training_frames:
            raise RuntimeError(
                f"No source training systems for outer={outer_region}, held={held_region}, k={k}"
            )
        training = pd.concat(training_frames, ignore_index=True)
        training_regions = set(int(value) for value in training[REGION].unique())
        overlap = set(training[GROUP]).intersection(validation[GROUP])
        if overlap:
            raise AssertionError(
                f"Policy system leakage outer={outer_region}, held={held_region}: {len(overlap)}"
            )
        if int(outer_region) in training_regions or int(held_region) in training_regions:
            raise AssertionError("Outer or policy-held region leaked into selector training")
        for alpha in RIDGE_ALPHAS:
            models = fit_selector_models(training, alpha)
            predicted = predict_selector(models, validation)
            predicted["policy_held_region"] = int(held_region)
            predicted["outer_target_region"] = int(outer_region)
            oof_frames[float(alpha)].append(predicted)
        policy_audit_rows.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "outer_target_region": int(outer_region),
                "k": int(k),
                "policy_held_region": int(held_region),
                "training_regions": "|".join(map(str, sorted(training_regions))),
                "validation_region": int(held_region),
                "training_systems": int(training[GROUP].nunique()),
                "validation_systems": int(validation[GROUP].nunique()),
                "system_overlap": int(len(overlap)),
                "outer_region_absent_from_selector_training": True,
                "policy_held_region_absent_from_selector_training": True,
                "base_models_exclude_outer_policyheld_and_pseudo_regions": True,
                "target_outcomes_used_for_variant_selection": False,
            }
        )

    oof_by_alpha = {
        alpha: pd.concat(frames, ignore_index=True)
        for alpha, frames in oof_frames.items()
        if frames
    }
    if set(oof_by_alpha) != {float(alpha) for alpha in RIDGE_ALPHAS}:
        raise RuntimeError("Incomplete source-only selector OOF predictions")
    final_source = pd.concat(final_source_frames, ignore_index=True)

    # This is the policy lock: alpha and margin are selected before target candidates load.
    spec, source_decisions, policy_search = tune_policy(
        oof_by_alpha,
        PRIMARY_RISK_BUDGET,
        "all",
    )
    spec_lock_sha256 = json_digest(spec.__dict__)
    policy_search = policy_search.copy()
    policy_search["outer_target_region"] = int(outer_region)
    policy_search["k"] = int(k)
    policy_search["selector_variant"] = "counts_sites_removed"
    policy_search["tuning_data_role"] = "source_regions_only"

    target_systems, target_samples = store.get(
        (outer_region,),
        outer_region,
        k,
        True,
        "held_region_assessment_after_policy_lock",
    )
    final_models = fit_selector_models(final_source, spec.alpha)
    target_policy_predictions = predict_selector(final_models, target_systems)
    target_decisions = decisions_from_predictions(
        target_policy_predictions,
        spec.margin,
        spec.action_set,
    )
    source_invariance = fixed_spec_decision_invariant_to_future_losses(
        oof_by_alpha[spec.alpha],
        spec.margin,
        spec.action_set,
        SEED + 100 * int(outer_region) + int(k),
    )
    target_invariance = fixed_spec_decision_invariant_to_future_losses(
        target_policy_predictions,
        spec.margin,
        spec.action_set,
        SEED + 1000 + 100 * int(outer_region) + int(k),
    )
    if not source_invariance or not target_invariance:
        raise AssertionError("Fixed selector decisions changed after future-label perturbation")

    selected_samples = selected_sample_predictions(target_samples, target_decisions)
    selected_samples["outer_target_region"] = int(outer_region)
    selected_samples["selector_variant"] = "counts_sites_removed"
    selected_samples["analysis_status"] = ANALYSIS_STATUS
    selected_samples["evidence_boundary"] = (
        "Prediction negative transfer and CVaR are algorithmic prediction outcomes, "
        "not health-safety outcomes"
    )

    target_decisions = target_decisions.merge(
        target_systems[
            [GROUP, "calibration_samples", "calibration_sites"]
        ],
        on=GROUP,
        how="left",
        validate="one_to_one",
    )
    target_decisions["outer_target_region"] = int(outer_region)
    target_decisions["selector_variant"] = "counts_sites_removed"
    target_decisions["policy_locked_before_target_outcome_assessment"] = True
    target_decisions["target_outcomes_used_for_variant_selection"] = False

    source_decisions = source_decisions.copy()
    source_decisions["outer_target_region"] = int(outer_region)
    source_decisions["selector_variant"] = "counts_sites_removed"
    source_decisions["evidence_role"] = "source-only cross-fitted policy tuning"

    spec_row = {
        "analysis_status": ANALYSIS_STATUS,
        "outer_target_region": int(outer_region),
        "k": int(k),
        "selector_variant": "counts_sites_removed",
        **spec.__dict__,
        "policy_spec_lock_sha256": spec_lock_sha256,
        "policy_locked_before_target_candidate_load": True,
        "full_feature_count": len(POLICY_FEATURES),
        "ablated_feature_count": len(ABLATED_POLICY_FEATURES),
        "removed_features": "|".join(REMOVED_SELECTOR_FEATURES),
        "source_fixed_spec_future_loss_invariance": source_invariance,
        "target_fixed_spec_future_loss_invariance": target_invariance,
        "held_region_outcomes_used_to_choose_variant": False,
        "source_metrics_role": "Policy tuning constraints; not independent validation evidence",
    }
    invariant_row = {
        "analysis_status": ANALYSIS_STATUS,
        "outer_target_region": int(outer_region),
        "k": int(k),
        "joint_removal_verified": set(REMOVED_SELECTOR_FEATURES).isdisjoint(
            ABLATED_POLICY_FEATURES
        ),
        "only_prespecified_features_removed": set(POLICY_FEATURES).difference(
            ABLATED_POLICY_FEATURES
        )
        == set(REMOVED_SELECTOR_FEATURES),
        "ablated_features_exclude_future_outcomes": set(
            ABLATED_POLICY_FEATURES
        ).isdisjoint(FUTURE_OUTCOME_FIELDS),
        "source_fixed_spec_future_loss_invariance": source_invariance,
        "target_fixed_spec_future_loss_invariance": target_invariance,
        "policy_locked_before_target_candidate_load": True,
        "held_region_outcomes_used_to_choose_variant": False,
        "health_safety_interpretation_permitted": False,
    }
    return {
        "predictions": selected_samples,
        "target_decisions": target_decisions,
        "source_decisions": source_decisions,
        "policy_spec": pd.DataFrame([spec_row]),
        "policy_search": policy_search,
        "policy_audit": pd.DataFrame(policy_audit_rows),
        "invariant_audit": pd.DataFrame([invariant_row]),
    }


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("Unsupported boolean values in adaptation field")
    return normalized.isin({"true", "1"})


def match_locked_full_predictions(
    ablated_predictions: pd.DataFrame,
    locked_prediction_path: Path,
    regions: Sequence[int],
    k_values: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "sample_id",
        GROUP,
        "group_site_id",
        REGION,
        "sample_date",
        "round_index",
        "k",
        "observed",
        "Zero-shot",
        POLICY_NAME,
        "selected_action",
        "adapted",
    ]
    full = pd.read_csv(locked_prediction_path, usecols=columns, low_memory=False)
    full = full.loc[
        full[REGION].isin([int(value) for value in regions])
        & full["k"].isin([int(value) for value in k_values])
    ].copy()
    keys = ["sample_id", GROUP, "group_site_id", REGION, "round_index", "k"]
    ablated_columns = [
        *keys,
        "observed",
        "Zero-shot",
        POLICY_NAME,
        "selected_action",
        "adapted",
    ]
    ablated = ablated_predictions[ablated_columns].copy()
    if full.duplicated(keys).any() or ablated.duplicated(keys).any():
        raise AssertionError("Prediction match keys are not unique")
    matched = full.merge(
        ablated,
        on=keys,
        how="outer",
        suffixes=("__locked_full", "__counts_sites_removed"),
        indicator=True,
        validate="one_to_one",
    )
    audit_rows = []
    for (region, k), subset in matched.groupby([REGION, "k"], sort=True):
        both = subset["_merge"].eq("both")
        observed_diff = np.abs(
            subset.loc[both, "observed__locked_full"].to_numpy(float)
            - subset.loc[both, "observed__counts_sites_removed"].to_numpy(float)
        )
        zero_diff = np.abs(
            subset.loc[both, "Zero-shot__locked_full"].to_numpy(float)
            - subset.loc[both, "Zero-shot__counts_sites_removed"].to_numpy(float)
        )
        audit_rows.append(
            {
                "analysis_status": ANALYSIS_STATUS,
                "outer_target_region": int(region),
                "k": int(k),
                "locked_full_rows": int(
                    subset["_merge"].isin(["both", "left_only"]).sum()
                ),
                "ablated_rows": int(
                    subset["_merge"].isin(["both", "right_only"]).sum()
                ),
                "matched_rows": int(both.sum()),
                "left_only_rows": int(subset["_merge"].eq("left_only").sum()),
                "right_only_rows": int(subset["_merge"].eq("right_only").sum()),
                "max_observed_difference": float(observed_diff.max(initial=0.0)),
                "max_zero_shot_difference": float(zero_diff.max(initial=0.0)),
                "equal_system_equal_future_round_cohort": bool(
                    both.all()
                    and observed_diff.max(initial=0.0) <= 1e-12
                    and zero_diff.max(initial=0.0) <= 1e-12
                ),
                "held_region_outcomes_used_to_choose_variant": False,
            }
        )
    audit = pd.DataFrame(audit_rows)
    if matched["_merge"].ne("both").any():
        raise AssertionError("Locked full and ablated prediction cohorts differ")
    for column in ("observed", "Zero-shot"):
        if not np.allclose(
            matched[f"{column}__locked_full"],
            matched[f"{column}__counts_sites_removed"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"Matched predictions differ in {column}")
    return matched.drop(columns="_merge"), audit


def system_outcomes_from_matched(matched: pd.DataFrame) -> pd.DataFrame:
    id_columns = [REGION, GROUP, "round_index", "k"]
    system_frames = []
    for variant in ("locked_full", "counts_sites_removed"):
        work = matched[id_columns].copy()
        observed = matched[f"observed__{variant}"].to_numpy(float)
        selector = matched[f"{POLICY_NAME}__{variant}"].to_numpy(float)
        zero = matched[f"Zero-shot__{variant}"].to_numpy(float)
        work["selector_absolute_error"] = np.abs(observed - selector)
        work["zero_shot_absolute_error"] = np.abs(observed - zero)
        round_errors = work.groupby(id_columns, as_index=False)[
            ["selector_absolute_error", "zero_shot_absolute_error"]
        ].mean()
        system = round_errors.groupby(
            [REGION, GROUP, "k"], as_index=False
        )[["selector_absolute_error", "zero_shot_absolute_error"]].mean()
        action = pd.DataFrame(
            {
                REGION: matched[REGION],
                GROUP: matched[GROUP],
                "k": matched["k"],
                "adapted": _boolean_series(matched[f"adapted__{variant}"]),
            }
        )
        if action.groupby([REGION, GROUP, "k"])["adapted"].nunique().max() > 1:
            raise AssertionError("Adaptation decision changes within a system/depth")
        action = action.groupby([REGION, GROUP, "k"], as_index=False)[
            "adapted"
        ].first()
        system = system.merge(
            action,
            on=[REGION, GROUP, "k"],
            how="left",
            validate="one_to_one",
        )
        system["regret_vs_zero_shot"] = (
            system["selector_absolute_error"] - system["zero_shot_absolute_error"]
        )
        system["selector_variant"] = variant
        system["analysis_status"] = ANALYSIS_STATUS
        system_frames.append(system)
    return pd.concat(system_frames, ignore_index=True)


def _summary_row(
    subset: pd.DataFrame,
    k: int,
    variant: str,
    scope: str,
    region: int | str,
) -> dict[str, object]:
    regrets = subset["regret_vs_zero_shot"].to_numpy(float)
    return {
        "analysis_status": ANALYSIS_STATUS,
        "scope": scope,
        "outer_target_region": region,
        "k": int(k),
        "selector_variant": variant,
        "systems": int(len(subset)),
        "regions": int(subset[REGION].nunique()),
        "equal_system_equal_future_round_mae": float(
            subset["selector_absolute_error"].mean()
        ),
        "zero_shot_equal_system_equal_future_round_mae": float(
            subset["zero_shot_absolute_error"].mean()
        ),
        "mean_regret_vs_zero_shot": float(np.mean(regrets)),
        "prediction_negative_transfer_rate_gt_1e-12": float(
            np.mean(regrets > DECISION_TOLERANCE)
        ),
        "strict_cvar90_regret": cvar90(regrets),
        "p95_regret": float(np.quantile(regrets, 0.95)),
        "maximum_regret": float(np.max(regrets)),
        "adaptation_rate": float(_boolean_series(subset["adapted"]).mean()),
        "prediction_metrics_are_health_safety_outcomes": False,
        "held_region_outcomes_used_to_choose_variant": False,
    }


def summarize_selector_variants(
    system_outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (region, k, variant), subset in system_outcomes.groupby(
        [REGION, "k", "selector_variant"], sort=True
    ):
        rows.append(_summary_row(subset, int(k), str(variant), "held_region", int(region)))
    for (k, variant), subset in system_outcomes.groupby(
        ["k", "selector_variant"], sort=True
    ):
        rows.append(_summary_row(subset, int(k), str(variant), "pooled_regions", "All"))
    summary = pd.DataFrame(rows)
    metrics = [
        "equal_system_equal_future_round_mae",
        "prediction_negative_transfer_rate_gt_1e-12",
        "strict_cvar90_regret",
        "p95_regret",
        "maximum_regret",
        "adaptation_rate",
    ]
    index = ["scope", "outer_target_region", "k"]
    full = summary.loc[summary["selector_variant"].eq("locked_full")].set_index(index)
    ablated = summary.loc[
        summary["selector_variant"].eq("counts_sites_removed")
    ].set_index(index)
    if not full.index.equals(ablated.index):
        full, ablated = full.align(ablated, join="outer", axis=0)
    contrast = ablated[metrics].subtract(full[metrics]).reset_index()
    contrast.rename(
        columns={metric: f"ablated_minus_locked_full__{metric}" for metric in metrics},
        inplace=True,
    )
    contrast["analysis_status"] = ANALYSIS_STATUS
    contrast["contrast_role"] = "descriptive post-hoc sensitivity; no variant selection"
    contrast["held_region_outcomes_used_to_choose_variant"] = False
    return summary, contrast


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked Stage 4 selector sampling-intensity feature ablation"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-package", type=Path, default=DEFAULT_DATA_PACKAGE)
    parser.add_argument("--locked-root", type=Path, default=DEFAULT_LOCKED_ROOT)
    parser.add_argument("--held-region", type=int, action="append")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    output_root = args.output_root.resolve()
    locked_root = args.locked_root.resolve()
    data_package = args.data_package.resolve()
    ensure_output_is_separate(output_root, locked_root)
    validate_feature_contract()
    package_audit = validate_cleaned_package(data_package)
    if not locked_root.is_dir():
        raise FileNotFoundError(f"Locked strict-v4 root is missing: {locked_root}")
    locked_metadata_path = locked_root / "run_metadata.json"
    locked_metadata = json.loads(locked_metadata_path.read_text(encoding="utf-8"))
    if locked_metadata.get("status") != "PASS_EXECUTION_AND_AUDIT":
        raise RuntimeError("Locked strict-v4 run metadata is not PASS_EXECUTION_AND_AUDIT")
    before_state = _directory_state(locked_root)

    paths = Paths(
        data_package,
        output_root,
        output_root / "tables",
        output_root / "figures",
        output_root / "locks",
    )
    core, _, _, feature_sets = load_data(paths)
    operational_features = list(feature_sets["us_operational_core"])
    if not set(operational_features).isdisjoint(FORBIDDEN):
        raise AssertionError("Operational source-model features contain forbidden fields")
    all_regions = tuple(sorted(int(value) for value in core[REGION].unique()))
    regions = (
        tuple(dict.fromkeys(int(value) for value in args.held_region))
        if args.held_region
        else all_regions
    )
    if not set(regions).issubset(all_regions):
        raise ValueError(f"Held regions must be drawn from {all_regions}")
    k_values = tuple(sorted(set(int(value) for value in args.k_values)))
    if not k_values or not set(k_values).issubset({1, 2, 3}):
        raise ValueError("Monitoring depths must be selected from k=1,2,3")

    store = ReadOnlyLockedCandidateStore(core, operational_features, locked_root)
    collected: dict[str, list[pd.DataFrame]] = defaultdict(list)
    total = len(regions) * len(k_values)
    step = 0
    for outer_region in regions:
        for k in k_values:
            step += 1
            print(
                f"[selector ablation {step}/{total}] outer EPA region {outer_region}, k={k}",
                flush=True,
            )
            result = build_outer_ablation(core, store, outer_region, k)
            for name, frame in result.items():
                collected[name].append(frame)

    tables = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in collected.items()
    }
    prediction_columns = [
        column
        for column in (
            "sample_id",
            GROUP,
            "group_site_id",
            REGION,
            "sample_date",
            "round_index",
            "k",
            "observed",
            "Zero-shot",
            POLICY_NAME,
            "selected_action",
            "selected_predicted_delta",
            "adapted",
            "outer_target_region",
            "selector_variant",
            "analysis_status",
            "evidence_boundary",
        )
        if column in tables["predictions"].columns
    ]
    tables["predictions"] = tables["predictions"][prediction_columns]

    locked_predictions_path = locked_root / "tables" / "us_predictions.csv"
    matched, match_audit = match_locked_full_predictions(
        tables["predictions"], locked_predictions_path, regions, k_values
    )
    system_outcomes = system_outcomes_from_matched(matched)
    summary, contrast = summarize_selector_variants(system_outcomes)
    locked_specs = pd.read_csv(locked_root / "tables" / "policy_spec.csv")
    locked_specs = locked_specs.loc[
        locked_specs["outer_target_region"].isin(regions)
        & locked_specs["k"].isin(k_values),
        ["outer_target_region", "k", "alpha", "margin", "feasible"],
    ].rename(
        columns={
            "alpha": "locked_full_alpha",
            "margin": "locked_full_margin",
            "feasible": "locked_full_source_feasible",
        }
    )
    tables["policy_spec"] = tables["policy_spec"].merge(
        locked_specs,
        on=["outer_target_region", "k"],
        how="left",
        validate="one_to_one",
    )
    if tables["policy_spec"]["locked_full_alpha"].isna().any():
        raise RuntimeError("Locked full-selector policy specification is incomplete")

    tables.update(
        {
            "matched_predictions": matched,
            "match_audit": match_audit,
            "system_outcomes": system_outcomes,
            "comparison": summary,
            "contrasts": contrast,
            "base_cache_audit": store.base_audit_frame(),
            "candidate_cache_audit": store.candidate_audit_frame(),
        }
    )
    tables_dir = output_root / "tables"
    output_paths = {}
    for name, frame in tables.items():
        path = tables_dir / f"{TABLE_PREFIX}_{name}.csv"
        _save_csv(frame, path)
        output_paths[name] = path

    after_state = _directory_state(locked_root)
    if before_state != after_state:
        raise RuntimeError("Locked strict-v4 directory state changed during read-only ablation")
    runtime = time.time() - started
    run_manifest = {
        "analysis_status": ANALYSIS_STATUS,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime,
        "scientific_role": (
            "Post-hoc non-confirmatory selector shortcut sensitivity; held-region outcomes "
            "are assessment-only and cannot select between variants"
        ),
        "health_safety_boundary": (
            "Prediction negative transfer, CVaR90, p95, and maximum regret are prediction "
            "outcomes, not health, treatment, exposure, or regulatory safety outcomes"
        ),
        "seed": SEED,
        "invariance_seeds": {
            "source": "SEED + 100 * outer_region + k",
            "target": "SEED + 1000 + 100 * outer_region + k",
        },
        "risk_budget": PRIMARY_RISK_BUDGET,
        "ridge_alphas": list(RIDGE_ALPHAS),
        "outer_regions": list(regions),
        "k_values": list(k_values),
        "full_selector_features": list(POLICY_FEATURES),
        "ablated_selector_features": list(ABLATED_POLICY_FEATURES),
        "jointly_removed_selector_features": list(REMOVED_SELECTOR_FEATURES),
        "source_only_retuning": True,
        "held_region_outcomes_used_to_choose_variant": False,
        "equal_system_equal_future_round_comparison": True,
        "data_package_audit": package_audit,
        "locked_inputs": {
            "root_state_before_and_after": before_state,
            "run_metadata_sha256": sha256_file(locked_metadata_path),
            "protocol_lock_sha256": sha256_file(
                locked_root / "locks" / "protocol_lock_before_optimized_run.json"
            ),
            "locked_full_predictions_sha256": sha256_file(locked_predictions_path),
            "locked_policy_spec_sha256": sha256_file(
                locked_root / "tables" / "policy_spec.csv"
            ),
            "candidate_manifest_sha256": sha256_file(store.manifest_path),
            "locked_cache_producer_script_sha256": store.manifest[
                "script_sha256"
            ],
            "current_runner_sha256": store.current_runner_sha256,
            "current_runner_matches_locked_cache_producer": (
                store.current_runner_matches_locked_producer
            ),
            "provenance_note": (
                "The candidate manifest producer hash is required to match the strict-v4 "
                "protocol lock. The current runner hash is recorded separately because the "
                "working copy may have changed after the locked run."
            ),
            "accessed_base_cache_inventory_sha256": ordered_digest(
                f"{row['relative_path']}|{row['sha256']}"
                for row in store.base_audit_frame().to_dict("records")
            ),
            "accessed_candidate_cache_inventory_sha256": ordered_digest(
                f"{row['relative_path']}|{row['sha256']}"
                for row in store.candidate_audit_frame().to_dict("records")
            ),
        },
        "code_sha256": {
            "stage4_selector_feature_ablation.py": sha256_file(Path(__file__)),
            "run_optimized_experiments.py": sha256_file(
                PROJECT_DIR / "run_optimized_experiments.py"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": {
            name: {
                "path": str(path),
                "rows": int(len(tables[name])),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
    }
    manifest_path = output_root / f"{TABLE_PREFIX}_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Completed {total} held-region/depth cells in {runtime:.1f} s; "
        f"outputs: {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
