from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from run_new_experiments import GROUP, REGION  # noqa: E402
from run_optimized_experiments import (  # noqa: E402
    ACTIONS,
    POLICY_FEATURES,
)
from stage4_selector_feature_ablation import (  # noqa: E402
    ABLATED_POLICY_FEATURES,
    REMOVED_SELECTOR_FEATURES,
    ensure_output_is_separate,
    fit_selector_models,
    match_locked_full_predictions,
    predict_selector,
    summarize_selector_variants,
    system_outcomes_from_matched,
    validate_feature_contract,
)


def synthetic_candidate_table(rows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(20260730)
    frame = pd.DataFrame(
        {
            GROUP: [f"S{i}" for i in range(rows)],
            REGION: np.repeat([1, 2, 3], rows // 3),
            "k": 1,
            "base_mae": rng.uniform(1.0, 4.0, rows),
            "evaluation_rounds": 2,
            "evaluation_samples": 3,
        }
    )
    for index, feature in enumerate(POLICY_FEATURES):
        frame[feature] = rng.normal(index, 1.0, rows)
    for index, action in enumerate(ACTIONS):
        frame[f"actual__{action}"] = (
            0.05 * frame["mean_residual"] + rng.normal(index / 100.0, 0.1, rows)
        )
    return frame


def synthetic_matched_predictions() -> pd.DataFrame:
    rows = []
    for system_index, system in enumerate(("A", "B")):
        for round_index, samples in ((2, 1), (3, 3)):
            for sample_index in range(samples):
                observed = 10.0 + system_index + round_index + sample_index
                rows.append(
                    {
                        "sample_id": f"{system}-{round_index}-{sample_index}",
                        GROUP: system,
                        "group_site_id": f"{system}-site",
                        REGION: 1,
                        "round_index": round_index,
                        "k": 1,
                        "observed__locked_full": observed,
                        "observed__counts_sites_removed": observed,
                        "Zero-shot__locked_full": observed - 2.0,
                        "Zero-shot__counts_sites_removed": observed - 2.0,
                        "SRCS__locked_full": observed - 1.0,
                        "SRCS__counts_sites_removed": observed - 0.5,
                        "selected_action__locked_full": "HistoryMean_050",
                        "selected_action__counts_sites_removed": "HistoryMean_050",
                        "adapted__locked_full": True,
                        "adapted__counts_sites_removed": True,
                    }
                )
    return pd.DataFrame(rows)


def test_feature_contract_removes_counts_and_sites_jointly() -> None:
    validate_feature_contract()
    assert set(POLICY_FEATURES).difference(ABLATED_POLICY_FEATURES) == set(
        REMOVED_SELECTOR_FEATURES
    )
    assert set(REMOVED_SELECTOR_FEATURES).isdisjoint(ABLATED_POLICY_FEATURES)


def test_predictions_ignore_jointly_removed_features() -> None:
    frame = synthetic_candidate_table()
    models = fit_selector_models(frame, alpha=10.0)
    reference = predict_selector(models, frame)
    perturbed = frame.copy()
    perturbed["calibration_samples"] = np.arange(len(frame)) + 10000
    perturbed["calibration_sites"] = np.arange(len(frame)) + 5000
    repeated = predict_selector(models, perturbed)
    prediction_columns = [f"pred__{action}" for action in ACTIONS]
    assert np.allclose(
        reference[prediction_columns],
        repeated[prediction_columns],
        rtol=0.0,
        atol=0.0,
    )


def test_system_metrics_use_equal_round_then_equal_system_weights() -> None:
    outcomes = system_outcomes_from_matched(synthetic_matched_predictions())
    full = outcomes.loc[outcomes["selector_variant"].eq("locked_full")]
    ablated = outcomes.loc[
        outcomes["selector_variant"].eq("counts_sites_removed")
    ]
    assert np.allclose(full["selector_absolute_error"], 1.0)
    assert np.allclose(ablated["selector_absolute_error"], 0.5)
    assert np.allclose(full["regret_vs_zero_shot"], -1.0)
    assert np.allclose(ablated["regret_vs_zero_shot"], -1.5)

    summary, contrast = summarize_selector_variants(outcomes)
    pooled = summary.loc[summary["scope"].eq("pooled_regions")]
    assert len(pooled) == 2
    assert np.isclose(
        contrast.loc[
            contrast["scope"].eq("pooled_regions"),
            "ablated_minus_locked_full__equal_system_equal_future_round_mae",
        ].iloc[0],
        -0.5,
    )
    assert not contrast["held_region_outcomes_used_to_choose_variant"].any()


def test_output_root_cannot_be_inside_locked_root(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    with pytest.raises(ValueError, match="outside"):
        ensure_output_is_separate(locked / "new", locked)
    ensure_output_is_separate(tmp_path / "stage4", locked)


def test_match_rejects_unequal_cohorts(tmp_path: Path) -> None:
    locked_path = tmp_path / "us_predictions.csv"
    full = pd.DataFrame(
        {
            "sample_id": ["x"],
            GROUP: ["S1"],
            "group_site_id": ["site"],
            REGION: [1],
            "sample_date": ["2020-01-01"],
            "round_index": [2],
            "k": [1],
            "observed": [2.0],
            "Zero-shot": [1.0],
            "SRCS": [1.5],
            "selected_action": ["Zero-shot"],
            "adapted": [False],
        }
    )
    full.to_csv(locked_path, index=False)
    ablated = full.iloc[0:0].copy()
    with pytest.raises(AssertionError, match="cohorts differ"):
        match_locked_full_predictions(ablated, locked_path, [1], [1])
