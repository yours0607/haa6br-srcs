from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from stage4_mechanism_attribution import (  # noqa: E402
    FIXED_THRESHOLD_ACTION,
    fixed_threshold_decisions,
    fixed_threshold_target_invariance,
    small_cluster_mechanism_sensitivity,
    source_utility_spec,
    tune_fixed_threshold,
    uncapped_same_selector,
)


def test_source_utility_spec_uses_source_mean_delta_without_feasibility() -> None:
    search = pd.DataFrame(
        {
            "action_set": ["all", "all", "history_only"],
            "alpha": [1.0, 10.0, 1.0],
            "margin": [0.5, 0.0, 0.0],
            "mean_delta": [-1.0, -2.0, -9.0],
            "cvar90": [1.0, 3.0, 0.1],
            "negative_transfer": [0.1, 0.3, 0.01],
            "adaptation_rate": [0.7, 1.0, 1.0],
        }
    )
    selected = source_utility_spec(search)
    assert selected["alpha"] == 10.0
    assert selected["margin"] == 0.0
    assert selected["source_mean_delta"] == -2.0


def test_uncapped_variant_preserves_fallback_and_removes_shift_cap() -> None:
    selected = pd.DataFrame(
        {
            "group_system_id": ["A", "B", "C"],
            "Zero-shot": [10.0, 10.0, 10.0],
            "selected_action": ["HistoryMean_100", "RawMean_100", "Zero-shot"],
            "Baseline__Persistence": [10.0, 10.0, 10.0],
            "Baseline__HistoryMean": [40.0, 10.0, 10.0],
            "Baseline__HistoryMedian": [10.0, 10.0, 10.0],
            "Baseline__RawMean": [10.0, 30.0, 10.0],
            "Baseline__RawMedian": [10.0, 10.0, 10.0],
            "SRCS": [22.0, 22.0, 10.0],
        }
    )
    target_systems = pd.DataFrame(
        {
            "group_system_id": ["A", "B", "C"],
            "mean_residual": [0.0, 20.0, 0.0],
            "median_residual": [0.0, 0.0, 0.0],
        }
    )
    output = uncapped_same_selector(selected, target_systems)
    assert np.allclose(output["SRCS"], [40.0, 30.0, 10.0])


def test_fixed_threshold_gate_uses_calibration_residual_not_future_loss() -> None:
    systems = pd.DataFrame(
        {
            "group_system_id": ["A", "B"],
            "epa_region": [1, 1],
            "k": [1, 1],
            "base_mae": [2.0, 2.0],
            "evaluation_rounds": [2, 2],
            "evaluation_samples": [4, 4],
            "abs_mean_residual": [1.0, 3.0],
            f"actual__{FIXED_THRESHOLD_ACTION}": [100.0, -1.0],
        }
    )
    decisions = fixed_threshold_decisions(systems, threshold=2.0)
    assert decisions["selected_action"].tolist() == ["Zero-shot", FIXED_THRESHOLD_ACTION]
    assert np.allclose(decisions["selected_actual_delta"], [0.0, -1.0])
    assert fixed_threshold_target_invariance(systems, threshold=2.0, seed=17)


def test_fixed_threshold_tuning_selects_best_source_feasible_gate() -> None:
    systems = pd.DataFrame(
        {
            "group_system_id": [f"S{i:03d}" for i in range(100)],
            "epa_region": np.repeat(np.arange(1, 11), 10),
            "k": 1,
            "base_mae": 2.0,
            "evaluation_rounds": 2,
            "evaluation_samples": 4,
            "abs_mean_residual": np.linspace(0.0, 9.9, 100),
            f"actual__{FIXED_THRESHOLD_ACTION}": -1.0,
        }
    )
    selected, search = tune_fixed_threshold(systems, risk_budget=0.12)
    assert selected["threshold_abs_mean_residual_ug_l"] == 0.0
    assert selected["feasible"]
    assert selected["adaptation_rate"] == 1.0
    assert len(search) >= 51


def test_small_cluster_mechanism_table_covers_each_metric() -> None:
    rows = []
    for region in (1, 2):
        rows.append(
            {
                "k": 1,
                "comparator": "forced_action_no_abstention",
                "outer_target_region": region,
                "equal_system_mae_difference": -0.1 * region,
                "negative_transfer_rate_difference": -0.01 * region,
                "strict_cvar90_regret_difference": -0.2 * region,
                "p95_regret_difference": -0.3 * region,
                "maximum_regret_difference": 0.0,
                "adaptation_rate_difference": -0.1 * region,
            }
        )
    result = small_cluster_mechanism_sensitivity(pd.DataFrame(rows))
    assert len(result) == 6
    assert result["regions"].eq(2).all()
    assert np.isfinite(result["student_t_ci_low"]).all()
