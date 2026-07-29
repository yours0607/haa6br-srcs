from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from stage4_revision_analyses import (  # noqa: E402
    METHODS,
    assign_rounds,
    bonferroni_family_sensitivity,
    ensure_output_is_separate,
    exact_sign_flip_p,
    joint_family_bootstrap,
    joint_family_draws_frame,
    matched_bootstrap_contrasts,
    summarize_system_frame,
    system_method_errors,
)


def synthetic_predictions() -> pd.DataFrame:
    rows = []
    for region in (1, 2):
        for system_index in (1, 2):
            system = f"USA::{region}{system_index}"
            for round_index in (2, 3):
                observed = 10.0 + region + system_index + round_index
                zero = observed - 2.0
                row = {
                    "k": 1,
                    "epa_region": region,
                    "group_system_id": system,
                    "round_index": round_index,
                    "observed": observed,
                    "adapted": True,
                    "selected_action": "HistoryMean_050",
                    "SRCS": observed - 0.5,
                    "History mean": observed - 1.0,
                    "Capped History mean": observed - 0.8,
                    "Raw residual": observed - 1.2,
                    "Capped Raw residual": observed - 0.9,
                    "Zero-shot": zero,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def test_system_metrics_are_paired_and_directional() -> None:
    system = system_method_errors(synthetic_predictions(), 1)
    assert len(system) == 4
    assert set(METHODS).issubset(system.columns)
    assert np.allclose(system["regret__SRCS"], -1.5)
    assert np.allclose(system["regret__History mean"], -1.0)
    summary = summarize_system_frame(system, 1, "synthetic")
    srcs = summary.loc[summary["method"].eq("SRCS")].iloc[0]
    assert srcs["systems"] == 4
    assert np.isclose(srcs["equal_system_mae"], 0.5)
    assert np.isclose(srcs["negative_transfer_rate_gt_1e-12"], 0.0)


def test_assign_rounds_uses_dense_system_dates() -> None:
    frame = pd.DataFrame(
        {
            "group_system_id": ["A", "A", "A", "B"],
            "sample_id": ["a1", "a2", "a3", "b1"],
            "sample_date": ["2020-01-01", "2020-01-01", "2020-02-01", "2020-03-01"],
        }
    )
    result = assign_rounds(frame)
    assert result.loc[result["group_system_id"].eq("A"), "round_index"].tolist() == [1, 1, 2]
    assert result.loc[result["group_system_id"].eq("B"), "round_index"].tolist() == [1]


def test_exact_sign_flip_is_bounded_and_symmetric() -> None:
    p_positive = exact_sign_flip_p(np.array([1.0, 2.0, 3.0]))
    p_negative = exact_sign_flip_p(np.array([-1.0, -2.0, -3.0]))
    assert 0.0 <= p_positive <= 1.0
    assert np.isclose(p_positive, p_negative)


def test_bonferroni_family_resolves_missing_threshold_distribution_keys() -> None:
    system = system_method_errors(synthetic_predictions(), 1)
    contrast_frames = []
    distributions = {}
    for k in (1, 2, 3):
        contrasts, k_distributions = matched_bootstrap_contrasts(
            system,
            k=k,
            n_boot=100,
            seed=20260730 + k,
        )
        contrast_frames.append(contrasts)
        distributions.update(k_distributions)

    family = bonferroni_family_sensitivity(
        pd.concat(contrast_frames, ignore_index=True),
        distributions,
    )

    assert len(family) == 6
    assert family["family_size"].eq(6).all()
    cvar_rows = family.loc[
        family["metric"].eq("strict_cvar90_regret_difference")
    ]
    assert cvar_rows["regret_threshold_ug_l"].isna().all()
    assert np.isfinite(family[["adjusted_ci_low", "adjusted_ci_high"]]).all().all()


def test_joint_family_bootstrap_is_complete_and_deterministic() -> None:
    system = system_method_errors(synthetic_predictions(), 1)
    system_by_k = {k: system.copy() for k in (1, 2, 3)}
    first = joint_family_bootstrap(system_by_k, n_boot=100, seed=20260730)
    second = joint_family_bootstrap(system_by_k, n_boot=100, seed=20260730)

    assert len(first) == 6
    assert all(len(values) == 100 for values in first.values())
    assert all(np.array_equal(first[key], second[key]) for key in first)
    draws = joint_family_draws_frame(first, seed=20260730)
    assert len(draws) == 600
    assert set(draws["k"]) == {1, 2, 3}
    assert set(draws["metric"]) == {
        "negative_transfer_rate_difference",
        "strict_cvar90_regret_difference",
    }


def test_output_directory_cannot_be_nested_in_protected_input(tmp_path: Path) -> None:
    protected = tmp_path / "locked"
    protected.mkdir()
    with pytest.raises(ValueError, match="protected input"):
        ensure_output_is_separate(protected / "stage4", (protected,))
    ensure_output_is_separate(tmp_path / "separate", (protected,))
