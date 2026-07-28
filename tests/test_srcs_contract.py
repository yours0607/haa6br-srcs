from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze_optimized_results as analysis
import evaluate_risk_budget_variants as variants
import run_new_experiments as historical
import run_optimized_experiments as optimized


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "run_new_experiments.py",
    ROOT / "run_optimized_experiments.py",
    ROOT / "analyze_optimized_results.py",
    ROOT / "evaluate_risk_budget_variants.py",
)


def test_strict_cvar_uses_fixed_top_decile_count() -> None:
    one_positive = np.array([5.0, *([0.0] * 19)])
    boundary_ties = np.array([5.0, 5.0, *([0.0] * 18)])

    assert optimized.cvar90(one_positive) == pytest.approx(2.5)
    assert analysis.strict_cvar90(one_positive) == pytest.approx(2.5)
    assert optimized.cvar90(boundary_ties) == pytest.approx(5.0)


def test_primary_and_exploratory_risk_budgets_are_separated() -> None:
    assert optimized.PRIMARY_RISK_BUDGET == pytest.approx(0.12)
    assert optimized.RISK_BUDGETS == (0.08, 0.10, 0.12, 0.15)
    assert 0.06 not in optimized.RISK_BUDGETS
    assert 0.06 in variants.RISK_BUDGETS
    assert len(optimized.ACTIONS) == 20


def test_policy_features_exclude_target_and_audit_fields() -> None:
    normalized = {name.lower() for name in optimized.POLICY_FEATURES}
    forbidden = {name.lower() for name in historical.FORBIDDEN}
    forbidden.update({historical.TARGET.lower(), "observed", "future_loss", "target_loss"})

    assert normalized.isdisjoint(forbidden)
    assert all("audit_only" not in name for name in normalized)


def test_integrated_v1_validator_accepts_only_declared_relative_files(tmp_path: Path) -> None:
    for relative in historical.INTEGRATED_V1_US_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert historical.validate_integrated_v1(
        tmp_path,
        historical.INTEGRATED_V1_US_FILES,
    ) == tmp_path.resolve()

    with pytest.raises(FileNotFoundError, match="incomplete"):
        historical.validate_integrated_v1(tmp_path)


def test_prediction_cache_audit_does_not_record_private_path(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            optimized.REGION: [1, 2],
            optimized.GROUP: ["held", "train"],
            "sample_id": ["held-sample", "train-sample"],
        }
    )
    private_source = tmp_path / "private-user-cache"
    cache = optimized.RegionalPredictionCache(
        frame,
        features=[],
        model_name="Median",
        cache_dir=tmp_path / "working-cache",
        track="test",
        source_cache_dir=private_source,
    )
    np.save(cache._path((1,), 1), np.array([1.0]))

    cache.ensure([1])

    assert cache.audit_rows[0]["cache_source"] == "external validated cache"
    assert str(private_source) not in repr(cache.audit_rows[0])


def test_public_scripts_have_no_private_dependency_or_absolute_path() -> None:
    drive_literal = re.compile(r"[\"'][A-Za-z]:[\\/]")
    for path in SCRIPTS:
        source = path.read_text(encoding="utf-8")
        assert ".deps" not in source
        assert "PROJECT_DIR.parent / \"haa6br_data\"" not in source
        assert drive_literal.search(source) is None
