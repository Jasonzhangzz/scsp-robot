"""Finite-response VI bounds, including dependent contact rows and inactivity."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_motion_bound import bound_diagnostics
import numpy as np
import pytest


@pytest.mark.parametrize("seed", range(6))
def test_bounds_with_dependent_rows_and_different_active_sets(seed):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(12, 6))
    J[6:] = J[:6]  # rank-deficient mobility, regularized coupled LCP
    inverse = np.diag(rng.uniform(.1, 2., 6))
    gap = rng.uniform(-.0002, .0002, 12)
    free = rng.normal(size=6)*.1
    row, _, _ = bound_diagnostics(J, inverse, gap, free, .002, .008, .25, 1e-8)
    assert row["motion_error_hard"] <= row["combined_bound_hard"]+1e-7
    assert row["motion_error_matched_response"] <= row["matched_response_bound"]+1e-7
    assert row["lcp_residual"] <= 1e-8
    assert row["matched_response_lcp_residual"] <= 1e-8


def test_inactive_offsets_can_make_bound_positive_with_exact_zero_motion_error():
    row, scm, hard = bound_diagnostics(np.eye(6), np.eye(6), np.ones(6)*.01,
                                      np.ones(6)*.2, .002, .008, .25, 1e-8)
    np.testing.assert_array_equal(scm, hard)
    assert row["combined_bound_hard"] > 0
    assert row["motion_error_hard"] == 0
    assert row["matched_response_bound"] == 0
