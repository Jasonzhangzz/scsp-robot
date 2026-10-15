"""Verify paired statistical quantities used in the manuscript table."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_lcp_comparison import display_units, formatted, paired_penetration, statistics


def test_penetration_is_paired_before_pooling_and_aligned_by_sample_id():
    rows = [
        dict(engine="scm_soft", sample_id="0", reference_mode="sliding", poststate_max_penetration_m=1.e-6),
        dict(engine="scm_soft", sample_id="1", reference_mode="sliding", poststate_max_penetration_m=3.e-6),
        dict(engine="mjwarp", sample_id="1", reference_mode="sliding", poststate_max_penetration_m=1.e-6),
        dict(engine="mjwarp", sample_id="0", reference_mode="sliding", poststate_max_penetration_m=3.e-6),
    ]
    stats = statistics(paired_penetration(rows, "mjwarp")["sliding"])
    # Equal per-model means do not imply zero paired differences.
    assert stats["mean"] == pytest.approx(2.e-6)
    assert stats["std"] == pytest.approx(0.)
    with pytest.raises(ValueError, match="Unmatched penetration sample IDs"):
        paired_penetration(rows[:-1], "mjwarp")


def test_nano_units_expose_roundoff_but_preserve_exact_zero():
    scale, unit, _ = display_units("separation", "linear_velocity_error_m_s")
    assert unit == "nm/s"
    assert formatted(statistics(np.array([1.e-10, 3.e-10]) * scale)) == "0.20 ± 0.10"
    assert formatted(statistics(np.zeros(2) * scale)) == "0.00 ± 0.00"
    assert display_units("sticking", "linear_velocity_error_m_s")[1] == "mm/s"
