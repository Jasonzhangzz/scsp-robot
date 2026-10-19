"""MIQP projection must select a complementary mode, not just positive slacks."""

import numpy as np
import pytest

pytest.importorskip('pyscipopt')

from .scip_miqp import project


@pytest.mark.parametrize('target,expected', [([2., 2., 0.], [2., 0., 0.]),
                                            ([0., 2., 0.], [-.5, 1.5, 0.])])
def test_miqp_selects_globally_better_complementary_branch(target, expected):
    # gap=x+lambda-1. The two branches have analytic quadratic minima.
    value, info = project(np.eye(3), np.array(target), np.ones((1, 1)),
                          np.ones((1, 1)), np.zeros((1, 1)), np.array([-1.]), time_limit=2.)
    gap, force = value[0]+value[1]-1., value[1]
    assert gap >= -1e-7 and force >= -1e-7
    assert abs(gap*force) < 1e-7
    assert np.allclose(value, expected, atol=2e-4)
    assert info['status'] in ('optimal', 'gaplimit')
    assert info['mip_gap'] <= 1e-3
