import numpy as np

from rsnn.optim.utils import *


def test_modulo_with_offset():
    assert np.allclose(
        modulo_with_offset(np.array([1.0, 2.0, 3.0, 4.0]), 3.0),
        np.array([1.0, 2.0, 0.0, 1.0]),
    )
    assert np.allclose(
        modulo_with_offset(np.array([1.0, 2.0, 3.0, 4.0]), 3.0, 2.0),
        np.array([4.0, 2.0, 3.0, 4.0]),
    )
    assert np.allclose(
        modulo_with_offset(np.array([1.0, 2.0, 3.0, 4.0]), 3.0, -5.0),
        np.array([-5.0, -4.0, -3.0, -5.0]),
    )


def test_compute_states():
    tmin, tmax = 0.0, 10.0
    in_times = np.array([1.0, 2.0, 3.0])
    in_channels = np.array([0, 1, 2])
    c0_tmin = np.array([0.0] * 3 + [1.0])
    c1_tmin = np.zeros(4)
    weights = np.array([1.0] * 3 + [-1.0])

    start, c0, c1 = compute_states(in_times, in_channels, c0_tmin, c1_tmin, tmin, tmax)

    assert start.shape == (4,)
    assert c0.shape == (4, 4)
    assert c1.shape == (4, 4)

    assert np.allclose(start, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(c0[0], c0_tmin)
    assert np.allclose(c1[0], c1_tmin)

    assert np.isclose(
        np.inner((c0[0] + c1[0] * 0.1) * np.exp(-0.1), weights),
        -np.exp(-0.1)
        + np.sum((0.1 - in_times) * np.exp(-(0.1 - in_times)), where=in_times <= 0.1),
    )
    assert np.isclose(
        np.inner((c0[-1] + c1[-1] * 2.0) * np.exp(-2.0), weights),
        -np.exp(-5.0)
        + np.sum((5.0 - in_times) * np.exp(-(5.0 - in_times)), where=in_times <= 5.0),
    )


def test_find_maximum_violation():
    # Test a few single-interval cases
    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([10.0]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 0.36787944117144233)
    assert np.isclose(dtmax, 1.0)

    res = find_maximum_violation(
        np.array([-1.0]), np.array([0.0]), np.array([10.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([-1.0]), np.array([1.0]), np.array([10.0]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 0.1353352832366127)
    assert np.isclose(dtmax, 2.0)

    res = find_maximum_violation(
        np.array([-1.0]), np.array([-1.0]), np.array([1.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([-0.5]), np.array([-1.0]), np.array([1.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([-0.5]), np.array([-1.0]), np.array([2.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([1.0]), np.array([0.0]), np.array([2.0]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 1.0)
    assert np.isclose(dtmax, 0.0)

    res = find_maximum_violation(
        np.array([0.5]), np.array([1.0]), np.array([2.0]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 0.6065306597126334)
    assert np.isclose(dtmax, 0.5)

    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([np.inf]), np.array([0.0])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 0.36787944117144233)
    assert np.isclose(dtmax, 1.0)

    res = find_maximum_violation(
        np.array([-1.0]), np.array([0.0]), np.array([np.inf]), np.array([0.0])
    )
    assert res is None

    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([-np.inf])
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, np.inf)
    assert np.isclose(dtmax, 0.0)

    res = find_maximum_violation(
        np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([np.inf])
    )
    assert res is None

    # Test with multiple random intervals, all with the same length
    np.random.seed(42)  # For reproducibility
    res = find_maximum_violation(
        np.random.randn(10), np.random.randn(10), np.ones(10), np.zeros(10)
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 1.5792128155073915)
    assert np.isclose(dtmax, 0.0)

    res = find_maximum_violation(
        np.random.randn(10), np.random.randn(10), np.full(10, np.inf), np.zeros(10)
    )
    assert res is not None
    vmax, _, dtmax = res
    assert np.isclose(vmax, 1.465648768921554)
    assert np.isclose(dtmax, 0.0)
