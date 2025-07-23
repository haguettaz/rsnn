import numpy as np

from rsnn.jitters import compute_phi


def test_compute_phi():
    period = 6.0
    f_sources = np.array([0, 1, 2])
    f_times = np.array([0.0, 2.0, 4.0])
    c_sources = np.array([[1, 2, 2], [0, 2, 0], [0, 1, 1]])
    c_weights = np.ones_like(c_sources, dtype=np.float64)
    c_delays = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])

    phi = compute_phi(f_sources, f_times, c_sources, c_weights, c_delays, period)
    assert np.isclose(phi, 0.75764602)
    assert False