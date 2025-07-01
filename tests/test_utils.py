import numpy as np

from rsnn.utils import *


def test_is_valid_f_times():

    # Test with valid firing times
    f_times = np.array([0.0, 1.0, 2.0, 3.0])
    assert is_valid_f_times(f_times, 4.0) is True

    # Test with invalid firing times (ISI < REFRACTORY_PERIOD)
    f_times_invalid = np.array([0.0, 1.0, 1.5])
    assert is_valid_f_times(f_times_invalid, 4.0) is False

    # Test with single firing time
    f_times_single = np.array([0.0])
    assert is_valid_f_times(f_times_single, 4.0) is True

    # Test with empty firing times
    f_times_empty = np.array([])
    assert is_valid_f_times(f_times_empty, 4.0) is True

    # Test with infinite period
    f_times_empty = np.array([0.0, 1.0, 2.0, 3.0])
    assert is_valid_f_times(f_times_empty, np.inf) is True
