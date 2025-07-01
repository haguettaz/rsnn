from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from rsnn.constants import REFRACTORY_PERIOD


def is_valid_f_times(f_times: npt.NDArray[np.float64], period: float | np.float64=np.inf) -> bool | np.bool:
    """
    Returns a boolean indicating whether the firing times satisfy the refractory condition.

    Args:
        f_times (npt.NDArray[np.float64]): The firing times of a neuron.
        period (float): the cycle period. For non-periodic spike trains, use np.inf (default).
    
    Returns:
        (bool): the boolean indicating satisfaction of the refractory condition.
    """
    if f_times.size > 1:
        sorted_f_times = np.sort(f_times)
        isi = np.diff(sorted_f_times, append=sorted_f_times[0] + period)
        return bool(np.all(isi >= REFRACTORY_PERIOD))
    elif f_times.size == 1:
        return period > REFRACTORY_PERIOD
    else:
        return True
    
def are_valid_f_times(f_times: List[npt.NDArray[np.float64]], period: float | np.float64=np.inf) -> bool | np.bool:
    """
    Returns a boolean indicating whether the firing times satisfy the refractory condition on every channel.

    Args:
        f_times (List[npt.NDArray[np.float64]]): List of firing times for each neuron.
        period (float): the cycle period. For non-periodic spike trains, use np.inf (default).

    Returns:
        (bool): the boolean indicating satisfaction of the refractory condition on every channel.
    """
    for f_times_c in f_times:
        if not is_valid_f_times(f_times_c, period):
            return False
    return True

def in_times_from_f_times(
    f_times: List[npt.NDArray[np.float64]], in_delays: npt.NDArray[np.float64], in_sources: npt.NDArray[np.intp]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """
    Extracts the input times from the firing times and input delays.

    Args:
        f_times (List[npt.NDArray[np.float64]]): List of firing times for each neuron.

    Returns:
        npt.NDArray[np.float64]: Array of input times.
    """
    in_times = np.concatenate(
        [
            f_times[in_src] + in_delay
            for in_src, in_delay in zip(in_sources, in_delays)
        ]
    )
    in_channels = np.concatenate(
        [
            np.full(f_times[in_src].size, in_channel, dtype=np.intp)
            for in_channel, in_src in enumerate(in_sources)
        ]
    )
    print(f"Input times shape: {in_times.shape}, Input channels shape: {in_channels.shape}")
    return in_times, in_channels

def connections_to_in_delays(
    connections: Dict[Tuple[int, int], List[Tuple[float, float]]], target_id: int
) -> Dict[int, List[float]]:
    """
    Converts a connections dictionary to a dictionary of input delays.

    Args:
        connections (Dict[Tuple[int, int], List[Tuple[float, float]]]): Connections dictionary.

    Returns:
        Dict[int, List[float]]: Dictionary mapping neuron source IDs to lists of delays.
    """
    in_delays = defaultdict(list)
    for (src_id, tgt_id), conns in connections.items() :
        if tgt_id == target_id:
            in_delays[src_id].extend(d for d, _ in conns)
    return in_delays

