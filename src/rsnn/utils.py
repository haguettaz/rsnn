from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt


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