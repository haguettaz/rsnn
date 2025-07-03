import warnings
from dataclasses import dataclass, field
from math import dist
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.fftpack import shift

from rsnn.constants import REFRACTORY_PERIOD
from rsnn.utils import are_valid_f_times


def score(
    dists: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    shift: float,
) -> float:
    """
    Args:
        dist (npt.NDArray[np.float64]): The (sorted) time differences between the firing times and the reference firing times.
        shift (float): The time shift applied to the firing times.

    Returns:
        float: the score corresponding to the given shift.
    """
    imin = np.searchsorted(dists, shift - REFRACTORY_PERIOD / 2, side="right")
    imax = np.searchsorted(dists, shift + REFRACTORY_PERIOD / 2, side="left")
    # print(f"Score for shift {shift}: {score}")
    # print(f"Distances: {dists[imin:imax]}")
    # print(f"Weights: {weights[imin:imax]}")
    # print("\n")
    return np.sum(
        (REFRACTORY_PERIOD / 2.0 - np.abs(shift - dists[imin:imax]))
        * weights[imin:imax],
    )


class SimilarityMetric:
    """
    A class to measure the similarity of a multi-channel spike train with a reference.
    """

    def __init__(
        self, r_f_times: List[npt.NDArray[np.float64]], period: float = np.inf
    ):
        """
        Initializes the SimilarityMetric with reference firing times and a period.

        Args:
            r_f_times (List[npt.NDArray[np.float64]]): The reference firing times for each channel.
            period (float): The cycle period. For non-periodic spike trains, use np.inf (default).
        """
        if not are_valid_f_times(r_f_times, period):
            raise ValueError(
                "The reference firing times do not satisfy the refractory condition."
            )
        self.r_f_times = r_f_times
        self.period = period

    def measure(
        self, f_times: List[npt.NDArray[np.float64]]
    ) -> Tuple[float | np.float64, float | np.float64]:
        """
        Computes the similarity between the reference firing times and the provided firing times.

        Args:
            f_times (List[npt.NDArray[np.float64]]): The firing times to compare against the reference.

        Raises:
            ValueError: If the number of channels in the reference and the firing times do not match
                        or if the firing times do not satisfy the refractory condition.

        Returns:
            Tuple[float | np.float64, float | np.float64]: A tuple containing the precision and recall scores.
        """
        if len(self.r_f_times) != len(f_times):
            raise ValueError(
                "The number of channels in the reference and the firing times must match."
            )

        if not are_valid_f_times(f_times, self.period):
            warnings.warn(
                "The firing times do not satisfy the refractory condition.", UserWarning
            )
            return (np.nan, np.nan)

        # dist = [(r_f_times_n[np.newaxis, :] - f_times_n[:, np.newaxis]).reshape(-1) for (r_f_times_n, f_times_n) in zip(self.r_f_times, f_times)]

        n_channels = len(self.r_f_times)

        dists = np.concatenate(
            [
                # np.minimum(
                #     np.abs(r_f_times_n[np.newaxis, :] - f_times_n[:, np.newaxis]),
                #     np.abs(
                #         r_f_times_n[np.newaxis, :]
                #         + self.period
                #         - f_times_n[:, np.newaxis]
                #     ),
                # ).reshape(-1)
                (
                    r_f_times_n[np.newaxis, :]
                    - f_times_n[:, np.newaxis]
                    - self.period
                    * np.floor_divide(
                        r_f_times_n[np.newaxis, :]
                        - f_times_n[:, np.newaxis]
                        + self.period / 2,
                        self.period,
                    )
                ).reshape(-1)
                for (r_f_times_n, f_times_n) in zip(self.r_f_times, f_times)
            ]
        )
        print(f"Distances: {dists}")
        p_weights = np.concatenate(
            [
                np.full(
                    r_f_times_n.size * f_times_n.size,
                    np.divide(
                        2.0,
                        REFRACTORY_PERIOD * f_times_n.size * n_channels,
                        dtype=np.float64,
                    ),
                )
                for (r_f_times_n, f_times_n) in zip(self.r_f_times, f_times)
            ]
        )
        r_weights = np.concatenate(
            [
                np.full(
                    r_f_times_n.size * f_times_n.size,
                    np.divide(
                        2.0,
                        REFRACTORY_PERIOD * r_f_times_n.size * n_channels,
                        dtype=np.float64,
                    ),
                )
                for (r_f_times_n, f_times_n) in zip(self.r_f_times, f_times)
            ]
        )

        # Sort the distances and weights
        ids = np.argsort(dists)
        dists = dists[ids]
        p_weights = p_weights[ids]
        r_weights = r_weights[ids]

        precision = max(
            [score(dists, p_weights, shift) for shift in dists], default=0.0
        )
        recall = max([score(dists, r_weights, shift) for shift in dists], default=0.0)

        return precision, recall
