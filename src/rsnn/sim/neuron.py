from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from numba import float64, int64
from numba.experimental import jitclass

from rsnn.constants import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.sim.utils import first_crossing

spec = [
    ("id", int64),
    ("threshold", float64),
    ("f_times", float64[:]),  # Firing times
    ("start", float64[:]),  # Start times of states
    ("length", float64[:]),  # Length of states
    ("c0s", float64[:]),  # c0 coefficients of states
    ("c1s", float64[:]),  # c1 coefficients of states
    ("dc0s", float64[:]),  # dc0 coefficients of states
    ("dc1s", float64[:]),  # dc1 coefficients of states
]

@jitclass
class Neuron:
    """Represents a neuron with its states and firing times."""
    spec = [
        ("id", int64),
        ("threshold", float64),
        ("f_times", float64[:]),  # Firing times
        ("start", float64[:]),  # Start times of states
        ("length", float64[:]),  # Length of states
        ("c0s", float64[:]),  # c0 coefficients of states
        ("c1s", float64[:]),  # c1 coefficients of states
        ("dc0s", float64[:]),  # dc0 coefficients of states
        ("dc1s", float64[:]),  # dc1 coefficients of states
    ]

    def __init__(self, id: int):
        """Initialize the neuron with default parameters.

        Args:
            id (int): The unique identifier for the neuron.
        """

        self.id = id  # Neuron ID

        self.threshold = FIRING_THRESHOLD  # Firing threshold

        self.f_times = np.array([], dtype=np.float64)  # Firing times

        # State variables
        self.start = np.array([-np.inf, np.inf])
        self.length = np.array([np.inf, np.inf])
        self.c0s = np.zeros((2,), dtype=np.float64)
        self.c1s = np.zeros((2,), dtype=np.float64)
        self.dc0s = np.zeros((2,), dtype=np.float64)
        self.dc1s = np.zeros((2,), dtype=np.float64)

    def add_f_times(self, f_times: npt.NDArray[np.float64]):
        """Add firing times to the neuron's firing times.

        Args:
            f_times (npt.NDArray[np.float64]): Array of firing times to add."""
        self.f_times = np.sort(np.concatenate((self.f_times, f_times)))

    def init_initial_state(self):
        """Initialize the initial state of the neuron."""
        self.start[0] = -np.inf
        self.length[0] = np.inf
        self.c0s[0] = 0.0
        self.c1s[0] = 0.0
        self.dc0s[0] = 0.0
        self.dc1s[0] = 0.0

    def clear_states(self):
        """Clear the neuron's states and reset to initial state."""
        self.start = np.array([-np.inf, np.inf], dtype=np.float64)
        self.length = np.array([np.inf, np.inf], dtype=np.float64)
        self.c0s = np.zeros((2,), dtype=np.float64)
        self.c1s = np.zeros((2,), dtype=np.float64)
        self.dc0s = np.zeros((2,), dtype=np.float64)
        self.dc1s = np.zeros((2,), dtype=np.float64)
        # self.states = np.concatenate((initial_state(), final_state()), axis=0)

    def add_states(
        self,
        start: npt.NDArray[np.float64],
        length: npt.NDArray[np.float64],
        c0s: npt.NDArray[np.float64],
        c1s: npt.NDArray[np.float64],
        dc0s: npt.NDArray[np.float64],
        dc1s: npt.NDArray[np.float64],
    ):
        """Merge new states into the existing states.

        Args:
            start (npt.NDArray[np.float64]): Array of start times for the states.
            length (npt.NDArray[np.float64]): Array of length for the states.
            c0s (npt.NDArray[np.float64]): Array of c0 coefficients for the states.
            c1s (npt.NDArray[np.float64]): Array of c1 coefficients for the states.
            dc0s (npt.NDArray[np.float64]): Array of dc0 coefficients for the states.
            dc1s (npt.NDArray[np.float64]): Array of dc1 coefficients for the states.
        """

        self.start = np.concatenate((self.start, start), axis=0)
        self.length = np.concatenate((self.length, length), axis=0)
        self.c0s = np.concatenate((self.c0s, c0s), axis=0)
        self.c1s = np.concatenate((self.c1s, c1s), axis=0)
        self.dc0s = np.concatenate((self.dc0s, dc0s), axis=0)
        self.dc1s = np.concatenate((self.dc1s, dc1s), axis=0)

        sorter = np.argsort(self.start)
        self.start = self.start[sorter]
        self.length = self.length[sorter]
        self.c0s = self.c0s[sorter]
        self.c1s = self.c1s[sorter]
        self.dc0s = self.dc0s[sorter]
        self.dc1s = self.dc1s[sorter]

    def fire(self, f_time: float, noise: float = 0.0):
        # Add the firing time to the list of firing times
        self.f_times = np.append(self.f_times, f_time)

        # Add threshold noise
        self.threshold = FIRING_THRESHOLD + noise

        # Enter refractory period
        self.recover(f_time)

    def recover(self, f_time: float):
        """
        Clear the states and enter the refractory period at the given firing time.
        This method ensures that the neuron's states consist at least of the initial state, the refractory state, and the final state.
        """
        # Clear states and enter refractory period
        ipos = np.searchsorted(self.start, f_time, side="left")  # always >= 1
        if ipos > 1:
            self.start = self.start[(ipos - 2) :]
            self.length = self.length[(ipos - 2) :]
            self.c0s = self.c0s[(ipos - 2) :]
            self.c1s = self.c1s[(ipos - 2) :]
            self.dc0s = self.dc0s[(ipos - 2) :]
            self.dc1s = self.dc1s[(ipos - 2) :]

            self.init_initial_state()
        else:
            self.start = np.concatenate((np.array([-np.inf]), self.start))
            self.length = np.concatenate((np.array([np.inf]), self.length))
            self.c0s = np.concatenate((np.array([0.0]), self.c0s))
            self.c1s = np.concatenate((np.array([0.0]), self.c1s))
            self.dc0s = np.concatenate((np.array([0.0]), self.dc0s))
            self.dc1s = np.concatenate((np.array([0.0]), self.dc1s))

        self.start[1] = f_time
        self.length[1] = np.inf
        self.c0s[1] = REFRACTORY_RESET
        self.c1s[1] = 0.0
        self.dc0s[1] = REFRACTORY_RESET
        self.dc1s[1] = 0.0

    def clean_states(self, time: float):
        """
        Clean the states by removing those that end before the given time, while ensuring validity of the first state.
        Note 1: the states necessarily contain at least two states: the initial state and the final state.
        Note 2: one should have self.states[0]["start"] <= time

        Args:
            time (float): The time before which states should be removed.
        """
        ipos = np.searchsorted(self.start, time, side="right") - 1  # always >= 0

        if ipos > 0:  # if there are states to clean
            for i in range(1, ipos + 1):

                dt = np.nan_to_num(self.start[i] - self.start[i - 1])
                self.c0s[i] = (self.c0s[i - 1] + dt * self.c1s[i - 1]) * np.exp(
                    -dt
                ) + self.dc0s[i]
                self.c1s[i] = self.c1s[i - 1] * np.exp(-dt) + self.dc1s[i]

                # update_state_forward_(
                #     self.states[i],
                #     self.states[i - 1],
                # )

            self.start = self.start[ipos - 1 :]
            self.length = self.length[ipos - 1 :]
            self.c0s = self.c0s[ipos - 1 :]
            self.c1s = self.c1s[ipos - 1 :]
            self.dc0s = self.dc0s[ipos - 1 :]
            self.dc1s = self.dc1s[ipos - 1 :]

            self.init_initial_state()

            self.dc0s[1] = self.c0s[1]
            self.dc1s[1] = self.c1s[1]

    def next_firing_time(self, tmax: float) -> np.float64 | float:
        # t = np.nan
        for n in range(1, self.start.shape[0] - 1):
            if self.start[n] < tmax:
                self.length[n] = self.start[n + 1] - self.start[n]
                self.c0s[n] = (
                    self.c0s[n - 1]
                    + np.nan_to_num(self.length[n - 1]) * self.c1s[n - 1]
                ) * np.exp(-self.length[n - 1]) + self.dc0s[n]
                self.c1s[n] = (
                    self.c1s[n - 1] * np.exp(-self.length[n - 1]) + self.dc1s[n]
                )

                # update_state_forward_backward_(
                #     self.states[n],
                #     prev_state=self.states[n - 1],
                #     next_state=self.states[n + 1],
                # )

                t = first_crossing(
                    self.start[n],
                    self.length[n],
                    self.c0s[n],
                    self.c1s[n],
                    self.threshold,
                )
                # t = first_crossing(self.states[n], self.threshold)
                if np.isfinite(t):
                    return t if t < tmax else np.nan
            else:
                return np.nan

        return np.nan
        # return t if t is not None and t < tmax else None

    def step(self, tmin, tmax) -> np.float64 | float:
        self.clean_states(tmin)
        return self.next_firing_time(tmax)
