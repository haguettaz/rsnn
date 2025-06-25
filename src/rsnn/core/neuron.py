import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import lambertw

from rsnn.core.optim import Solver

FIRING_THRESHOLD = 1.0
REFRACTORY_RESET = -1.0
REFRACTORY_PERIOD = 1.0


@dataclass
class InSpike:
    conn_id: float
    time: float


@dataclass
class State:
    """Represents a neuronal state."""

    start: float
    c0: float
    c1: float
    dc0: float = 0.0
    dc1: float = 0.0
    length: float = float("inf")

    def forward(self, prev_state):
        dt = self.start - prev_state.start
        exp_mdt = math.exp(-dt)
        self.c0 = (prev_state.c0 + dt * prev_state.c1) * exp_mdt + self.dc0
        self.c1 = prev_state.c1 * exp_mdt + self.dc1

    def backward(self, next_state):
        self.length = next_state.start - self.start

    def first_crossing(self, threshold: float) -> Optional[float]:
        if self.c0 < threshold:
            if self.c1 > 0:
                dt = (
                    -(lambertw(-threshold / self.c1 * math.exp(-self.c0 / self.c1), 0))
                    - self.c0 / self.c1
                )
                if np.isreal(dt) and (dt >= 0.0) and (dt < self.length):
                    return float(self.start + dt)
            elif self.c1 < 0:
                dt = (
                    -(lambertw(-threshold / self.c1 * math.exp(-self.c0 / self.c1), -1))
                    - self.c0 / self.c1
                )
                if np.isreal(dt) and (dt >= 0.0) and (dt < self.length):
                    return float(self.start + dt.real)
            elif threshold < 0:
                dt = math.log(self.c0 / threshold)
                if (dt >= 0.0) and (dt < self.length):
                    return self.start + dt
            return None

        else:
            return self.start


@dataclass
class Neuron:
    """Represents a neuron with its states and firing times."""

    def __init__(
        self,
        threshold: float = FIRING_THRESHOLD,
        states: Optional[List[State]] = None,
        f_times: Optional[List[float]] = None,
    ):
        self.threshold = threshold
        self.states = states if states is not None else []
        self.f_times = f_times if f_times is not None else []

        # Sort states by start time in descending order
        self.states.sort(key=lambda s: s.start, reverse=True)

        # Sort firing times in ascending order
        self.f_times.sort()

    def merge_states(self, new_state: List[State]):
        """Merge new states into the existing states."""
        self.states += new_state
        self.states.sort(key=lambda s: s.start, reverse=True)

    def fire(self, time: float, noise: float = 0.0):
        self.f_times.append(time)
        self.threshold = FIRING_THRESHOLD + noise

        # Remove all states that start before the firing time and add the reset state
        index = bisect_left(self.states, -time, key=lambda s: -s.start)
        del self.states[index:]
        self.states.append(State(time, REFRACTORY_RESET, 0.0, dc0=REFRACTORY_RESET))

    def clean_states(self, time: float):
        """
        Clean the states by removing those that end before the given time, while ensuring validity of the first state.

        Args:
            time (float): The time before which states should be removed.
        """
        for i in range(len(self.states) - 1, 0, -1):
            if self.states[i - 1].start > time:
                break

            self.states[i - 1].forward(self.states[i])
            self.states.pop()

    # def receive_inspikes(self, inspikes):
    #     self.states += [
    #         State(inspike.time, 0.0, inspike.weight, dc1=inspike.weight)
    #         for inspike in inspikes
    #     ]
    #     self.states.sort(key=lambda s: s.start, reverse=True)

    def next_firing_time(self) -> Optional[float]:
        if len(self.states) > 1:
            self.states[-1].backward(self.states[-2])
            t = self.states[-1].first_crossing(self.threshold)
            if t is not None:
                return t

            for i in range(len(self.states) - 2, 0, -1):
                self.states[i].forward(self.states[i + 1])
                self.states[i].backward(self.states[i - 1])
                t = self.states[i].first_crossing(self.threshold)
                if t is not None:
                    return t

            self.states[0].forward(self.states[1])
            return self.states[0].first_crossing(self.threshold)

        elif len(self.states) == 1:
            return self.states[0].first_crossing(self.threshold)

        return None

    def step(self, time) -> Optional[float]:
        self.clean_states(time)
        return self.next_firing_time()

    def learn(
        self,
        f_times: np.ndarray,
        in_times: np.ndarray,
        in_channels: np.ndarray,
        eps,
        period,
        max_iter: int = 1000,
    ) -> Dict[int, float]:
        """Learn the synaptic weights to produce the desired firing times when fed with the prescribed input spikes.

        Args:
            f_times (List[float]): the desired firing times.
            inspikes (List[InSpike]): the collection of input spikes used to generate the firing times. The input spikes have an input ID attribute that is used to identify the corresponding synaptic weight.

        Returns:
            Dict[int, float]: the learned synaptic weights corresponding to the connection IDs of the input spikes. The keys are the connection IDs, and the values are the synaptic weights.
        """

        # # inspike.conn_id and inspike.time
        # conn_ids = list(set(inspike.conn_id for inspike in inspikes))
        # conn_ids.sort()

        if eps < 0.0 or eps > REFRACTORY_PERIOD:
            raise ValueError(f"eps must be in [0, {REFRACTORY_PERIOD}], got {eps}.")

        n_in_channels = np.max(in_channels, initial=-1) + 1

        # Init in_coef and ends, which remain constant during the optimization
        # Contains both information for potential and derivatives

        # Add constraints at firing times
        solver = Solver(n_in_channels)

        in_coef, starts, durations, bounds = [], [], [], []
        for n in range(f_times.size):
            # f_time = (f_times[n] - f_times[n - 1]) % period
            # bf_time = f_time - eps
            in_coef_n, starts_n, durations_n, bounds_n = compute_in_states(
                (f_times[n] - f_times[n - 1]) % period,
                (f_times[n] - eps - f_times[n - 1]) % period,
                (in_times - f_times[n - 1]) % period,
                in_channels,
                n_in_channels,
            )
            in_coef.append(in_coef_n[:, :-1])
            starts.append(starts_n[:-1] + f_times[n - 1])
            durations.append(durations_n[:-1])
            bounds.append(bounds_n[:-1])

            # Add the firing time constraint a_n x <= b_n (equivalent to -a_n x >= -b_n)
            solver.add_constraint(in_coef_n[0, -1], bounds_n[-1])

        in_coef = np.concatenate(in_coef, axis=1)
        starts = np.concatenate(starts)
        durations = np.concatenate(durations)
        bounds = np.concatenate(bounds)

        for i in range(max_iter):
            # 1. DCD algorithm: repeat the following steps until convergence (of the primal cost) or max_iter reached:
            # 1.1 dual coordinate descent step in the dual space
            # 1.2 convert the dual vector to a primal vector
            # 1.3 compute the cost of the primal vector
            solver.dual_coordinate_descent()

            # 2. Refine constraints based on the current primal solution. If no constraints are violated, then the primal solution is optimal.
            coef = np.inner(in_coef, solver.x)

            res = find_max_violation(coef, durations, bounds)

            if res is None:
                print("No violations found, optimal solution reached.")
                break
            else:
                imax, dtmax = res
                print(f"Violation found at index {imax} with tmax={tmax}")

                # 3. Add a new constraint based on the violation found
                # a = (in_coef[0, imax] + dtmax * in_coef[1, imax]) * np.exp(
                #     -dtmax
                # ).reshape(1, -1)
                # b = np.array([bounds[imax]])
                solver.add_constraint(
                    (in_coef[0, imax] + dtmax * in_coef[1, imax]) * np.exp(-dtmax),
                    bounds[imax],
                )

        # Needs to define a mapping from input IDs to synaptic weights. Either directly (sequentially, starting at 0) or through a dictionary.
        # In the end, we want to be able to update the connections at the network level.


# move the following in utils???
# def critical_potential(f_times, in_spikes, C, b, intervals, in_coef, in_int, weights, bound, tmin, tmax, opt=0.0, leq=True):
def find_max_violation(
    coef: np.ndarray, durations: np.ndarray, bounds: np.ndarray, tol: float = 1e-6
) -> Optional[Tuple[int, float]]:
    # coef and ends are assumed to be non-empty (otherwise, this is an unconstrained problem)
    # for derivatives, adapt coef and zmax accordingly
    # the intervals of interest are defined by the ends array, the intervals are (0, ends)
    # it has shape (n_intervals)
    # in_coef should only contain the coefficients for the intervals of interest
    # it has shape (2, n_inputs, n_intervals) with dtype np.float32
    # both in_coef and ends are constant

    ts = np.where(coef[1] <= 0, 0.0, np.clip(1 - coef[0] / coef[1], 0.0, durations))
    vs = (coef[0] + coef[1] * ts) * np.exp(-ts) - bounds
    imax = np.argmax(vs)

    if vs[imax] > tol:
        return (int(imax), float(ts[imax]))

    return None


# def init_in_coef(
#     f_times: np.ndarray,
#     in_times: np.ndarray,
#     in_channels: np.ndarray,
#     n_in_channels: np.intp,
# ) -> np.ndarray:
#     # assume ti < f_times[-1] for every ti in_times
#     f_times = np.sort(f_times)  # not necessary??
#     in_times = np.sort(in_times)

#     in_coef = np.zeros((2, in_times.size, n_in_channels))  # shape (n_inputs, 2)

#     # compute the insertion positions of the firing times in the input times array
#     ipos = np.searchsorted(
#         in_times, f_times, side="left"
#     )  # number of intervals for each firing time
#     ipos = np.append(ipos, in_times.size)
#     ipos = np.unique(ipos)

#     print(f"f_times: {f_times}")
#     print(f"in_times: {in_times}")
#     print(f"in_channels: {in_channels}")

#     n0 = 0
#     for nmax in ipos:
#         in_coef[1, n0, in_channels[n0]] = 1.0
#         for n in range(n0 + 1, nmax):
#             dt = in_times[n] - in_times[n - 1]
#             in_coef[0, n] += (in_coef[0, n - 1] + in_coef[1, n - 1] * dt) * np.exp(-dt)
#             in_coef[1, n] += in_coef[1, n - 1] * np.exp(-dt)
#             in_coef[1, n, in_channels[n]] += 1.0
#         n0 = nmax

#     return in_coef


def compute_in_states(
    f_time: np.double,
    bf_time: np.double,
    in_times: np.ndarray,
    in_channels: np.ndarray,
    n_in_channels: np.intp,
    firing_threshold: np.double = np.double(FIRING_THRESHOLD),
    zmax: np.double = np.double(0.0),
    dzmin: np.double = np.double(1e-6),
) -> Tuple[
    np.ndarray,  # in_coef: shape (2, n_intervals, n_channels)
    np.ndarray,  # starts: shape (n_intervals)
    np.ndarray,  # lengths: shape (n_intervals)
    np.ndarray,  # bounds: shape (n_intervals)
]:
    """
    Compute the coefficients (c0nk and c1nk) defining the states of every input (indexed by k) for any time between 0 and f_time, on disjoint intervals (indexed by n).
    The intervals partition the time range [0, f_time] in n_intervals = in_times.size + 3 intervals from the following time markers:
    - 0.0, the start of the time range
    - bf_time, the time before firing (the beginning of the active region)
    - f_time, the firing time
    - in_times, the input spike times.
    The intervals are reconstructed from their start and length.
    The signal (c0nk + c1nk * dt) * exp(-dt) for 0 <= dt < length[n] then corresponds to
    a) the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] + length[n] < bf_time.
    b) the derivative of the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] + length[n] < f_time and start[n] >= f_time.
    c) the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] = f_time.

    Args:
        f_time (np.double): _description_
        bf_time (np.double): _description_
        in_times (np.ndarray): _description_
        in_channels (np.ndarray): _description_
        n_in_channels (np.intp): _description_
        firing_threshold (np.double): _description_
        zmax (np.double): _description_
        dzmin (np.double): _description_

    Returns:
        np.ndarray: the coefficients defining the input signals by parts with shape (2, n_intervals, n_channels)
        np.ndarray: the times at which the intervals start with shape (n_intervals)
        np.ndarray: the lengths the intervals, with shape (n_intervals)
        np.ndarray: the template bounds for each interval, with shape (n_intervals)
    """

    # Extract the in_times and in_channels that are valid, i.e., within the range [0, f_time)
    valid = (in_times >= 0.0) & (in_times < f_time)
    in_times = in_times[valid]
    in_channels = in_channels[valid]

    # Initialize the starts array
    starts = np.concatenate((in_times, np.array([0.0, f_time, bf_time])))

    # Initialize the coefficients array
    in_coef = np.zeros((2, starts.size, n_in_channels))
    in_coef[1, np.arange(in_times.size), in_channels] = 1.0

    # Sort the coefficients according to their starts
    sort_ids = np.argsort(starts)
    starts = starts[sort_ids]
    lengths = np.diff(
        starts, append=f_time
    )  # time differences = lengths of the intervals
    # ends = starts + lengths
    bounds = np.full(starts.shape, zmax)

    in_coef = in_coef[:, sort_ids, :]
    for n in range(starts.size - 1):
        in_coef[1, n + 1] += in_coef[1, n] * np.exp(-lengths[n])
        in_coef[0, n + 1] += (in_coef[0, n] + in_coef[1, n] * lengths[n]) * np.exp(
            -lengths[n]
        )

    # The signal of interest on the active region is the (negative) derivative
    active = (starts >= bf_time) & (starts < f_time)
    in_coef[0, active] -= in_coef[1, active]  # (negative) derivative
    bounds[active] = -dzmin

    # The signal of interest for the firing time is the potential
    in_coef[:, -1] = -in_coef[:, -1]  # (negative) derivative
    bounds[-1] = -firing_threshold

    return in_coef, starts, lengths, bounds
