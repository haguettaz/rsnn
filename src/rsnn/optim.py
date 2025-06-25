from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from rsnn.constants import (FIRING_THRESHOLD, LEFT_LIMIT, REFRACTORY_PERIOD,
                            REFRACTORY_RESET)

# def compute_state_template(
#     f_time: np.float64,  # optional??
#     bf_time: np.float64,  # optional??
#     in_times: npt.NDArray[np.float64],
#     in_channels: npt.NDArray[np.intp],
#     n_in_channels: np.intp,
#     f_thresh: np.float64,
#     zmax: np.float64,
#     dzmin: np.float64,
# ) -> Tuple[
#     npt.NDArray[np.float64],  # zi_f_c: shape (n_in_channels + 1)
#     npt.NDArray[np.float64],  # z_f_lim: shape (1,)
#     npt.NDArray[np.float64],  # z_start: shape (n_z_intervals)
#     npt.NDArray[np.float64],  # z_length: shape (n_z_intervals)
#     npt.NDArray[np.float64],  # z_lim: shape (n_z_intervals)
#     npt.NDArray[np.float64],  # zi_c0: shape (n_z_intervals, n_in_channels + 1)
#     npt.NDArray[np.float64],  # zi_c1: shape (n_z_intervals, n_in_channels + 1)
#     npt.NDArray[np.float64],  # dz_start: shape (n_dz_intervals)
#     npt.NDArray[np.float64],  # dz_length: shape (n_dz_intervals)
#     npt.NDArray[np.float64],  # dz_lim: shape (n_dz_intervals)
#     npt.NDArray[np.float64],  # dzi_c0: shape (n_dz_intervals, n_in_channels + 1)
#     npt.NDArray[np.float64],  # dzi_c1: shape (n_dz_intervals, n_in_channels + 1)
# ]:
#     """
#     Compute the coefficients (c0nk and c1nk) defining the states of every input (indexed by k) for any time between 0 and f_time, on disjoint intervals (indexed by n).
#     The intervals partition the time range [0, f_time] in n_intervals = in_times.size + 3 intervals from the following time markers:
#     - 0.0, the start of the time range
#     - bf_time, the time before firing (the beginning of the active region)
#     - f_time, the firing time
#     - in_times, the input spike times.
#     The intervals are reconstructed from their start and length.
#     The signal (c0nk + c1nk * dt) * exp(-dt) for 0 <= dt < length[n] then corresponds to
#     a) the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] + length[n] < bf_time.
#     b) the derivative of the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] + length[n] < f_time and start[n] >= f_time.
#     c) the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] = f_time.

#     Args:
#         f_time (np.double): _description_
#         bf_time (np.double): _description_
#         in_times (np.ndarray): _description_
#         in_channels (np.ndarray): _description_
#         n_in_channels (np.intp): _description_
#         zmax (np.double): _description_
#         dzmin (np.double): _description_

#     Returns:
#         Tuple[npt.NDArray[np.float64], ...]: A tuple containing:
#             - zi_f_c: shape (n_in_channels + 1)
#             - z_f_lim: shape (1,)
#             - z_start: shape (n_z_intervals)
#             - z_length: shape (n_z_intervals)
#             - z_lim: shape (n_z_intervals)
#             - zi_c0: shape (n_z_intervals, n_in_channels + 1)
#             - zi_c1: shape (n_z_intervals, n_in_channels + 1)
#             - dz_start: shape (n_dz_intervals)
#             - dz_length: shape (n_dz_intervals)
#             - dz_lim: shape (n_dz_intervals)
#             - dzi_c0: shape (n_dz_intervals, n_in_channels + 1)
#             - dzi_c1: shape (n_dz_intervals, n_in_channels + 1)
#     """
#     # Extract the in_times and in_channels that are valid
#     if f_time is not None:
#         valid = (in_times >= 0.0) & ((in_times < f_time))
#     else:
#         valid = in_times >= 0.0

#     in_times = in_times[valid]
#     in_channels = in_channels[valid]

#     # Initialize the starts array
#     if bf_time is not None:
#         z_start = np.concatenate((in_times, np.array([bf_time, 0.0])))
#     else:
#         z_start = np.concatenate((in_times, np.array([0.0])))

#     # Initialize the coefficients array
#     zi_c0 = np.zeros((z_start.size, n_in_channels + 1))
#     zi_c1 = np.zeros((z_start.size, n_in_channels + 1))
#     zi_c1[np.arange(in_times.size), in_channels] = 1.0
#     zi_c0[-1, n_in_channels] = REFRACTORY_RESET  # refractory reset

#     # Sort the coefficients according to their starts
#     sorter = np.argsort(z_start)
#     z_start = z_start[sorter]
#     z_length = np.diff(
#         z_start, append=f_time
#     )  # time differences = lengths of the intervals

#     # Input signals for the potential
#     zi_c0 = zi_c0[sorter]
#     zi_c1 = zi_c1[sorter]
#     for n in range(z_start.size - 1):
#         zi_c0[n + 1] += (zi_c0[n] + zi_c1[n] * z_length[n]) * np.exp(-z_length[n])
#         zi_c1[n + 1] += zi_c1[n] * np.exp(-z_length[n])

#     # print(f"zi_c0={zi_c0}")
#     # print(f"zi_c1={zi_c1}")

#     # Potential constraints at f_time (z >= f_thresh <=> -z <= -f_thresh)
#     zi_f_c = -(zi_c0[-1] + zi_c1[-1] * z_length[-1]) * np.exp(-z_length[-1])
#     # print(f"zi_f_c={zi_f_c}")
#     # The last coefficient is the refractory reset
#     z_f_lim = np.array(-f_thresh)

#     # Potential constraints before f_time (z <= zlim))
#     z_lim = np.full(z_start.shape, zmax)
#     z_lim[z_start >= bf_time] = f_thresh

#     # Potential derivative constraints (z' >= dzmin <=> -z' <= -dzmin)
#     active = z_start >= bf_time
#     dz_start = z_start[active]
#     dz_length = z_length[active]
#     dz_lim = np.full(dz_start.shape, -dzmin)
#     dzi_c0 = zi_c0[active] - zi_c1[active]
#     dzi_c1 = zi_c1[active]

#     return (
#         zi_f_c,
#         z_f_lim,
#         z_start,
#         z_length,
#         z_lim,
#         zi_c0,
#         zi_c1,
#         dz_start,
#         dz_length,
#         dz_lim,
#         dzi_c0,
#         dzi_c1,
#     )


def compute_ck(
    in_times: np.float64 | npt.NDArray[np.float64],
    in_channels: np.intp | npt.NDArray[np.intp],
    n_in_channels: np.intp | int,
    # f_time: Optional[float],
    times: (
        np.float64 | npt.NDArray[np.float64]
    ),  # time markers (largest is the end time, start time is 0.0)
    reset: (
        float | np.float64 | npt.NDArray[np.float64]
    ) = REFRACTORY_RESET,  # initial value (at start time)
    # offset: (
    #     float | np.float64 | npt.NDArray[np.float64]
    # ) = 0.0,  # offset to apply to start times
) -> Tuple[
    npt.NDArray[np.float64],  # start: shape (n_intervals)
    npt.NDArray[np.float64],  # length: shape (n_intervals)
    npt.NDArray[np.float64],  # ck0: shape (n_intervals, n_in_channels + 1)
    npt.NDArray[np.float64],  # ck1: shape (n_intervals, n_in_channels + 1)
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
        zmax (np.double): _description_
        dzmin (np.double): _description_

    Returns:
        Tuple[npt.NDArray[np.float64], ...]: A tuple containing:
            - start: shape (n_intervals)
            - length: shape (n_intervals)
            - ck0: shape (n_intervals, n_in_channels + 1)
            - ck1: shape (n_intervals, n_in_channels + 1)
    """
    times = np.asarray(times, dtype=np.float64)
    in_times = np.asarray(in_times, dtype=np.float64)
    in_channels = np.asarray(in_channels, dtype=np.intp)

    # Extract the in_times and in_channels that are valid
    end = np.max(times, initial=0.0)
    valid = (in_times >= 0.0) & ((in_times < end))
    in_times = in_times[valid]
    in_channels = in_channels[valid]

    # Initialize the starts array
    start = np.concatenate((in_times, times, np.array([0.0])))

    # Initialize the coefficients array
    ck0 = np.zeros((start.size, n_in_channels + 1))
    ck1 = np.zeros((start.size, n_in_channels + 1))
    ck1[np.arange(in_times.size), in_channels] = 1.0
    ck0[-1, n_in_channels] = reset  # refractory reset

    # Sort the coefficients according to their starts
    sorter = np.argsort(start)
    start = start[sorter]
    length = np.diff(start, append=end)  # time differences = lengths of the intervals

    # Input signals for the potential
    ck0 = ck0[sorter]
    ck1 = ck1[sorter]
    for n in range(start.size - 1):
        ck0[n + 1] += (ck0[n] + ck1[n] * length[n]) * np.exp(-length[n])
        ck1[n + 1] += ck1[n] * np.exp(-length[n])

    return start, length, ck0, ck1


def find_maximum(
    c0: npt.NDArray[np.float64],
    c1: npt.NDArray[np.float64],
    length: npt.NDArray[np.float64],
    lim: npt.NDArray[np.float64],
) -> Tuple[int | np.intp, float | np.float64, float | np.float64]:
    # coef and ends are assumed to be non-empty (otherwise, this is an unconstrained problem)
    # for derivatives, adapt coef and zmax accordingly
    # the intervals of interest are defined by the ends array, the intervals are (0, ends)
    # it has shape (n_intervals)
    # in_coef should only contain the coefficients for the intervals of interest
    # it has shape (2, n_inputs, n_intervals) with dtype np.float32
    # both in_coef and ends are constant

    # ignore the RuntimeWarning due to division by zero
    ts = np.where(c1 <= 0, 0.0, np.clip(1 - c0 / c1, 0.0, length))
    vs = (c0 + c1 * ts) * np.exp(-ts) - lim
    imax = np.argmax(vs)

    return imax, vs[imax], ts[imax]


class Solver:
    def __init__(
        self,
        n_vars: int | np.intp,
    ):
        """
        Initialize the solver with parameters.

        Parameters:
        """
        self.n_vars = n_vars  # Number of variables
        self.n_cstrs = np.intp(0)  # Number of constraints (= number of dual variables)

        self.x = np.zeros(self.n_vars, dtype=np.float64)
        self.xt = np.zeros(self.n_cstrs, dtype=np.float64)
        self.cost = np.float64(0.0)

        self.A = np.empty((0, n_vars))
        self.b = np.empty((0,))

        self.Vfxt = np.empty((0, 0), dtype=np.float64)
        self.mfxt = np.empty((0), dtype=np.float64)

    def reset(self, n_vars: int | np.intp):
        """
        Reset the solver with a new number of variables.

        Parameters:
            n_vars (int | np.intp): The new number of variables.
        """
        self.n_vars = n_vars
        self.n_cstrs = np.intp(0)

        self.x = np.zeros(self.n_vars, dtype=np.float64)
        self.xt = np.zeros(self.n_cstrs, dtype=np.float64)
        self.cost = np.float64(0.0)

        self.A = np.empty((0, n_vars))
        self.b = np.empty((0,))

        self.Vfxt = np.empty((0, 0), dtype=np.float64)
        self.mfxt = np.empty((0), dtype=np.float64)

    def dual_coordinate_descent(
        self, feas_tol: float = 1e-6, conv_tol: float = 1e-9, n_iter: int = 1000
    ) -> int:
        """
        Run coordinate descent in the dual space to optimize the dual solution.
        This updates the dual solution vector xt based on the current constraints.

        Returns:
            (int): the status of the optimization:
                - 1 if the optimization converged to a solution,
                - 0 if the optimization failed to converge,
                - -1 if the optimization failed to find a feasible solution.

        """
        if self.n_cstrs > 0:
            prev_cost = np.inf
            for i in range(n_iter):
                for n in range(self.n_cstrs):
                    self.xt[n] = np.clip(
                        self.xt[n] + self.mfxt[n] - self.Vfxt[n] @ self.xt, 0, None
                    )
                self.x = -self.xt @ self.A

                self.cost = np.linalg.norm(self.x).astype(np.float64)
                if np.abs(prev_cost - self.cost) < conv_tol:
                    print(f"Converged after {i+1} iterations with cost {self.cost}")
                    if self.is_valid(feas_tol):
                        print("Feasible solution found.")
                        return 1
                    # if np.all(self.A @ self.x - self.b <= self.feas_tol):
                    #     return (1, cost)
                    else:
                        print("Primal problem is infeasible.")
                        self.cost = np.inf
                        return -1

                prev_cost = self.cost

            print("Reached maximum iterations without convergence.")
            return 0
        else:
            self.cost = 0.0
            self.x = np.zeros(self.n_vars, dtype=np.float64)
            print("Feasible (trivial) solution found.")
            return 1

    def is_valid(self, feas_tol: float) -> np.bool:
        """
        Check if the current solution is valid, i.e., if all constraints are satisfied.

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        return np.all(self.A @ self.x - self.b <= feas_tol)

    def add_constraint(
        self,
        ant: npt.NDArray[np.float64],
        bn: npt.NDArray[np.float64],
    ) -> int:
        ant_norm = np.linalg.norm(ant)

        if ant_norm > 0.0:
            self.n_cstrs += 1

            self.xt = np.append(self.xt, 0.0)

            ant /= ant_norm
            bn /= ant_norm

            Aant = (self.A @ ant).reshape(-1, 1)

            self.Vfxt = np.block([[self.Vfxt, Aant], [Aant.T, 1.0]])
            self.mfxt = np.append(self.mfxt, -bn)

            self.A = np.vstack((self.A, ant))
            self.b = np.append(self.b, bn)
            return 0
        else:
            return -1


@dataclass
class Neuron:
    """
    A class representing a neuron with its input channels.
    """

    # Synaptic weights for each input channel
    weight: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )

    # State variables
    z_start: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    z_length: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    z_lim: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    z_ck0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    z_ck1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    z_c0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )  # c0 = np.inner(i_c0[:, :-1], weight) + self.i_c0[:, -1]
    z_c1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )  # c1 = np.inner(i_c1[:, :-1], weight) + self.i_c1[:, -1]
    dz_start: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    dz_length: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    dz_lim: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    dz_ck0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    dz_ck1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    dz_c0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    dz_c1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )

    # Solver
    solver: Solver = field(default_factory=lambda: Solver(0))

    @property
    def n_in_channels(self) -> int:
        """Number of input channels."""
        return self.weight.size

    @property
    def n_z_intervals(self) -> int:
        """Number of intervals partitioning the time axis for the potential."""
        return self.z_start.size

    @property
    def n_dz_intervals(self) -> int:
        """Number of intervals partitioning the time axis for the potential derivative."""
        return self.dz_start.size

    def init_solver(
        self,
        f_times: npt.NDArray[np.float64],
        in_times: npt.NDArray[np.float64],
        in_channels: npt.NDArray[np.intp],
        period: np.float64 | float,
        eps: np.float64 | float = REFRACTORY_PERIOD,
        f_thresh: np.float64 | float = FIRING_THRESHOLD,
        zmax: np.float64 | float = 0.0,
        dzmin: np.float64 | float = 1e-2,
    ):
        if eps < 0.0 or eps > REFRACTORY_PERIOD:
            raise ValueError(f"eps must be in [0, {REFRACTORY_PERIOD}], got {eps}.")

        if zmax >= f_thresh:
            raise ValueError(
                f"zmax must be less than f_thresh ({f_thresh}), got {zmax}."
            )

        if dzmin <= 0.0:
            raise ValueError(f"dzmin must be positive, got {dzmin}.")

        n_in_channels = np.max(in_channels, initial=-1) + 1
        self.solver.reset(n_in_channels)
        self.weight = np.zeros(n_in_channels, dtype=np.float64)

        if f_times.size > 0:
            f_times.sort()
            times = np.diff(f_times, append=f_times[0] + period)[
                :, np.newaxis
            ] - np.array([[eps, 0.0]])
            in_times = (in_times[np.newaxis, :] - f_times[:, np.newaxis]) % period

            z_start, z_length, z_lim, z_ck0, z_ck1 = [], [], [], [], []
            dz_start, dz_length, dz_lim, dz_ck0, dz_ck1 = [], [], [], [], []

            for in_times_n, times_n, offset in zip(in_times, times, f_times):
                start, length, ck0, ck1 = compute_ck(
                    in_times_n,
                    in_channels,
                    n_in_channels,
                    times_n,
                    REFRACTORY_RESET,
                )

                # Firing time constraint | z >= f_thresh
                self.solver.add_constraint(
                    # zi_f_c_n[:-1], z_f_lim_n - zi_f_c_n[-1]
                    -ck0[-1, :-1],
                    -f_thresh + ck0[-1, -1],
                )

                ## Silent time intervals | z <= zmax
                silent = start < times_n[0]
                z_start.append(start[silent] + offset)
                z_length.append(length[silent])
                z_lim.append(np.full(start[silent].shape, zmax))
                z_ck0.append(ck0[silent])
                z_ck1.append(ck1[silent])

                ## Active time intervals | z <= f_thresh and dz >= dzmin
                active = (start >= times_n[0]) & (start < times_n[-1])
                z_start.append(start[active] + offset)
                z_length.append(length[active])
                z_lim.append(np.full(start[active].shape, f_thresh))
                z_ck0.append(ck0[active])
                z_ck1.append(ck1[active])
                dz_start.append(start[active] + offset)
                dz_length.append(
                    (length[active] - LEFT_LIMIT).clip(min=0.0)
                )  # the potential derivative is right-continuous
                dz_lim.append(np.full(start[active].shape, -dzmin))
                dz_ck0.append(ck0[active] - ck1[active])
                dz_ck1.append(ck1[active])

            self.z_start = np.concatenate(z_start)
            self.z_length = np.concatenate(z_length)
            self.z_lim = np.concatenate(z_lim)
            self.z_ck0 = np.concatenate(z_ck0)
            self.z_ck1 = np.concatenate(z_ck1)

            self.dz_start = np.concatenate(dz_start)
            self.dz_length = np.concatenate(dz_length)
            self.dz_lim = np.concatenate(dz_lim)
            self.dz_ck0 = np.concatenate(dz_ck0)
            self.dz_ck1 = np.concatenate(dz_ck1)
        else:
            in_times = in_times % period
            start, length, ck0, ck1 = compute_ck(
                in_times,
                in_channels,
                n_in_channels,
                np.array([0.0, period]),
                0.0,
            )

            ## Silent time intervals | z <= zmax
            silent = start < period
            self.z_start = start[silent]
            self.z_length = length[silent]
            self.z_lim = np.full(start[silent].shape, zmax)
            self.z_ck0 = ck0[silent]
            self.z_ck1 = ck1[silent]

            self.dz_start = np.empty((0,), dtype=np.float64)
            self.dz_length = np.empty((0,), dtype=np.float64)
            self.dz_lim = np.empty((0,), dtype=np.float64)
            self.dz_ck0 = np.empty((0, n_in_channels + 1), dtype=np.float64)
            self.dz_ck1 = np.empty((0, n_in_channels + 1), dtype=np.float64)

    def refine_constraints(self, feas_tol: float = 1e-6) -> int:

        if self.n_z_intervals > 0:
            self.z_c0 = np.inner(self.z_ck0[:, :-1], self.solver.x) + self.z_ck0[:, -1]
            self.z_c1 = np.inner(self.z_ck1[:, :-1], self.solver.x) + self.z_ck1[:, -1]
            z_imax, z_vmax, z_dtmax = find_maximum(
                self.z_c0, self.z_c1, self.z_length, self.z_lim
            )
        else:
            z_vmax = -np.inf

        if self.n_dz_intervals > 0:
            self.dz_c0 = (
                np.inner(self.dz_ck0[:, :-1], self.solver.x) + self.dz_ck0[:, -1]
            )
            self.dz_c1 = (
                np.inner(self.dz_ck1[:, :-1], self.solver.x) + self.dz_ck1[:, -1]
            )
            dz_imax, dz_vmax, dz_dtmax = find_maximum(
                self.dz_c0, self.dz_c1, self.dz_length, self.dz_lim
            )
        else:
            dz_vmax = -np.inf

        if z_vmax <= feas_tol and dz_vmax <= feas_tol:
            return 1
        else:
            if z_vmax > dz_vmax:
                return self.solver.add_constraint(
                    (self.z_ck0[z_imax, :-1] + z_dtmax * self.z_ck1[z_imax, :-1])
                    * np.exp(-z_dtmax),
                    self.z_lim[z_imax]
                    - (self.z_ck0[z_imax, -1] + z_dtmax * self.z_ck1[z_imax, -1])
                    * np.exp(-z_dtmax),
                )
                # print(
                #     f"New potential constraint added at time {self.z_start[z_imax] + z_dtmax}!"
                # )

            else:
                return self.solver.add_constraint(
                    (self.dz_ck0[dz_imax, :-1] + dz_dtmax * self.dz_ck1[dz_imax, :-1])
                    * np.exp(-dz_dtmax),
                    self.dz_lim[dz_imax]
                    - (self.dz_ck0[dz_imax, -1] + dz_dtmax * self.dz_ck1[dz_imax, -1])
                    * np.exp(-dz_dtmax),
                )

                # print(
                #     f"New potential derivative constraint added at time {self.dz_start[dz_imax] + dz_dtmax}!"
                # )
            # return 0

    def learn(
        self,
        feas_tol: float = 1e-6,
        conv_tol: float = 1e-9,
        n_iter: int = 1000,
    ) -> int:
        """Learn the synaptic weights to produce the desired firing times when fed with the prescribed input spikes.

        Args:
            f_times (List[float]): the desired firing times.
            inspikes (List[InSpike]): the collection of input spikes used to generate the firing times. The input spikes have an input ID attribute that is used to identify the corresponding synaptic weight.

        Returns:
            (int): the status of the optimization:
                - 1 if the optimization converged to a solution,
                - 0 if the optimization failed to converge,
                - -1 if the optimization failed to find a feasible solution.
        """

        # # inspike.conn_id and inspike.time
        # conn_ids = list(set(inspike.conn_id for inspike in inspikes))
        # conn_ids.sort()

        # n_in_channels = np.max(in_channels, initial=-1) + 1

        # Init in_coef and ends, which remain constant during the optimization
        # Contains both information for potential and derivatives

        # # Add constraints at firing times
        # solver = Solver(n_in_channels)

        # in_coef, starts, durations, bounds = [], [], [], []
        # for n in range(f_times.size):
        #     # f_time = (f_times[n] - f_times[n - 1]) % period
        #     # bf_time = f_time - eps
        #     in_coef_n, starts_n, durations_n, bounds_n = compute_in_states(
        #         (f_times[n] - f_times[n - 1]) % period,
        #         (f_times[n] - eps - f_times[n - 1]) % period,
        #         (in_times - f_times[n - 1]) % period,
        #         in_channels,
        #         n_in_channels,
        #     )
        #     in_coef.append(in_coef_n[:, :-1])
        #     starts.append(starts_n[:-1] + f_times[n - 1])
        #     durations.append(durations_n[:-1])
        #     bounds.append(bounds_n[:-1])

        #     # Add the firing time constraint a_n x <= b_n (equivalent to -a_n x >= -b_n)
        #     solver.add_constraint(in_coef_n[0, -1], bounds_n[-1])

        # in_coef = np.concatenate(in_coef, axis=1)
        # starts = np.concatenate(starts)
        # durations = np.concatenate(durations)
        # bounds = np.concatenate(bounds)

        # if self.solver is None:
        #     raise ValueError("Solver is not initialized. Call init_solver() first.")

        res_refine = self.refine_constraints(feas_tol)
        if res_refine == 1:
            print(f"Constraints refined, optimal solution found in 0 iteration.")
            self.weight = np.copy(self.solver.x)
            return 1

        for i in range(n_iter):
            # 2. Refine constraints based on the current primal solution. If no constraints are violated, then the primal solution is optimal.
            res_refine = self.refine_constraints(feas_tol)
            if res_refine == 1:
                print(
                    f"Constraints refined, optimal solution found in {i+1} iterations."
                )
                self.weight = np.copy(self.solver.x)
                return 1
            
            if res_refine == -1:
                print(
                    f"Constraints refinement failed in iteration {i+1}, no feasible solution found."
                )
                return -1

            # 1. DCD algorithm: repeat the following steps until convergence (of the primal cost) or n_iter reached:
            # 1.1 dual coordinate descent step in the dual space
            # 1.2 convert the dual vector to a primal vector
            # 1.3 compute the cost of the primal vector
            res_dcd = self.solver.dual_coordinate_descent(feas_tol, conv_tol, n_iter)
            if res_dcd < 1:
                return res_dcd

        print("Maximum iterations reached without convergence.")
        return 0

    # def init_states(
    #     f_time: float,
    #     bf_time: float,
    #     in_times: np.ndarray,
    #     in_channels: np.ndarray,
    #     n_in_channels: int,
    #     zmax: float = 0.0,
    #     dzmin: float = 1e-6,
    # ) -> Tuple[
    #     np.ndarray,  # in_coef: shape (2, n_intervals, n_channels)
    #     np.ndarray,  # starts: shape (n_intervals)
    #     np.ndarray,  # lengths: shape (n_intervals)
    #     np.ndarray,  # bounds: shape (n_intervals)
    # ]:
    #     """
    #     Compute the coefficients (c0nk and c1nk) defining the states of every input (indexed by k) for any time between 0 and f_time, on disjoint intervals (indexed by n).
    #     The intervals partition the time range [0, f_time] in n_intervals = in_times.size + 3 intervals from the following time markers:
    #     - 0.0, the start of the time range
    #     - bf_time, the time before firing (the beginning of the active region)
    #     - f_time, the firing time
    #     - in_times, the input spike times.
    #     The intervals are reconstructed from their start and length.
    #     The signal (c0nk + c1nk * dt) * exp(-dt) for 0 <= dt < length[n] then corresponds to
    #     a) the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] + length[n] < bf_time.
    #     b) the derivative of the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] + length[n] < f_time and start[n] >= f_time.
    #     c) the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] = f_time.

    #     Args:
    #         f_time (np.double): _description_
    #         bf_time (np.double): _description_
    #         in_times (np.ndarray): _description_
    #         in_channels (np.ndarray): _description_
    #         n_in_channels (np.intp): _description_
    #         zmax (np.double): _description_
    #         dzmin (np.double): _description_

    #     Returns:
    #         np.ndarray: the coefficients defining the input signals by parts with shape (2, n_intervals, n_channels)
    #         np.ndarray: the times at which the intervals start with shape (n_intervals)
    #         np.ndarray: the lengths the intervals, with shape (n_intervals)
    #         np.ndarray: the template bounds for each interval, with shape (n_intervals)
    #     """

    #     # Extract the in_times and in_channels that are valid, i.e., within the range [0, f_time)
    #     valid = (in_times >= 0.0) & (in_times < f_time)
    #     in_times = in_times[valid]
    #     in_channels = in_channels[valid]

    #     # Initialize the starts array
    #     starts = np.concatenate((in_times, np.array([0.0, f_time, bf_time])))

    #     # Initialize the coefficients array
    #     in_coef = np.zeros((2, starts.size, n_in_channels))
    #     in_coef[1, np.arange(in_times.size), in_channels] = 1.0

    #     # Sort the coefficients according to their starts
    #     sort_ids = np.argsort(starts)
    #     starts = starts[sort_ids]
    #     lengths = np.diff(
    #         starts, append=f_time
    #     )  # time differences = lengths of the intervals
    #     # ends = starts + lengths
    #     bounds = np.full(starts.shape, zmax)

    #     in_coef = in_coef[:, sort_ids, :]
    #     for n in range(starts.size - 1):
    #         in_coef[1, n + 1] += in_coef[1, n] * np.exp(-lengths[n])
    #         in_coef[0, n + 1] += (in_coef[0, n] + in_coef[1, n] * lengths[n]) * np.exp(
    #             -lengths[n]
    #         )

    #     # The signal of interest on the active region is the (negative) derivative
    #     active = (starts >= bf_time) & (starts < f_time)
    #     in_coef[0, active] -= in_coef[1, active]  # (negative) derivative
    #     bounds[active] = -dzmin

    #     # The signal of interest for the firing time is the potential
    #     in_coef[:, -1] = -in_coef[:, -1]  # (negative) derivative
    #     bounds[-1] = -FIRING_THRESHOLD

    #     return in_coef, starts, lengths, bounds
