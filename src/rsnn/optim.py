from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from rsnn.constants import *


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
) -> Tuple[np.intp, np.float64, np.float64]:
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
        self.n_cstrs = 0  # Number of constraints (= number of dual variables)

        self.x = np.zeros(self.n_vars, dtype=np.float64)
        self.xt = np.zeros(self.n_cstrs, dtype=np.float64)
        self.cost = 0.0

        self.A = np.empty((0, n_vars))
        self.b = np.empty((0,))

        self.xibxt_N = np.empty((0,))
        self.Wbxt_N = np.empty((0, 0), dtype=np.float64)

        self.mfut = np.zeros((self.n_cstrs,))
        self.Vfut = np.zeros((self.n_cstrs,))
        self.beta = np.full((self.n_cstrs,), 1e-12)

    def solve(self, feas_tol: float=1e-6, conv_tol: float=1e-6, n_iter: int=1000, order:int | np.intp=2) -> int:
        """
        Solve the optimization problem defined by the constraints in the dual space.

        Args:
            feas_tol (float): Tolerance for feasibility.
            conv_tol (float): Tolerance for convergence.
            n_iter (int): Maximum number of iterations to run.

        Returns:
            int: The status of the optimization:
                - 1 if the optimization converged to a solution,
                - 0 if the optimization failed to converge,
                - -1 if the optimization failed to find a feasible solution.
        """

        if self.n_cstrs > 0:
            if order == 2:
                return self.dbffd(feas_tol, conv_tol, n_iter)
            elif order == 1:
                return self.dpcd(feas_tol, conv_tol, n_iter)
            else:
                raise ValueError(f"Unsupported order {order}. Supported orders are 1 (=projected dual coordinate descent) and 2 (=backward filtering forward deciding).")
            
        else:
            self.cost = 0.0
            self.x = np.zeros(self.n_vars, dtype=np.float64)
            print("Feasible (trivial) solution found.")
            return 1

    def dpcd(self, feas_tol: float=1e-6, conv_tol: float=1e-6, n_iter: int=1000) -> int:
        """
        Dual projected coordinate descent algorithm.
        This updates the dual solution vector xt based on the current constraints.

        Returns:
            (int): the status of the optimization:
                - 1 if the optimization converged to a solution,
                - 0 if the optimization failed to converge,
                - -1 if the optimization failed to find a feasible solution.

        """
        en_xibxtn = np.empty((self.n_cstrs, 1))

        xibxt = np.empty((self.n_cstrs,))

        cost_buf = np.full((2,), np.inf)
        
        for i in range(n_iter):
            # Backward Filtering
            np.copyto(xibxt, self.xibxt_N)
            for n in range(self.n_cstrs - 1, -1, -1):
                np.copyto(en_xibxtn[n], xibxt[n])
                xibxt -= self.mfut[n] * self.Wbxt_N[n]
            
            # Forward Deciding
            self.xt.fill(0.0)
            for n in range(self.n_cstrs):
                Vbut = 1 / self.Wbxt_N[n, n]
                mbut = (
                    Vbut
                    * (
                        en_xibxtn[n] - np.inner(self.Wbxt_N[n], self.xt)
                    ).squeeze()
                )
                utn = np.clip(mbut, 0.0, None)  # cf. Table 2 in LiLoeliger2024
                self.xt[n] += utn
                self.mfut[n] = np.abs(utn)  # cf. Table 1 in LiLoeliger2024

            self.x = -self.xt @ self.A
            self.cost = np.linalg.norm(self.x).astype(np.float64)
            
            if np.any(np.abs(self.cost - cost_buf) < conv_tol):
                print(f"Converged after {i+1} iterations")
                max_violation = self.max_violation()
                if max_violation <= feas_tol:
                    print(f"Feasible solution found with cost {self.cost}.")
                    return 1
                else:
                    self.cost = np.inf
                    print(f"Primal problem is infeasible, maximum violation is {max_violation}.")
                    return -1

            cost_buf = np.append(cost_buf[1:], self.cost)

        print("Reached maximum number of coordinate descents.")
        return 0

    def dbffd(self, feas_tol: float=1e-6, conv_tol: float=1e-6, n_iter: int=1000) -> int:
        """
        Run backward-filtering forward-deciding in the dual space to optimize the dual solution.
        This updates the dual solution vector xt based on the current constraints.

        Returns:
            (int): the status of the optimization:
                - 1 if the optimization converged to a solution,
                - 0 if the optimization failed to converge,
                - -1 if the optimization failed to find a feasible solution.

        """
        en_xibxtn = np.empty((self.n_cstrs, 1))
        Wbxtn_en = np.empty((self.n_cstrs, self.n_cstrs))

        xibxt = np.empty((self.n_cstrs,))
        Wbxt = np.empty((self.n_cstrs, self.n_cstrs))

        cost_buf = np.full((2,), np.inf)
        
        for i in range(n_iter):
            # print(f"Iteration {i+1}")
            # Backward Filtering
            np.copyto(xibxt, self.xibxt_N)
            np.copyto(Wbxt, self.Wbxt_N)
            for n in range(self.n_cstrs - 1, -1, -1):
                np.copyto(en_xibxtn[n], xibxt[n])
                np.copyto(Wbxtn_en[n], Wbxt[n])
                H = np.divide(
                    self.Vfut[n], 1 + self.Vfut[n] * Wbxt[n, n]
                ).squeeze()
                h = np.divide(
                    self.mfut[n] + self.Vfut[n] * xibxt[n],
                    1 + self.Vfut[n] * Wbxt[n, n],
                ).squeeze()
                xibxt -= h * Wbxt[n]
                Wbxt -= H * np.outer(Wbxt[n], Wbxt[n])
            
            # print(f"\txibxt = {xibxt}")

            # Forward Deciding
            self.xt.fill(0.0)
            for n in range(self.n_cstrs):
                Vbut = 1 / Wbxtn_en[n, n]
                mbut = (
                    Vbut
                    * (
                        en_xibxtn[n] - np.inner(Wbxtn_en[n], self.xt)
                    ).squeeze()
                )
                self.beta[n] = np.maximum(
                    self.beta[n], -mbut / Vbut
                )  # enforce utn >= 0
                utn = np.clip(mbut, 0.0, None)  # cf. Table 2 in LiLoeliger2024
                self.xt[n] += utn
                self.mfut[n] = np.abs(utn)  # cf. Table 1 in LiLoeliger2024
                self.Vfut[n] = (
                    2 * self.mfut[n] / self.beta[n]
                )  # cf. Table 1 in LiLoeliger2024


            self.x = -self.xt @ self.A
            self.cost = np.linalg.norm(self.x).astype(np.float64)

            if np.any(np.abs(self.cost - cost_buf) < conv_tol):
                print(f"Converged after {i+1} iterations")
                max_violation = self.max_violation()
                if max_violation <= feas_tol:
                    print(f"Feasible solution found with cost {self.cost}.")
                    return 1
                else:
                    self.cost = np.inf
                    print(f"Primal problem is infeasible, maximum violation is {max_violation}.")
                    return -1

            cost_buf = np.append(cost_buf[1:], self.cost)

        print("Reached maximum number of coordinate descents.")
        return 0

    def max_violation(self) -> float:
        """
        Check if the current solution is valid, i.e., if all constraints are satisfied.

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        return np.max(np.clip(self.A @ self.x - self.b, 0.0, None))

    def add_constraint(
        self,
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> int:
        a_norm = np.linalg.norm(a)

        if a_norm > 0.0:  # a is not the zero vector
            self.n_cstrs += 1

            # self.xt = np.append(self.xt, 0.0)
            self.xt = np.zeros(self.n_cstrs, dtype=np.float64)

            a /= a_norm
            b /= a_norm

            Aat = (self.A @ a).reshape(-1, 1)

            # self.Vfxt = np.block([[self.Vfxt, Aat], [Aat.T, 1.0]])
            # self.mfxt = np.append(self.mfxt, -b)

            self.A = np.vstack((self.A, a))
            self.b = np.append(self.b, b)

            self.xibxt_N = np.append(self.xibxt_N, -b)
            self.Wbxt_N = np.block([[self.Wbxt_N, Aat], [Aat.T, 1.0]])

            # self.en_xibxtn = np.empty((self.n_cstrs, 1))
            # self.Wbxtn_en = np.empty((self.n_cstrs, self.n_cstrs))
            
            self.mfut = np.append(self.mfut, 0.0)
            self.Vfut = np.append(self.Vfut, 0.0)
            self.beta = np.append(self.beta, 1e-12)

            return 0
        elif b >= 0.0:  # if b >= 0 then ax = 0 <= b for every x
            return 0
        else:  # if b < 0 then a x = 0 > b for every x
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
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    z_ck1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
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
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    dz_ck1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
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
        self.solver.__init__(n_in_channels)
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
                res = self.solver.add_constraint(
                    # zi_f_c_n[:-1], z_f_lim_n - zi_f_c_n[-1]
                    -ck0[-1, :-1],
                    -f_thresh + ck0[-1, -1],
                )
                if res < 0:
                    raise ValueError(
                        f"The firing time constraint at time {start[-1]} is not feasible."
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
            z_dtmax = 0.0

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
            dz_dtmax = 0.0

        if z_vmax <= feas_tol and dz_vmax <= feas_tol:
            return 1
        else:
            if z_vmax > dz_vmax and np.isfinite(z_dtmax):
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
        feas_tol: float = 1e-3,
        conv_tol: float = 1e-6,
        n_cstrs: int = 1000,
        n_iter: int = 10000,
        order: int = 2,
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

        for i in range(n_cstrs):
            # 1. DCD algorithm: repeat the following steps until convergence (of the primal cost) or n_iter reached:
            # 1.1 dual coordinate descent step in the dual space
            # 1.2 convert the dual vector to a primal vector
            # 1.3 compute the cost of the primal vector
            res = self.solver.solve(
                feas_tol, conv_tol, n_iter, order=order
            )
            if res < 1:
                return res

            # 2. Refine constraints based on the current primal solution. If no constraints are violated, then the primal solution is optimal.
            res_refine = self.refine_constraints(feas_tol)
            if res_refine == 0:
                print(f"Iteration {i+1}: {self.solver.n_cstrs} constraints.")
            elif res_refine == 1:
                print(
                    f"Solved in {i+1} iterations! Cost is {self.solver.cost:.3f} for {self.solver.n_cstrs} constraints."
                )
                self.weight = np.copy(self.solver.x)
                return 1
            else:
                print(
                    f"Constraints refinement failed in iteration {i+1}, problem is infeasible."
                )
                return -1

        print("Maximum number of constraint refinements reached without convergence.")
        return 0
