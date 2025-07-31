import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
import numpy.typing as npt

from rsnn.constants import *
from rsnn.learning.gmp import GMPModel
from rsnn.learning.grb import GRBModel
from rsnn.learning.utils import (compute_states, find_maximum_violation,
                                 modulo_with_offset)
from rsnn.utils import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")


@dataclass
class Neuron:
    """
    A class representing a neuron with its input channels.
    """

    # def __init__(self, n_in_channels: int):
    #     """
    #     Initialize the neuron with a specified number of input channels.

    #     Args:
    #         n_in_channels (int | np.intp): Number of input channels.
    #     """
    #     if n_in_channels < 0:
    #         raise ValueError(
    #             f"n_in_channels must be non-negative, got {n_in_channels}."
    #         )
    #     self.n_in_channels = n_in_channels

    #     # Synaptic weights for each input channel
    #     self.weight = np.zeros(n_in_channels, dtype=np.float64)

    #     # Potential state variables
    #     self.z_start = np.empty((0,), dtype=np.float64)
    #     self.z_length = np.empty((0,), dtype=np.float64)
    #     self.z_lim = np.empty((0,), dtype=np.float64)
    #     self.z_c0 = np.empty((0,), dtype=np.float64)
    #     self.z_c1 = np.empty((0,), dtype=np.float64)
    #     self.z_c0 = np.empty((0, n_in_channels + 1), dtype=np.float64)
    #     self.z_c1 = np.empty((0, n_in_channels + 1), dtype=np.float64)

    #     # Derivative of the potential state variables
    #     self.dz_start = np.empty((0,), dtype=np.float64)
    #     self.dz_length = np.empty((0,), dtype=np.float64)
    #     self.dz_lim = np.empty((0,), dtype=np.float64)
    #     self.dz_c0 = np.empty((0,), dtype=np.float64)
    #     self.dz_c1 = np.empty((0,), dtype=np.float64)
    #     self.dz_c0 = np.empty((0, n_in_channels + 1), dtype=np.float64)
    #     self.dz_c1 = np.empty((0, n_in_channels + 1), dtype=np.float64)

    #     # QProgram
    #     self.solver = QProgram(n_in_channels)

    @property
    def n_intervals(self) -> int:
        """Number of intervals partitioning the time axis"""
        return len(self.z_start)

    # @property
    # def n_dz_intervals(self) -> int:
    #     """Number of intervals partitioning the time axis for the potential derivative."""
    #     return self.dz_start.size

    def __init__(
        self,
        id: int,
        solver: GMPModel | GRBModel,
        # n_in_channels: int,
        f_times: npt.NDArray[np.float64],
        in_times: npt.NDArray[np.float64],
        in_channels: npt.NDArray[np.int64],
        period: np.float64 | float,
        eps: np.float64 | float = 0.2 * REFRACTORY_PERIOD,
        f_thresh: np.float64 | float = FIRING_THRESHOLD,
        zmax: np.float64 | float = 0.0,
        dzmin: np.float64 | float = 1e-2,
    ):
        """
        Initialize the neuron for learning.

        Args:
            n_in_channels (int | np.intp): Number of input channels.
        """
        self.id = id

        # Initialize the quadratic programming solver
        self.solver = solver

        if np.max(in_channels) > self.solver.n_vars - 1 or np.min(in_channels) < 0:
            raise ValueError("in_channels contains invalid channel indices.")

        if zmax >= f_thresh:
            raise ValueError(
                f"zmax must be less than f_thresh, got zmax={zmax} and f_thresh={f_thresh}."
            )

        if dzmin <= 0.0:
            raise ValueError(
                f"dzmin must be positive, got {dzmin}. It is used to ensure the potential derivative is non-negative."
            )

        # Synaptic weights for each input channel
        self.weight = np.full(self.solver.n_vars, np.nan, dtype=np.float64)

        self.z_start = []
        self.z_length = []
        self.z_lim = []
        self.z_c0_in = []
        self.z_c1_in = []
        self.z_c0 = []
        self.z_c1 = []

        self.dz_start = []
        self.dz_length = []
        self.dz_lim = []
        self.dz_c0_in = []
        self.dz_c1_in = []
        self.dz_c0 = []
        self.dz_c1 = []

        # Initialize the neuron states based on the provided firing times and input times
        if f_times.size > 1:
            f_times = np.sort(f_times)

            A_f_times = []
            b_f_times = []

            for n in range(f_times.size):
                tmin = f_times[(n - 1) % f_times.size]  # Previous firing time
                tmax = modulo_with_offset(
                    f_times[n], period, tmin
                ).item()  # Firing time
                in_times = modulo_with_offset(
                    in_times, period, tmin
                )  # All input times arrive after the previous firing time
                c0_tmin = np.array([0.0] * self.solver.n_vars + [1.0])
                c1_tmin = np.zeros(self.solver.n_vars + 1)
                start, c0_in, c1_in = compute_states(
                    in_times,
                    in_channels,
                    c0_tmin,
                    c1_tmin,
                    tmin,
                    tmax,
                    np.array([tmax - eps]),
                )

                logger.debug(
                    f"The time interval [{tmin, tmax}] starting has been partitioned into {start.size} intervals."
                )

                # Firing time constraint | z >= f_thresh
                A_f_times.append(
                    -(c0_in[-1, :-1] + c1_in[-1, :-1] * (tmax - start[-1]))
                    * np.exp(-(tmax - start[-1]))
                )
                b_f_times.append(
                    -f_thresh
                    + REFRACTORY_RESET
                    * (c0_in[-1, -1] + c1_in[-1, -1] * (tmax - start[-1]))
                    * np.exp(-(tmax - start[-1]))
                )

                ## Silent time intervals | z <= zmax
                logger.debug(f"Silent zone is [{tmin}, {tmax - eps}].")
                silent = start < tmax - eps
                # last = np.searchsorted(start, tmax - eps, side='right')
                self.z_start.append(start[silent])
                self.z_length.append(np.diff(start[silent], append=tmax - eps))
                self.z_lim.append(np.full_like(start[silent], zmax))
                self.z_c0_in.append(c0_in[silent])
                self.z_c1_in.append(c1_in[silent])
                self.z_c0.append(np.full_like(start[silent], np.nan))
                self.z_c1.append(np.full_like(start[silent], np.nan))

                ## Active time intervals | dz >= dzmin
                logger.debug(f"Active zone is [{tmax - eps}, {tmax}].")
                active = start >= tmax - eps
                # first = (np.searchsorted(start, tmax - eps, side='right') - 1).clip(0, None)
                self.dz_start.append(start[active])
                self.dz_length.append(np.diff(start[active], append=tmax))
                self.dz_lim.append(np.full_like(start[active], -dzmin))
                self.dz_c0_in.append(c0_in[active] - c1_in[active])
                self.dz_c1_in.append(c1_in[active])
                self.dz_c0.append(np.full_like(start[active], np.nan))
                self.dz_c1.append(np.full_like(start[active], np.nan))

            # Add the firing time constraints to the solver
            self.solver.add_constraints(np.vstack(A_f_times), np.array(b_f_times))

        elif f_times.size == 1:
            tmin = f_times.item() - period  # Previous firing time
            tmax = f_times.item()  # Firing time
            in_times = modulo_with_offset(
                in_times, period, tmin
            )  # All input times arrive after the previous firing time
            c0_tmin = np.array([0.0] * self.solver.n_vars + [1.0])
            c1_tmin = np.zeros(self.solver.n_vars + 1)
            start, c0_in, c1_in = compute_states(
                in_times,
                in_channels,
                c0_tmin,
                c1_tmin,
                tmin,
                tmax,
                np.array([tmax - eps]),
            )

            logger.debug(
                f"The time interval [{tmin, tmax}] starting has been partitioned into {start.size} intervals."
            )

            # Firing time constraint | z >= f_thresh
            A_f_times = -(
                c0_in[-1, :-1] + c1_in[-1, :-1] * (tmax - start[-1])
            ) * np.exp(-(tmax - start[-1]))
            b_f_times = -f_thresh + REFRACTORY_RESET * (
                c0_in[-1, -1] + c1_in[-1, -1] * (tmax - start[-1])
            ) * np.exp(-(tmax - start[-1]))

            ## Silent time intervals | z <= zmax
            logger.debug(f"Silent zone is [{tmin}, {tmax - eps}].")
            silent = start < tmax - eps
            # last = np.searchsorted(start, tmax - eps, side='right')
            self.z_start.append(start[silent])
            self.z_length.append(np.diff(start[silent], append=tmax - eps))
            self.z_lim.append(np.full_like(start[silent], zmax))
            self.z_c0_in.append(c0_in[silent])
            self.z_c1_in.append(c1_in[silent])
            self.z_c0.append(np.full_like(start[silent], np.nan))
            self.z_c1.append(np.full_like(start[silent], np.nan))

            ## Active time intervals | dz >= dzmin
            logger.debug(f"Active zone is [{tmax - eps}, {tmax}].")
            active = start >= tmax - eps
            # first = (np.searchsorted(start, tmax - eps, side='right') - 1).clip(0, None)
            self.dz_start.append(start[active])
            self.dz_length.append(np.diff(start[active], append=tmax))
            self.dz_lim.append(np.full_like(start[active], -dzmin))
            self.dz_c0_in.append(c0_in[active] - c1_in[active])
            self.dz_c1_in.append(c1_in[active])
            self.dz_c0.append(np.full_like(start[active], np.nan))
            self.dz_c1.append(np.full_like(start[active], np.nan))

            # Add the firing time constraints to the solver
            self.solver.add_constraints(A_f_times, b_f_times)

        else:
            tmin, tmax = 0.0, period
            in_times = modulo_with_offset(
                in_times, period, tmin
            )  # All input times arrive after the previous firing time
            c0_tmin = np.zeros(self.solver.n_vars + 1)
            c1_tmin = np.zeros(self.solver.n_vars + 1)
            start, c0_in, c1_in = compute_states(
                in_times, in_channels, c0_tmin, c1_tmin, tmin, tmax
            )

            logger.debug(
                f"The time interval [{tmin, tmax}] starting has been partitioned into {start.size} intervals."
            )

            ## Silent time intervals | z <= zmax
            logger.debug(f"Silent zone is [{tmin}, {tmax}].")
            self.z_start.append(start)
            self.z_length.append(np.diff(start, append=tmax))
            self.z_lim.append(np.full_like(start, zmax))
            self.z_c0_in.append(c0_in)
            self.z_c1_in.append(c1_in)
            self.z_c0.append(np.full_like(start, np.nan))
            self.z_c1.append(np.full_like(start, np.nan))

            ## No potential derivative constraints...
            self.dz_start.append(np.empty((0,), dtype=np.float64))
            self.dz_length.append(np.empty((0,), dtype=np.float64))
            self.dz_lim.append(np.empty((0,), dtype=np.float64))
            self.dz_c0_in.append(
                np.empty((0, self.solver.n_vars + 1), dtype=np.float64)
            )
            self.dz_c1_in.append(
                np.empty((0, self.solver.n_vars + 1), dtype=np.float64)
            )
            self.dz_c0.append(np.empty((0,), dtype=np.float64))
            self.dz_c1.append(np.empty((0,), dtype=np.float64))

        logger.debug(
            f"Neuron initialized successfully! The solver has been initialized with {self.solver.n_cstrs} constraints for {self.solver.n_vars} variables."
        )

    def refine_constraints(self) -> int:
        """
        Refine the constraints based on the current optimal weights.

        Args:
            ftol (float, optional): minimum value for a constraint to be considered violated. Defaults to 1e-6.
            n_cstrs_per_template (int, optional): number of constraints to add for each template. Defaults to 1.

        Returns:
            (int): the status of the constraint refinement:
                - 1 if all constraints are satisfied,
                - 0 if a new constraint was added,
                - -1 if the refinement failed.
        """
        A, b = [], []

        self.weight = self.solver.x_value()

        for n in range(self.n_intervals):
            # Potential constraints
            self.z_c0[n] = (
                np.inner(self.z_c0_in[n][:, :-1], self.weight)
                + self.z_c0_in[n][:, -1] * REFRACTORY_RESET
            )
            self.z_c1[n] = (
                np.inner(self.z_c1_in[n][:, :-1], self.weight)
                + self.z_c1_in[n][:, -1] * REFRACTORY_RESET
            )
            z_dvmax, z_imax, z_dtmax = find_maximum_violation(
                self.z_c0[n], self.z_c1[n], self.z_length[n], self.z_lim[n] + FEAS_TOL
            ) or (0.0, None, None)

            if z_dvmax > 0.0 and z_imax is not None and z_dtmax is not None:
                if np.isfinite(z_dtmax):
                    A.append(
                        (
                            self.z_c0_in[n][z_imax, :-1]
                            + z_dtmax * self.z_c1_in[n][z_imax, :-1]
                        )
                        * np.exp(-z_dtmax)
                    )
                    b.append(
                        self.z_lim[n][z_imax]
                        - (
                            self.z_c0_in[n][z_imax, -1]
                            + z_dtmax * self.z_c1_in[n][z_imax, -1]
                        )
                        * np.exp(-z_dtmax)
                    )
                else:
                    A.append(np.zeros(self.solver.n_vars))
                    b.append(self.z_lim[n][z_imax])

                logger.debug(
                    f"Constraint refinement: a violation of the potential constraint has been detected on the {n}th interval at t={self.z_start[n][z_imax] + z_dtmax}."
                )

            # Potential derivative constraints
            self.dz_c0[n] = (
                np.inner(self.dz_c0_in[n][:, :-1], self.weight)
                + self.dz_c0_in[n][:, -1] * REFRACTORY_RESET
            )
            self.dz_c1[n] = (
                np.inner(self.dz_c1_in[n][:, :-1], self.weight)
                + self.dz_c1_in[n][:, -1] * REFRACTORY_RESET
            )
            dz_dvmax, dz_imax, dz_dtmax = find_maximum_violation(
                self.dz_c0[n],
                self.dz_c1[n],
                self.dz_length[n],
                self.dz_lim[n] + FEAS_TOL,
            ) or (0.0, None, None)

            if dz_dvmax > 0.0 and dz_imax is not None and dz_dtmax is not None:
                logger.debug(f"{dz_imax=}")
                if np.isfinite(dz_dtmax):
                    A.append(
                        (
                            self.dz_c0_in[n][dz_imax, :-1]
                            + dz_dtmax * self.dz_c1_in[n][dz_imax, :-1]
                        )
                        * np.exp(-dz_dtmax)
                    )
                    b.append(
                        self.dz_lim[n][dz_imax]
                        - (
                            self.dz_c0_in[n][dz_imax, -1]
                            + dz_dtmax * self.dz_c1_in[n][dz_imax, -1]
                        )
                        * np.exp(-dz_dtmax)
                    )
                else:
                    A.append(np.zeros(self.solver.n_vars))
                    b.append(self.dz_lim[n][dz_imax])

                logger.debug(
                    f"Constraint refinement: a violation of the potential derivative constraint has been detected on the {n}th interval at t={self.dz_start[n][dz_imax] + dz_dtmax}."
                )

        if not b:
            logger.debug("Constraint refinement: all constraints are satisfied!")
            return 1

        # Add the new constraints to the solver
        self.solver.add_constraints(np.vstack(A), np.array(b))
        return 0

    def learn(
        self,
        max_cstrs: int = 1000,
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

        for i in range(max_cstrs):
            res = self.solver.solve()
            if res < 1:
                return res

            # 2. Refine constraints based on the current primal solution.
            res_refine = self.refine_constraints()
            if (
                res_refine > 0
            ):  # If no constraints are violated, then the primal solution is optimal.
                logger.info(
                    f"Learning succeeds! Solved in {i+1} refinement iterations. Number of variables: {self.solver.n_vars}. Number of constraints: {self.solver.n_cstrs}. Cost: {self.solver.cost_value():.6f}."
                )
                # self.weight = np.copy(self.solver.model.x.X)
                return 1

        logger.warning(
            "Neuron optimization did not converge within the maximum number of iterations."
        )
        return 0
