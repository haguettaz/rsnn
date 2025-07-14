import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
import numpy.typing as npt

from rsnn.constants import *

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d' - %(levelname)s - %(message)s",
    style="%",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel("DEBUG")
logger.addHandler(console_handler)

file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel("INFO")
logger.addHandler(file_handler)


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

    # logger.debug(
    #     f"The time interval [{start[0]}, {start[-1] + length[-1]}) has been partitioned into {start.size} intervals."
    # )

    return start, length, ck0, ck1


def find_maximum_violation(
    c0: npt.NDArray[np.float64],
    c1: npt.NDArray[np.float64],
    length: npt.NDArray[np.float64],
    lim: npt.NDArray[np.float64],
) -> Optional[Tuple[np.float64, np.intp, np.float64]]:
    """
    Compute the maximum violation of the condition c0[n] + c1[n] * dt * exp(-dt) > lim[n] for 0 <= dt < length[n] for each interval n, if any.

    Args:
        c0 (npt.NDArray[np.float64]): the 0th coefficients for each interval.
        c1 (npt.NDArray[np.float64]): the 1st coefficients for each interval.
        length (npt.NDArray[np.float64]): the length for each interval.
        lim (npt.NDArray[np.float64]): the maximum allowed value for each interval.

    Returns:
        Tuple[np.float64, np.intp, np.float64]: the maximum violation and the interval index nmax and the time difference dt in [0, length_nmax] at which the violation occurs.
    """
    dt = np.vstack(
        [np.zeros_like(c0), np.clip(1 - c0 / c1, 0.0, length), length]
    )  # shape (3, n_intervals)
    dv = np.clip(
        np.nan_to_num(c0[np.newaxis, :] + c1[np.newaxis, :] * dt) * np.exp(-dt)
        - lim[np.newaxis, :],
        0.0,
        None,
    )  # shape (3, n_intervals)
    imax = np.unravel_index(
        np.argmax(dv), dv.shape
    )  # index tuple of the maximum value in v

    if dv[imax] > 0.0:
        # logger.debug(
        #     f"Maximum violation (with value {dv[imax]}) found for the {imax[0]}th interval at dt = {dt[imax]}."
        # )
        return dv[imax], imax[1], dt[imax]

    # logger.debug("No violation found.")
    return None


class QProgram:
    def __init__(
        self,
        n_vars: int,
        mxf: Optional[npt.NDArray[np.float64]] = None,
        Vxf: Optional[npt.NDArray[np.float64]] = None,
        beta_init: float | np.float64 = 1e-15,  # initial value for beta
    ):
        """
        Initialize a solver for quadratic optimization problems with linear constraints.

        Args:
            n_vars (int | np.intp): Number of variables in the optimization problem.
            mxf (Optional[npt.NDArray[np.float64]]): Mean vector for the primal cost. If None, defaults to a zero vector.
            Vxf (Optional[npt.NDArray[np.float64]]): Covariance matrix for the primal cost. If None, defaults to an identity matrix.
            beta_init (float | np.float64): Initial value for the beta parameter used in the optimization. Defaults to 1e-12.

        Raises:
            ValueError: If the shapes of `mxf` and `Vxf` do not match the expected dimensions.
        """
        if mxf is not None and mxf.shape != (n_vars,):
            raise ValueError(
                f"If provided, mxf must have shape ({n_vars},) but has shape {mxf.shape}."
            )
        if Vxf is not None and Vxf.shape != (n_vars, n_vars):
            raise ValueError(
                f"If provided, Vxf must have shape ({n_vars},{n_vars}) but has shape {Vxf.shape}."
            )

        self.n_vars = n_vars  # Number of variables
        self.n_cstrs = 0  # Number of constraints (= number of dual variables)

        self.Vxf = Vxf
        self.Wxf = None if self.Vxf is None else np.linalg.inv(self.Vxf)
        self.mxf = mxf
        self.xixf = (
            None
            if self.mxf is None
            else self.mxf if self.Wxf is None else self.Wxf @ self.mxf
        )

        self.A = np.empty((self.n_cstrs, n_vars))
        self.b = np.empty((self.n_cstrs,))

        self.x = np.zeros(self.n_vars, dtype=np.float64)
        self.xt = np.zeros(self.n_cstrs, dtype=np.float64)

        self.xibxt_N = np.empty((self.n_cstrs,))
        self.Wbxt_N = np.empty((self.n_cstrs, self.n_cstrs), dtype=np.float64)

        self.mfut = np.zeros((self.n_cstrs,))
        self.Vfut = np.zeros((self.n_cstrs,))

        self.beta_init = beta_init
        self.beta = np.full((self.n_cstrs,), self.beta_init)

        self.is_feas = True  # whether the problem is feasible
        self.dual_cost = 0.0
        self.primal_cost = 0.0  # without any constraints, the cost is 0.0
        self.duality_gap = self.dual_cost - self.primal_cost
        # self.dual_residual = np.zeros(self.n_vars, dtype=np.float64)
        self.primal_residual = np.zeros(self.n_cstrs, dtype=np.float64)

        # Initialize a Gurobi model for feasibility checking
        self.gb_model = gp.Model("qp_model")
        self.gb_model.setParam("OutputFlag", 0)  # Suppress Gurobi output
        self.gb_x = self.gb_model.addMVar(self.n_vars, lb=-np.inf, ub=np.inf, name="x")

        logger.info(
            f"Quadratic program initialized with {self.n_vars} variables (and {self.n_cstrs} constraints)."
        )

    def add_constraints(
        self,
        A: npt.NDArray[np.float64],  # shape (n_cstrs, n_vars) or (n_vars, )
        b: npt.NDArray[np.float64],  # shape (n_cstrs, 1) or (n_cstrs, )
        # check_feas:bool = True,  # whether to check feasibility before adding constraints
    ):
        """
        Add linear inequality constraints to the optimization problem.

        The constraints are of the form A @ x <= b, where each row represents one constraint.
        The method validates constraint dimensions, checks feasibility using Gurobi, and updates
        internal matrices and vectors accordingly.

        Args:
            A (npt.NDArray[np.float64]): Constraint coefficient matrix with shape (n_new_constraints, n_vars)
            or (n_vars,) for a single constraint.
            b (npt.NDArray[np.float64]): Right-hand side vector with shape (n_new_constraints,)
            or scalar for a single constraint.

        Raises:
            ValueError: If A and b have incompatible dimensions with the problem variables.

        Note:
            After adding constraints, the problem's feasibility is automatically checked.
            If infeasible, subsequent solve() calls will return -1.
        """
        A = np.atleast_2d(A)
        if A.shape != (b.size, self.n_vars):
            raise ValueError(
                f"A must have shape ({b.size}, {self.n_vars}) but has shape {A.shape}."
            )

        self.gb_model.addConstr(A @ self.gb_x <= b, name="leq")
        self.gb_model.setObjective(
            0.0, gp.GRB.MINIMIZE
        )  # Objective is zero for feasibility check
        self.gb_model.optimize()
        self.is_feas = self.gb_model.status == gp.GRB.OPTIMAL

        self.n_cstrs += b.size

        self.xt = np.append(self.xt, np.zeros_like(b))

        self.Wbxt_N = np.block([[self.Wbxt_N, self.A @ A.T], [A @ self.A.T, A @ A.T]])
        self.xibxt_N = np.append(self.xibxt_N, -b)

        self.mfut = np.append(self.mfut, np.zeros_like(b))
        self.Vfut = np.append(self.Vfut, np.zeros_like(b))
        self.beta = np.append(self.beta, np.full_like(b, self.beta_init))

        self.A = np.vstack((self.A, A))
        self.b = np.append(self.b, b)

        # logger.debug(f"{A=}, {b=}, {A.shape=}, {b.shape=}")

        # s = self.compute_sum_of_infeasibility()
        # logger.debug(f"Sum of infeasibility: {s}")
        # if np.sum(s) < FEAS_TOL:
        #     logger.info(
        #         f"New constraints are feasible. The quadratic problem with {self.n_vars} variables and {self.n_cstrs} constraints is feasible."
        #     )
        #     self.is_feas = True
        # else:
        #     logger.warning(
        #         f"New constraints are infeasible. The quadratic problem with {self.n_vars} variables and {self.n_cstrs} constraints is infeasible."
        #     )
        #     self.is_feas = False

        # s = self.compute_infeasibility()
        # logger.debug(f"Infeasibility: {s}")
        # if s < FEAS_TOL:
        #     logger.info(
        #         f"New constraints are feasible. The quadratic problem with {self.n_vars} variables and {self.n_cstrs} constraints is feasible."
        #     )
        #     self.is_feas = True
        # else:
        #     logger.warning(
        #         f"New constraints are infeasible. The quadratic problem with {self.n_vars} variables and {self.n_cstrs} constraints is infeasible."
        #     )
        #     self.is_feas = False

        # with gp.Env(empty=True) as env:
        #     env.setParam('OutputFlag', 0)
        #     env.start()
        #     with gp.Model(env=env) as m:
        #         x = m.addMVar(self.n_vars, name="x")
        #         m.addConstr(self.A @ x <= self.b, name="leq")
        #         m.setObjective(0.0, gp.GRB.MINIMIZE)
        #         m.optimize()

        #         if m.status != gp.GRB.OPTIMAL:
        #             logger.warning(
        #                 f"Problem with new constraints: Gurobi model status {m.status}."
        #             )
        #             self.is_feas = False
        #         else:
        #             logger.debug(
        #                 f"New constraints are valid: Gurobi model status is {m.status}."
        #             )
        #             self.is_feas = True

        logger.debug(
            f"{b.size} new constraint(s) successfully added. The new quadratic problem with {self.n_vars} variables and {self.n_cstrs} constraints is {'in' if not self.is_feas else ''}feasible."
        )

        self.primal_cost = np.nan
        self.dual_cost = np.nan
        self.primal_residual = np.nan
        # self.dual_residual = np.nan

    def compute_min_sum_of_infeasibility(self) -> npt.NDArray[np.float64]:
        with gp.Env(empty=True) as env:
            # env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:
                # Add variables for the Gurobi model
                x = m.addMVar(self.n_vars, lb=-np.inf, ub=np.inf, name="x")
                s = m.addMVar(self.n_cstrs, name="s")

                # Add constraints to the Gurobi model
                m.addConstr(self.A @ x <= self.b + s)

                # Add the objective function to minimize the slack variables
                m.setObjective(s.sum(), gp.GRB.MINIMIZE)

                # Optimize the Gurobi model
                m.optimize()

                if m.status == gp.GRB.OPTIMAL:
                    logger.info(
                        f"Constraints analysis completed successfully with value {m.ObjVal}."
                    )
                    return np.array(s.X)

                return np.full(self.n_cstrs, np.nan, dtype=np.float64)

                # if m.status != gp.GRB.OPTIMAL:
                #     logger.warning(
                #         f"Problem with new constraints: Gurobi model status {m.status}."
                #     )
                #     self.is_feas = False
                # else:
                #     logger.debug(
                #         f"New constraints are valid: Gurobi model status is {m.status}."
                #     )
                #     self.is_feas = True

    def compute_min_infeasibility(self) -> float | np.float64:
        with gp.Env(empty=True) as env:
            # env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:
                A_ext = np.hstack((self.A, -np.ones((self.n_cstrs, 1))))

                # Add variables for the Gurobi model
                x = m.addMVar(self.n_vars + 1, name="x")

                # Add constraints to the Gurobi model
                m.addConstr(A_ext @ x <= self.b)

                # Add the objective function to minimize the slack variables
                m.setObjective(x[-1], gp.GRB.MINIMIZE)

                # Optimize the Gurobi model
                m.optimize()

                if m.status == gp.GRB.OPTIMAL:
                    logger.info(
                        f"Constraints analysis completed successfully with value {x[-1].X}."
                    )
                    return x[-1].X

                return np.nan

    def solve(
        self,
        n_iter: int = 1000,
        atol: float = 1e-9,  # absolute tolerance for convergence
    ) -> int:
        """
        Solve the optimization problem defined by the constraints in the dual space.

        Args:
            n_iter (int): Maximum number of iterations allowed to converge to a solution.
            atol (float): Absolute tolerance for convergence. The optimization is considered converged if the dual cost and primal cost are within this tolerance. After convergence, the solution is guaranteed to be atol-suboptimal.
            assume_feas (bool): Whether to assume the problem is feasible. If True, the solver will not check for feasibility and will directly attempt to solve the problem. If False, it will check for feasibility before solving.

        Returns:
            int: The status of the optimization:
                - 1 if the optimization converged to a solution.
                - 0 if the optimization failed to converge or another unexpected issue occurred.
                - -1 if the optimization problem is not feasible.
        """

        # logger.info(
        #     f"The quadratic problem has {self.n_cstrs} constraints for {self.n_vars} variables."
        # )
        if not self.is_feas:
            logger.warning("The problem is infeasible. No solution can be found.")
            self.x.fill(np.nan)
            self.primal_cost = np.nan
            self.dual_cost = np.nan
            self.duality_gap = np.nan
            self.xt.fill(np.nan)

            return -1

        if self.n_vars < self.n_cstrs:
            logger.info(
                f"The problem is overdetermined ({self.n_vars} variables and {self.n_cstrs} constraints). DBFFD might not be the best choice for solving it..."
            )

        return self.dbffd(n_iter, atol)

    def update_primal_solution(self):
        """
        Update the primal solution based on the current dual solution. This guarantees that (x, xt) is a stationary point of the Lagrangian function.
        """
        self.x = (0.0 if self.mxf is None else self.mxf) - (
            self.A.T @ self.xt if self.Vxf is None else self.Vxf @ self.A.T @ self.xt
        )

    def update_primal_cost(self):
        """
        Update the primal cost based on the current solution.
        """
        self.primal_cost = (
            np.inner(self.x, self.x) / 2.0
            if self.Wxf is None
            else np.inner(self.x, self.Wxf @ self.x) / 2.0
        ) + (0.0 if self.xixf is None else -np.inner(self.x, self.xixf))

    def update_dual_cost(self):
        """
        Update the dual cost based on the current solution.
        """
        self.dual_cost = (
            np.inf
            if np.any(np.isposinf(self.xt))
            else (
                np.inner(self.xt, self.xibxt_N)
                - np.inner(self.xt, self.Wbxt_N @ self.xt) / 2.0
            )
        )

    def update_primal_residual(self):
        """
        Update the primal residual based on the current solution.
        The primal residual is defined as the difference between the left-hand side and right-hand side of the constraints.
        """
        self.primal_residual = self.A @ self.x - self.b

    def max_violation(self) -> np.float64:
        """
        Compute the maximum violation of the constraints.

        Returns:
            np.float64: The maximum violation of the constraints.
        """
        return np.max(self.primal_residual, initial=0.0)

    def is_feasible(self) -> bool | np.bool:
        """
        Check if the current solution is feasible, i.e., if all constraints are satisfied.

        Returns:
            bool: True if the solution is feasible, False otherwise.
        """
        return np.all(self.primal_residual <= FEAS_TOL) and np.all(self.xt >= -FEAS_TOL)

    def dbffd(
        self,
        n_iter: int = 1000,
        rtol: float = 1e-5,  # relative tolerance for suboptimality convergence test
        atol: float = 1e-8,  # absolute tolerance for suboptimality convergence test
    ) -> int:
        """
        Run backward-filtering forward-deciding in the dual space to optimize the dual solution.
        This updates the dual solution vector xt based on the current constraints.

        Returns:
            (int): the status of the optimization:
                - 1 if the optimization converged to a solution.
                - 0 if the optimization failed to converge within the maximum number of iterations.
        """
        en_xibxtn = np.empty((self.n_cstrs, 1))
        Wbxtn_en = np.empty((self.n_cstrs, self.n_cstrs))

        xibxt = np.empty((self.n_cstrs,))
        Wbxt = np.empty((self.n_cstrs, self.n_cstrs))

        # logger.debug(
        #     f"Starting DBFFD with {self.n_cstrs} constraints and {self.n_vars} variables. Previous costs are {self.prev_cost}."
        # # )
        # logger.debug(f"{self.xibxt_N=}")
        # logger.debug(f"{self.Wbxt_N=}")

        for i in range(n_iter):
            # Backward Filtering
            np.copyto(xibxt, self.xibxt_N)
            np.copyto(Wbxt, self.Wbxt_N)
            # logger.debug("Backward Filtering")
            for n in range(self.n_cstrs - 1, -1, -1):
                np.copyto(en_xibxtn[n], xibxt[n])
                np.copyto(Wbxtn_en[n], Wbxt[n])
                H = np.divide(self.Vfut[n], 1 + self.Vfut[n] * Wbxt[n, n]).squeeze()
                h = np.divide(
                    self.mfut[n] + self.Vfut[n] * xibxt[n],
                    1 + self.Vfut[n] * Wbxt[n, n],
                ).squeeze()
                xibxt -= h * Wbxt[n]
                Wbxt -= H * np.outer(Wbxt[n], Wbxt[n])
                # logger.debug(f"{n=}")
                # logger.debug(f"{Wbxt[n, n]=}")
                # logger.debug(f"{h=}")
                # logger.debug(f"{H=}")

            # Forward Deciding
            # logger.debug("Forward Deciding")
            self.xt.fill(0.0)
            for n in range(self.n_cstrs):
                # logger.debug(f"{n=}")
                Vbut = 1 / Wbxtn_en[n, n]
                xibut = (en_xibxtn[n] - np.inner(Wbxtn_en[n], self.xt)).squeeze()
                mbut = Vbut * xibut
                self.beta[n] = np.maximum(
                    self.beta[n], -xibut  # is xiut well-defined???
                )  # enforce xt[n] >= 0
                # utn = np.clip(mbut, 0.0, None)  # cf. Table 2 in LiLoeliger2024
                self.xt[n] += np.clip(mbut, 0.0, None)
                self.mfut[n] = self.xt[
                    n
                ]  # cf. Table 1 in LiLoeliger2024, mfut[n] = xt[n]
                self.Vfut[n] = (
                    2 * self.xt[n] / self.beta[n]
                )  # cf. Table 1 in LiLoeliger2024

            self.update_primal_solution()
            self.update_dual_cost()
            self.update_primal_cost()
            self.update_primal_residual()
            logger.debug(
                f"DBFFD - Iteration {i+1}: dual cost is {self.dual_cost} and primal cost is {self.primal_cost}"
            )
            logger.debug(
                f"Complementary slackness condition: {self.xt * self.primal_residual}"
            )

            # logger.debug(
            #     f"DBFFD - Iteration {i+1}: primal cost is {self.primal_cost} and dual cost is {self.dual_cost} (duality gap is {self.dual_cost - self.primal_cost}). Maximum violation is {self.max_violation() if self.n_cstrs > 0 else 'N/A'}."
            # )
            if self.is_feasible() and np.isclose(
                self.primal_cost, self.dual_cost, rtol, atol
            ):  # if x, xt is primal-dual feasible, the associated primal and dual costs give suboptimality guarantee
                logger.info(
                    f"DBFFD - Feasible solution found in {i+1} iteration(s) with cost {self.primal_cost} and maximum violation {self.max_violation()}. The duality gap is {self.dual_cost - self.primal_cost}."
                )
                return 1

                # logger.info(f"Converged after {i+1} iteration(s).")
                # max_violation = self.max_violation()
                # if max_violation <= ftol:
                #     logger.info(f"Feasible solution found with cost {self.cost}.")
                #     return 1
                # else:
                #     self.cost = np.nan
                #     logger.info(
                #         f"Primal problem is infeasible, maximum violation is {max_violation}."
                #     )

            # if np.any(np.isclose(self.cost, self.prev_cost, atol=conv_tol, atol=1e-12)):
            #     logger.info(f"Converged after {i+1} iteration(s).")
            #     max_violation = self.max_violation()
            #     if max_violation <= ftol:
            #         logger.info(f"Feasible solution found with cost {self.cost}.")
            #         return 1
            #     else:
            #         self.cost = np.nan
            #         logger.info(
            #             f"Primal problem is infeasible, maximum violation is {max_violation}."
            #         )
            #         return -1

            # self.prev_cost = np.append(self.prev_cost[1:], self.cost)

        logger.warning(
            "DBFFD - Failed to converge within the maximum number of iterations. Check feasibility or increase n_iter."
        )
        return 0

    # def is_valid(self, ftol: np.float64 | float) -> bool | np.bool:
    #     """
    #     Check if the current solution is valid, i.e., if all constraints are satisfied.

    #     Args:
    #         ftol (np.float64 | float): Relative tolerance for feasibility.

    #     Returns:
    #         bool: True if the solution is valid, False otherwise.
    #     """

    # ge = self.A @ self.x > self.b
    # return np.allclose(self.A[ge] @ self.x, self.b[ge], rtol=ftol, atol=1e-12)


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
    #     self.z_ck0 = np.empty((0, n_in_channels + 1), dtype=np.float64)
    #     self.z_ck1 = np.empty((0, n_in_channels + 1), dtype=np.float64)

    #     # Derivative of the potential state variables
    #     self.dz_start = np.empty((0,), dtype=np.float64)
    #     self.dz_length = np.empty((0,), dtype=np.float64)
    #     self.dz_lim = np.empty((0,), dtype=np.float64)
    #     self.dz_c0 = np.empty((0,), dtype=np.float64)
    #     self.dz_c1 = np.empty((0,), dtype=np.float64)
    #     self.dz_ck0 = np.empty((0, n_in_channels + 1), dtype=np.float64)
    #     self.dz_ck1 = np.empty((0, n_in_channels + 1), dtype=np.float64)

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
        n_in_channels: int,
        f_times: npt.NDArray[np.float64],
        in_times: npt.NDArray[np.float64],
        in_channels: npt.NDArray[np.intp],
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

        if n_in_channels < 0:
            raise ValueError(
                f"n_in_channels must be non-negative, got {n_in_channels}."
            )
        self.n_in_channels = n_in_channels

        if np.max(in_channels) > self.n_in_channels - 1 or np.min(in_channels) < 0:
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
        self.weight = np.full(n_in_channels, np.nan, dtype=np.float64)

        # Initialize the QProgram solver
        self.solver = QProgram(n_in_channels)

        self.z_start = []
        self.z_length = []
        self.z_lim = []
        self.z_ck0 = []
        self.z_ck1 = []
        self.z_c0 = []
        self.z_c1 = []

        self.dz_start = []
        self.dz_length = []
        self.dz_lim = []
        self.dz_ck0 = []
        self.dz_ck1 = []
        self.dz_c0 = []
        self.dz_c1 = []

        # Initialize the neuron states based on the provided firing times and input times
        if f_times.size > 0:
            f_times = np.sort(f_times)
            times = np.diff(f_times, append=f_times[0] + period)[
                :, np.newaxis
            ] - np.array(
                [[eps, 0.0]]
            )  # times_n = [f_times_n - eps, f_times_n] with an offset
            in_times = (in_times[np.newaxis, :] - f_times[:, np.newaxis]) % period
            logger.debug(f"{in_times.shape=}, {times.shape=}, {f_times.shape=}")

            A_f_times = []
            b_f_times = []

            for in_times_n, times_n, offset in zip(in_times, times, f_times):
                start, length, ck0, ck1 = compute_ck(
                    in_times_n,
                    in_channels,
                    self.n_in_channels,
                    times_n,
                    REFRACTORY_RESET,
                )
                logger.debug(
                    f"The time interval starting at {offset} has been partitioned into {start.size} intervals and ends in {offset + start[-1] + length[-1]}."
                )

                # Firing time constraint | z >= f_thresh
                A_f_times.append(-ck0[-1, :-1])
                b_f_times.append(-f_thresh + ck0[-1, -1])

                ## Silent time intervals | z <= zmax
                logger.debug(f"Silent zone between {start[0]} and {times_n[0]}.")
                silent = start < times_n[0]
                self.z_start.append(start[silent] + offset)
                self.z_length.append(length[silent])
                self.z_lim.append(np.full_like(start[silent], zmax))
                self.z_ck0.append(ck0[silent])
                self.z_ck1.append(ck1[silent])
                self.z_c0.append(np.full_like(start[silent], np.nan))
                self.z_c1.append(np.full_like(start[silent], np.nan))

                ## Active time intervals | dz >= dzmin
                logger.debug(f"Silent zone between {times_n[0]} and {times_n[-1]}.")
                active = (start >= times_n[0]) & (start < times_n[-1])
                self.dz_start.append(start[active] + offset)
                self.dz_length.append(
                    (length[active] - LEFT_LIMIT).clip(min=0.0)
                )  # the potential derivative is right-continuous
                self.dz_lim.append(np.full_like(start[active], -dzmin))
                self.dz_ck0.append(ck0[active] - ck1[active])
                self.dz_ck1.append(ck1[active])
                self.dz_c0.append(np.full_like(start[active], np.nan))
                self.dz_c1.append(np.full_like(start[active], np.nan))

            # Add the firing time constraints to the solver
            self.solver.add_constraints(np.vstack(A_f_times), np.array(b_f_times))

        else:
            in_times = in_times % period
            start, length, ck0, ck1 = compute_ck(
                in_times,
                in_channels,
                self.n_in_channels,
                np.array([0.0, period]),
                0.0,
            )

            ## Silent time intervals | z <= zmax
            silent = start < period
            self.z_start.append(start[silent])
            self.z_length.append(length[silent])
            self.z_lim.append(np.full_like(start[silent], zmax))
            self.z_ck0.append(ck0[silent])
            self.z_ck1.append(ck1[silent])
            self.z_c0.append(np.full_like(start[silent], np.nan))
            self.z_c1.append(np.full_like(start[silent], np.nan))

            self.dz_start.append(np.empty((0,), dtype=np.float64))
            self.dz_length.append(np.empty((0,), dtype=np.float64))
            self.dz_lim.append(np.empty((0,), dtype=np.float64))
            self.dz_ck0.append(np.empty((0, n_in_channels + 1), dtype=np.float64))
            self.dz_ck1.append(np.empty((0, n_in_channels + 1), dtype=np.float64))
            self.dz_c0.append(np.empty((0,), dtype=np.float64))
            self.dz_c1.append(np.empty((0,), dtype=np.float64))

        logger.debug(
            f"Neuron initialized successfully! The solver has been initialized with {self.solver.n_cstrs} constraints for {self.solver.n_vars} variables."
        )

    #     # Potential state variables
    #     self.z_start = np.empty((0,), dtype=np.float64)
    #     self.z_length = np.empty((0,), dtype=np.float64)
    #     self.z_lim = np.empty((0,), dtype=np.float64)
    #     self.z_c0 = np.empty((0,), dtype=np.float64)
    #     self.z_c1 = np.empty((0,), dtype=np.float64)
    #     self.z_ck0 = np.empty((0, n_in_channels + 1), dtype=np.float64)
    #     self.z_ck1 = np.empty((0, n_in_channels + 1), dtype=np.float64)

    #     # Derivative of the potential state variables
    #     self.dz_start = np.empty((0,), dtype=np.float64)
    #     self.dz_length = np.empty((0,), dtype=np.float64)
    #     self.dz_lim = np.empty((0,), dtype=np.float64)
    #     self.dz_c0 = np.empty((0,), dtype=np.float64)
    #     self.dz_c1 = np.empty((0,), dtype=np.float64)
    #     self.dz_ck0 = np.empty((0, n_in_channels + 1), dtype=np.float64)
    #     self.dz_ck1 = np.empty((0, n_in_channels + 1), dtype=np.float64)

    #     # QProgram
    #     self.solver = QProgram(n_in_channels)

    # def init_learning(
    #     self,
    #     f_times: npt.NDArray[np.float64],
    #     in_times: npt.NDArray[np.float64],
    #     in_channels: npt.NDArray[np.intp],
    #     period: np.float64 | float,
    #     eps: np.float64 | float = REFRACTORY_PERIOD,
    #     f_thresh: np.float64 | float = FIRING_THRESHOLD,
    #     zmax: np.float64 | float = 0.0,
    #     dzmin: np.float64 | float = 1e-2,
    # ):
    #     if eps < 0.0 or eps > REFRACTORY_PERIOD:
    #         raise ValueError(f"eps must be in [0, {REFRACTORY_PERIOD}], got {eps}.")

    #     if zmax >= f_thresh:
    #         raise ValueError(
    #             f"zmax must be less than f_thresh ({f_thresh}), got {zmax}."
    #         )

    #     if dzmin <= 0.0:
    #         raise ValueError(f"dzmin must be positive, got {dzmin}.")

    #     logger.debug("Initialize the neuron for learning...")

    #     self.weight.fill(np.nan)

    #     if f_times.size > 0:
    #         f_times.sort()
    #         times = np.diff(f_times, append=f_times[0] + period)[
    #             :, np.newaxis
    #         ] - np.array([[eps, 0.0]])
    #         in_times = (in_times[np.newaxis, :] - f_times[:, np.newaxis]) % period

    #         z_start, z_length, z_lim, z_ck0, z_ck1 = [], [], [], [], []
    #         dz_start, dz_length, dz_lim, dz_ck0, dz_ck1 = [], [], [], [], []

    #         A_f_times = []
    #         b_f_times = []

    #         for in_times_n, times_n, offset in zip(in_times, times, f_times):
    #             start, length, ck0, ck1 = compute_ck(
    #                 in_times_n,
    #                 in_channels,
    #                 self.n_in_channels,
    #                 times_n,
    #                 REFRACTORY_RESET,
    #             )

    #             # Firing time constraint | z >= f_thresh
    #             A_f_times.append(-ck0[-1, :-1])
    #             b_f_times.append(-f_thresh + ck0[-1, -1])

    #             ## Silent time intervals | z <= zmax
    #             silent = start < times_n[0]
    #             z_start.append(start[silent] + offset)
    #             z_length.append(length[silent])
    #             z_lim.append(np.full(start[silent].shape, zmax))
    #             z_ck0.append(ck0[silent])
    #             z_ck1.append(ck1[silent])

    #             ## Active time intervals | z <= f_thresh and dz >= dzmin
    #             active = (start >= times_n[0]) & (start < times_n[-1])
    #             z_start.append(start[active] + offset)
    #             z_length.append(length[active])
    #             z_lim.append(np.full(start[active].shape, f_thresh))
    #             z_ck0.append(ck0[active])
    #             z_ck1.append(ck1[active])
    #             dz_start.append(start[active] + offset)
    #             dz_length.append(
    #                 (length[active] - LEFT_LIMIT).clip(min=0.0)
    #             )  # the potential derivative is right-continuous
    #             dz_lim.append(np.full(start[active].shape, -dzmin))
    #             dz_ck0.append(ck0[active] - ck1[active])
    #             dz_ck1.append(ck1[active])

    #         self.z_start = np.concatenate(z_start)
    #         self.z_length = np.concatenate(z_length)
    #         self.z_lim = np.concatenate(z_lim)
    #         self.z_ck0 = np.concatenate(z_ck0)
    #         self.z_ck1 = np.concatenate(z_ck1)

    #         self.dz_start = np.concatenate(dz_start)
    #         self.dz_length = np.concatenate(dz_length)
    #         self.dz_lim = np.concatenate(dz_lim)
    #         self.dz_ck0 = np.concatenate(dz_ck0)
    #         self.dz_ck1 = np.concatenate(dz_ck1)

    #         self.solver.add_constraints(np.vstack(A_f_times), np.array(b_f_times))

    #     else:
    #         in_times = in_times % period
    #         start, length, ck0, ck1 = compute_ck(
    #             in_times,
    #             in_channels,
    #             self.n_in_channels,
    #             np.array([0.0, period]),
    #             0.0,
    #         )

    #         ## Silent time intervals | z <= zmax
    #         silent = start < period
    #         self.z_start = start[silent]
    #         self.z_length = length[silent]
    #         self.z_lim = np.full(start[silent].shape, zmax)
    #         self.z_ck0 = ck0[silent]
    #         self.z_ck1 = ck1[silent]

    #         self.dz_start = np.empty((0,), dtype=np.float64)
    #         self.dz_length = np.empty((0,), dtype=np.float64)
    #         self.dz_lim = np.empty((0,), dtype=np.float64)
    #         self.dz_ck0 = np.empty((0, self.n_in_channels + 1), dtype=np.float64)
    #         self.dz_ck1 = np.empty((0, self.n_in_channels + 1), dtype=np.float64)

    #     logger.debug(
    #         f"Neuron initialized successfully! The time axis has been decomposed into {self.n_z_intervals} (potential) and {self.n_dz_intervals} (potential derivative) intervals. The solver has been initialized with {self.solver.n_cstrs} constraints for {self.solver.n_vars} variables."
    #     )

    def refine_constraints(
        self, ftol: float = 1e-9, n_cstrs_per_template: int = 1
    ) -> int:
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

        for n in range(self.n_intervals):
            # Potential constraints
            self.z_c0[n] = (
                np.inner(self.z_ck0[n][:, :-1], self.solver.x) + self.z_ck0[n][:, -1]
            )
            self.z_c1[n] = (
                np.inner(self.z_ck1[n][:, :-1], self.solver.x) + self.z_ck1[n][:, -1]
            )
            z_dvmax, z_imax, z_dtmax = find_maximum_violation(
                self.z_c0[n], self.z_c1[n], self.z_length[n], self.z_lim[n] + ftol
            ) or (0.0, None, None)

            if z_dvmax > 0.0 and z_imax is not None and z_dtmax is not None:
                if np.isfinite(z_dtmax):
                    A.append(
                        (
                            self.z_ck0[n][z_imax, :-1]
                            + z_dtmax * self.z_ck1[n][z_imax, :-1]
                        )
                        * np.exp(-z_dtmax)
                    )
                    b.append(
                        self.z_lim[n][z_imax]
                        - (
                            self.z_ck0[n][z_imax, -1]
                            + z_dtmax * self.z_ck1[n][z_imax, -1]
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
                np.inner(self.dz_ck0[n][:, :-1], self.solver.x) + self.dz_ck0[n][:, -1]
            )
            self.dz_c1[n] = (
                np.inner(self.dz_ck1[n][:, :-1], self.solver.x) + self.dz_ck1[n][:, -1]
            )
            dz_dvmax, dz_imax, dz_dtmax = find_maximum_violation(
                self.dz_c0[n], self.dz_c1[n], self.dz_length[n], self.dz_lim[n] + ftol
            ) or (0.0, None, None)

            if dz_dvmax > 0.0 and dz_imax is not None and dz_dtmax is not None:
                logger.debug(f"{dz_imax=}")
                if np.isfinite(dz_dtmax):
                    A.append(
                        (
                            self.dz_ck0[n][dz_imax, :-1]
                            + dz_dtmax * self.dz_ck1[n][dz_imax, :-1]
                        )
                        * np.exp(-dz_dtmax)
                    )
                    b.append(
                        self.dz_lim[n][dz_imax]
                        - (
                            self.dz_ck0[n][dz_imax, -1]
                            + dz_dtmax * self.dz_ck1[n][dz_imax, -1]
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
            logger.info("Constraint refinement: all constraints are satisfied!")
            return 1

        # Add the new constraints to the solver
        self.solver.add_constraints(np.vstack(A), np.array(b))
        return 0

        # self.z_c0 = np.inner(self.z_ck0[:, :-1], self.solver.x) + self.z_ck0[:, -1]
        # self.z_c1 = np.inner(self.z_ck1[:, :-1], self.solver.x) + self.z_ck1[:, -1]
        # z_dvmax, z_imax, z_dtmax = find_maximum_violation(
        #     self.z_c0, self.z_c1, self.z_length, self.z_lim + ftol
        # ) or (0.0, None, None)

        # self.dz_c0 = np.inner(self.dz_ck0[:, :-1], self.solver.x) + self.dz_ck0[:, -1]
        # self.dz_c1 = np.inner(self.dz_ck1[:, :-1], self.solver.x) + self.dz_ck1[:, -1]
        # dz_dvmax, dz_imax, dz_dtmax = find_maximum_violation(
        #     self.dz_c0, self.dz_c1, self.dz_length, self.dz_lim + ftol
        # ) or (0.0, None, None)

        # if z_dvmax == 0.0 and dz_dvmax == 0.0:
        #     logger.info("All constraints are satisfied.")
        #     return 1

        # if z_dvmax >= dz_dvmax and z_imax is not None and z_dtmax is not None:
        #     if np.isfinite(z_dtmax):
        #         self.solver.add_constraints(
        #             (self.z_ck0[z_imax, :-1] + z_dtmax * self.z_ck1[z_imax, :-1])
        #             * np.exp(-z_dtmax),
        #             self.z_lim[z_imax]
        #             - (self.z_ck0[z_imax, -1] + z_dtmax * self.z_ck1[z_imax, -1])
        #             * np.exp(-z_dtmax),
        #         )
        #     else:
        #         self.solver.add_constraints(
        #             np.zeros(self.solver.n_vars), self.z_lim[z_imax]
        #         )
        #     logger.debug(
        #         f"A violation of the potential template has been detected on the {z_imax}th time interval"
        #     )
        #     return 0

        # elif z_dvmax < dz_dvmax and dz_imax is not None and dz_dtmax is not None:
        #     if np.isfinite(dz_dtmax):
        #         self.solver.add_constraints(
        #             (self.dz_ck0[dz_imax, :-1] + dz_dtmax * self.dz_ck1[dz_imax, :-1])
        #             * np.exp(-dz_dtmax),
        #             self.dz_lim[dz_imax]
        #             - (self.dz_ck0[dz_imax, -1] + dz_dtmax * self.dz_ck1[dz_imax, -1])
        #             * np.exp(-dz_dtmax),
        #         )
        #     else:
        #         self.solver.add_constraints(
        #             np.zeros(self.solver.n_vars), self.z_lim[z_imax]
        #         )
        #     logger.debug(
        #         f"A violation of the potential derivative template has been detected on the {dz_imax}th time interval"
        #     )
        #     return 0

        # logger.critical("Refinement failed...")
        # return -1

    def learn(
        self,
        ftol: float = 1e-9,
        atol: float = 1e-9,
        n_cstrs: int = 1000,
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
            res = self.solver.solve(atol=atol)
            if res < 1:
                return res

            # 2. Refine constraints based on the current primal solution.
            res_refine = self.refine_constraints(ftol)
            if (
                res_refine > 0
            ):  # If no constraints are violated, then the primal solution is optimal.
                logger.debug(
                    f"Learning succeeds! Solved in {i+1} refinement iterations. Number of variables: {self.solver.n_vars}. Number of constraints: {self.solver.n_cstrs}. Cost: {self.solver.primal_cost:.3f}."
                )
                self.weight = np.copy(self.solver.x)
                return 1

        logger.debug(
            "Neuron optimization did not converge within the maximum number of iterations."
        )
        return 0
