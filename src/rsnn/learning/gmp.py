from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from rsnn.constants import *
from rsnn.utils import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")


class GMPModel:
    def __init__(
        self,
        n_vars: int,
        mxf: Optional[npt.NDArray[np.float64]] = None,
        Vxf: Optional[npt.NDArray[np.float64]] = None,
        beta_init: float | np.float64 = 1e-15,  # initial value for beta
    ):
        """
        Initialize a Gaussian message passing based solver for quadratic optimization problems subject to linear constraints.

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
        self.primal_residual = np.zeros(self.n_cstrs, dtype=np.float64)

        logger.debug(
            f"Quadratic program initialized with {self.n_vars} variables (and {self.n_cstrs} constraints)."
        )

    def x_value(self) -> npt.NDArray[np.float64]:
        """
        Get the current value of the optimization variables.

        Returns:
            npt.NDArray[np.float64]: The current values of the optimization variables.
        """
        return self.x

    def cost_value(self) -> float:
        """
        Get the current value of the cost function.

        Returns:
            float: The current value of the cost function.
        """
        return self.primal_cost

    def get_constraints(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get the current constraints of the optimization problem.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: The constraint matrix A and the right-hand side vector b.
        """
        return self.A, self.b

    def add_constraints(
        self,
        A: npt.NDArray[np.float64],  # shape (n_cstrs, n_vars) or (n_vars, )
        b: npt.NDArray[np.float64],  # shape (n_cstrs, 1) or (n_cstrs, )
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

        self.n_cstrs += b.size

        self.xt = np.append(self.xt, np.zeros_like(b))

        self.Wbxt_N = np.block([[self.Wbxt_N, self.A @ A.T], [A @ self.A.T, A @ A.T]])
        self.xibxt_N = np.append(self.xibxt_N, -b)

        self.mfut = np.append(self.mfut, np.zeros_like(b))
        self.Vfut = np.append(self.Vfut, np.zeros_like(b))
        self.beta = np.append(self.beta, np.full_like(b, self.beta_init))

        self.A = np.vstack((self.A, A))
        self.b = np.append(self.b, b)

        logger.debug(
            f"{b.size} new constraint(s) successfully added. The new quadratic problem with {self.n_vars} variables and {self.n_cstrs}."
        )

        self.primal_cost = np.nan
        self.dual_cost = np.nan
        self.primal_residual = np.nan

    def solve(
        self,
        n_iter: int = 10000,
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
        rtol: float = 1e-3,  # relative tolerance for suboptimality convergence test
        atol: float = 1e-6,  # absolute tolerance for suboptimality convergence test
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

        for i in range(n_iter):
            # Backward Filtering
            np.copyto(xibxt, self.xibxt_N)
            np.copyto(Wbxt, self.Wbxt_N)
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

            # Forward Deciding
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
                f"DBFFD - Iteration {i+1}: primal cost is {self.primal_cost} and dual cost is {self.dual_cost} (duality gap is {self.dual_cost - self.primal_cost}). Maximum violation is {self.max_violation() if self.n_cstrs > 0 else 'N/A'}."
            )
            if self.is_feasible() and np.isclose(
                self.primal_cost, self.dual_cost, rtol, atol
            ):  # if x, xt is primal-dual feasible, the associated primal and dual costs give suboptimality guarantee
                logger.debug(
                    f"DBFFD - Feasible solution found in {i+1} iteration(s) with cost {self.primal_cost} and maximum violation {self.max_violation()}. The duality gap is {self.dual_cost - self.primal_cost}."
                )
                return 1

        logger.warning(
            "DBFFD - Failed to converge within the maximum number of iterations. Check feasibility or increase n_iter."
        )
        return 0
