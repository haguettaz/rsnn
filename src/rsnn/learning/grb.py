import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
import numpy.typing as npt
import scipy as sp

from rsnn.constants import *
from rsnn.utils import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")

class GRBModel:
    def __init__(
        self,
        n_vars: int,
        xmin: float | np.float64 = -np.inf,
        xmax: float | np.float64 = np.inf,
        mxf: Optional[npt.NDArray[np.float64]] = None,
        Vxf: Optional[npt.NDArray[np.float64]] = None,
    ):
        """
        Initialize a Gurobi solver for quadratic optimization problems subject to linear constraints.

        Args:
            n_vars (int | np.intp): Number of variables in the optimization problem.
            mxf (Optional[npt.NDArray[np.float64]]): Mean vector for the primal cost. If None, defaults to a zero vector.
            Vxf (Optional[npt.NDArray[np.float64]]): Covariance matrix for the primal cost. If None, defaults to an identity matrix.

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

        # Initialize a Gurobi model
        self.model = gp.Model("qp_model")
        self.model.setParam("OutputFlag", 0)  # Suppress Gurobi output

        self.x = self.model.addMVar(self.n_vars, lb=xmin, ub=xmax, name="x")

        self.model.setObjective(
            0.5 * self.x @ (self.Wxf @ self.x if self.Wxf is not None else self.x)
            - (self.xixf @ self.x if self.xixf is not None else 0),
            gp.GRB.MINIMIZE,
        )
        self.model.update()  # Update the model to reflect the new variables and objective

        logger.debug(
            f"Quadratic program initialized with {self.n_vars} variables (and {self.n_cstrs} constraints)."
        )

    def x_value(self) -> npt.NDArray[np.float64]:
        """
        Get the current value of the optimization variables.

        Returns:
            npt.NDArray[np.float64]: The current values of the optimization variables.
        """
        return self.model.x if self.model.status == gp.GRB.OPTIMAL else np.full(self.n_vars, np.nan)
    
    def cost_value(self) -> float:
        """
        Get the current value of the objective function.

        Returns:
            float: The current value of the objective function.
        """
        return self.model.objVal if self.model.status == gp.GRB.OPTIMAL else np.nan

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

        self.model.addConstr(A @ self.x <= b, name="leq")
        self.model.update()

        self.n_cstrs += b.size

        logger.debug(
            f"{b.size} new constraint(s) successfully added. The new quadratic problem has {self.n_vars} variables and {self.n_cstrs} constraints."
        )

    def get_constraints(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get the current constraints of the optimization problem.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: A tuple containing:
                - A: The constraint coefficient matrix (shape (n_cstrs, n_vars)).
                - b: The right-hand side vector (shape (n_cstrs,)).
        """
        A = sp.sparse.csr_matrix.todense(self.model.getA())
        b = np.array(self.model.getAttr("RHS", self.model.getConstrs()))
        return A, b

    def solve(
        self,
    ) -> int:
        """
        Solve the optimization problem defined by the constraints in the dual space.

        Returns:
            int: The status of the optimization:
                - 1 if the optimization converged to a solution.
                - 0 if the optimization failed to converge or another unexpected issue occurred.
                - -1 if the optimization problem is not feasible.
        """

        self.model.optimize()

        if self.model.status == gp.GRB.OPTIMAL:
            logger.debug(
                f"The quadratic problem with {self.n_cstrs} constraints and {self.n_vars} variables has been solved successfully with cost {self.model.objVal}."
            )
            self.is_feas = True
            return 1
        elif self.model.status == gp.GRB.INFEASIBLE or self.model.status == gp.GRB.INF_OR_UNBD:
            logger.warning(
                "The quadratic problem is infeasible. No solution can be found."
            )
            self.is_feas = False
            return -1
        else:
            logger.error(
                f"Unexpected Gurobi model status: {self.model.status}. Cannot solve the problem."
            )
            return 0