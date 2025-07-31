from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
import numpy.typing as npt
import scipy.sparse as ss

from .constants import FIRING_THRESHOLD, REFRACTORY_PERIOD, REFRACTORY_RESET
from .log import setup_logging
from .utils import find_maximum_violation, fscan_states, modulo_with_offset

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")

FEAS_TOL = 1e-6  # Tolerance for feasibility checks

class Solver:
    def __init__(
        self,
        n_synapses: int,
        weights_min: np.float64 | float = -np.inf,
        weights_max: np.float64 | float = np.inf,
    ):
        """
        Initialize the quadratic programming solver for synaptic weight optimization.
        
        Creates a Gurobi optimization model with decision variables for synaptic weights
        subject to bound constraints.
        
        Parameters
        ----------
        n_synapses : int
            Number of synaptic weights to optimize.
        weights_min : float64 or float, optional
            Lower bound for synaptic weights. Default is -inf.
        weights_max : float64 or float, optional
            Upper bound for synaptic weights. Default is +inf.
            
        Notes
        -----
        The solver uses Gurobi as the underlying optimization engine with
        output suppressed by default.
        """
        self.n_synapses = n_synapses
        self.n_intervals = 0

        # Initialize a Gurobi model
        self.model = gp.Model("qp_model")
        self.model.setParam("OutputFlag", 0)  # Suppress Gurobi output

        self.weights = self.model.addMVar(
            self.n_synapses, lb=weights_min, ub=weights_max, name="weights"
        )
        logger.info(f"Solver initialized with {self.n_synapses} synapses.")

    @property
    def n_constraints(self) -> int:
        """
        Get the number of constraints in the optimization model.

        Returns
        -------
        int
            Current number of constraints in the Gurobi model.
        """
        return self.model.getAttr("NumConstrs")

    def init_objective(self, xif=None, Wf=None):
        """
        Initialize the quadratic objective function for the optimization.
        
        Sets up the objective as: 0.5 * w^T * W * w - x^T * w
        where w are the synaptic weights.
        
        Parameters
        ----------
        xif : array_like, optional
            Linear coefficient vector for the objective. If None, no linear term.
        Wf : array_like, optional  
            Quadratic coefficient matrix for the objective. If None, uses identity matrix.
            
        Notes
        -----
        The objective is set to minimize the quadratic form. The model is updated
        after setting the objective.
        """
        self.model.setObjective(
            0.5
            * self.weights
            @ (Wf @ self.weights if Wf is not None else self.weights)
            - (xif @ self.weights if xif is not None else 0),
            gp.GRB.MINIMIZE,
        )
        self.model.update()  # Update the model to reflect the new objective
        logger.info("Solver initialized with an objective.")

    def cost_value(self) -> float:
        """
        Get the current optimal objective function value.

        Returns
        -------
        float
            Objective function value if model is optimal, otherwise NaN.
        """
        return self.model.objVal if self.model.status == gp.GRB.OPTIMAL else np.nan

    def get_constraints(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Extract current constraint matrix and right-hand side vector.

        Returns
        -------
        A : ndarray of float64, shape (n_constraints, n_variables)
            Constraint coefficient matrix.
        b : ndarray of float64, shape (n_constraints,)
            Right-hand side vector of constraints.
        """
        A = ss.csr_matrix.todense(self.model.getA())
        b = np.array(self.model.getAttr("RHS", self.model.getConstrs()))
        return A, b

    def solve(
        self,
    ) -> int:
        """
        Solve the current quadratic programming problem.

        Returns
        -------
        int
            Optimization status code:
            * 1 : Optimal solution found
            * 0 : Solver failed or unexpected status
            * -1 : Problem is infeasible
            
        Notes
        -----
        Sets the `is_feas` attribute based on feasibility of the solution.
        """

        self.model.optimize()

        if self.model.status == gp.GRB.OPTIMAL:
            logger.debug(
                f"The quadratic problem with {self.n_constraints} constraints and {self.n_synapses} variables has been solved successfully with cost {self.model.objVal}."
            )
            self.is_feas = True
            return 1
        elif (
            self.model.status == gp.GRB.INFEASIBLE
            or self.model.status == gp.GRB.INF_OR_UNBD
        ):
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

    def run(
        self,
        niter: int = 1000,
    ) -> int:
        """
        Execute the iterative constraint refinement algorithm.
        
        Alternately solves the quadratic program and refines constraints until
        convergence or maximum iterations reached.

        Parameters
        ----------
        niter : int, optional
            Maximum number of refinement iterations. Default is 1000.

        Returns
        -------
        int
            Optimization status code:
            * 1 : Successfully converged to optimal solution
            * 0 : Failed to converge within maximum iterations
            * -1 : Problem became infeasible during refinement
            
        Notes
        -----
        The algorithm uses constraint refinement to handle the infinite number
        of potential constraints by iteratively adding violated constraints.
        """

        for i in range(niter):
            res = self.solve()
            if res < 1:
                return res

            # 2. Refine constraints based on the current primal solution.
            res_refine = self.refine_constraints()
            if (
                res_refine > 0
            ):  # If no constraints are violated, then the primal solution is optimal.
                logger.info(
                    f"Learning succeeds! Solved in {i+1} refinement iterations. Number of variables: {self.n_synapses}. Number of constraints: {self.n_constraints}. Cost: {self.cost_value():.6f}."
                )
                # self.weight = np.copy(self.solver.model.x.X)
                return 1

        logger.warning(
            "Neuron optimization did not converge within the maximum number of iterations."
        )
        return 0

    def init_template_constraints(
        self,
        f_times: npt.NDArray[np.float64],
        in_times: npt.NDArray[np.float64],
        in_sources: npt.NDArray[np.int64],
        period: np.float64 | float,
        eps: np.float64 | float = 0.2 * REFRACTORY_PERIOD,
        f_thresh: np.float64 | float = FIRING_THRESHOLD,
        zmax: np.float64 | float = 0.0,
        dzmin: np.float64 | float = 1e-2,
    ):
        """
        Initialize constraint templates for neuron dynamics optimization.
        
        Sets up the foundational constraints for neuron firing behavior including
        threshold crossing, silent periods, and derivative constraints.

        Parameters
        ----------
        f_times : ndarray of float64
            Desired firing times for the neuron.
        in_times : ndarray of float64
            Arrival times of input spikes.
        in_sources : ndarray of int64
            Source synapse indices for each input spike.
        period : float64 or float
            Period for periodic boundary conditions.
        eps : float64 or float, optional
            Duration before firing time defining the active zone. 
            Default is 0.2 * REFRACTORY_PERIOD.
        f_thresh : float64 or float, optional
            Firing threshold for the neuron. Default is FIRING_THRESHOLD.
        zmax : float64 or float, optional
            Maximum allowed membrane potential in silent zones. Default is 0.0.
        dzmin : float64 or float, optional
            Minimum required potential derivative in active zones. Default is 1e-2.
            
        Notes
        -----
        This method:
        - Clears existing constraints from the model
        - Partitions time into intervals based on firing times
        - Sets up state tracking for membrane potential dynamics
        - Adds threshold crossing constraints at firing times
        - Initializes template constraints for silent and active zones
        """
        # Clean all constraints from the model
        self.model.remove(self.model.getConstrs())

        # Sort the firing times
        f_times.sort()
        self.n_intervals = np.clip(f_times.size, 1, None)

        # Initialize the lists for storing the neuron states
        self.z_start = []
        self.z_length = []
        self.z_lim = []
        self.z_in_c0 = []
        self.z_in_c1 = []
        self.z_c0 = []
        self.z_c1 = []

        # Initialize the lists for storing the neuron derivative states
        self.dz_start = []
        self.dz_length = []
        self.dz_lim = []
        self.dz_in_c0 = []
        self.dz_in_c1 = []
        self.dz_c0 = []
        self.dz_c1 = []

        # Initialize the neuron states based on the provided firing times and input times
        if f_times.size > 0:
            for n in range(f_times.size):
                # Compute the time markers for the current firing time, taking care of periodicity
                f_time = f_times[n]  # Current firing time
                prev_f_time = modulo_with_offset(
                    f_times[(n - 1) % f_times.size], period, f_time - period
                ).item()
                a_time = np.clip(f_time - eps, prev_f_time, f_time)

                in_times = modulo_with_offset(in_times, period, prev_f_time)
                in_mask = in_times < f_time
                masked_in_times = in_times[in_mask]
                masked_in_sources = in_sources[in_mask]

                # Initialize the state variables
                start = np.concatenate(
                    [
                        masked_in_times,
                        np.array([prev_f_time, a_time, f_time]),
                    ]
                )
                in_ic0 = np.zeros((start.size, self.n_synapses + 1))
                in_ic0[-3, -1] = 1.0  # Refractory reset
                in_ic1 = np.zeros((start.size, self.n_synapses + 1))
                in_ic1[np.arange(masked_in_sources.size), masked_in_sources] = (
                    1.0  # Synaptic inputs
                )
                in_c0 = np.zeros_like(in_ic0)
                in_c1 = np.zeros_like(in_ic1)

                # Sort the states
                sorter = np.argsort(start, axis=0)
                start = np.take_along_axis(start, sorter, axis=0)
                in_ic0 = np.take_along_axis(in_ic0, sorter[:, None], axis=0)
                in_ic1 = np.take_along_axis(in_ic1, sorter[:, None], axis=0)

                # Scan the states in place (forward update)
                fscan_states(start, in_ic0, in_ic1, in_c0, in_c1)

                logger.debug(
                    f"The time interval [{prev_f_time}, {f_time}] has been partitioned into {start.size} intervals."
                )

                # Add the threshold crossing constraint at firing time constraint: z >= f_thresh
                self.model.addConstr(
                    in_c0[-1, :-1] @ self.weights
                    >= f_thresh - REFRACTORY_RESET * in_c0[-1, -1],
                    name=f"f_time_{n}",
                )

                ## Silent time intervals: z <= z_max
                logger.debug(f"Silent zone is [{prev_f_time}, {a_time}].")
                mask = start < a_time
                masked_start = start[mask]
                masked_in_c0 = in_c0[mask]
                masked_in_c1 = in_c1[mask]
                self.z_start.append(masked_start)
                self.z_length.append(np.diff(masked_start, append=a_time))
                self.z_lim.append(np.full_like(masked_start, zmax))
                self.z_in_c0.append(masked_in_c0)
                self.z_in_c1.append(masked_in_c1)
                self.z_c0.append(np.full_like(masked_start, np.nan))
                self.z_c1.append(np.full_like(masked_start, np.nan))

                ## Active time intervals: dz >= dz_min
                logger.debug(f"Active zone is [{a_time}, {f_time}].")
                mask = (start >= a_time) & (start < f_time)
                masked_start = start[mask]
                masked_in_c0 = in_c0[mask]
                masked_in_c1 = in_c1[mask]
                self.dz_start.append(masked_start)
                self.dz_length.append(np.diff(masked_start, append=f_time))
                self.dz_lim.append(np.full_like(masked_start, -dzmin))
                self.dz_in_c0.append(masked_in_c0 - masked_in_c1)
                self.dz_in_c1.append(masked_in_c1)
                self.dz_c0.append(np.full_like(masked_start, np.nan))
                self.dz_c1.append(np.full_like(masked_start, np.nan))
        else:
            in_times = modulo_with_offset(in_times, period)

            # Initialize the state variables
            start = np.concatenate(
                [
                    in_times,
                    np.array([0.0, period]),
                ]
            )
            in_ic0 = np.zeros((start.size, self.n_synapses + 1))
            in_ic1 = np.zeros((start.size, self.n_synapses + 1))
            in_ic1[np.arange(in_sources.size), in_sources] = 1.0  # Synaptic inputs
            in_c0 = np.zeros_like(in_ic0)
            in_c1 = np.zeros_like(in_ic1)

            # Sort the states
            sorter = np.argsort(start, axis=0)
            start = np.take_along_axis(start, sorter, axis=0)
            in_ic0 = np.take_along_axis(in_ic0, sorter[:, None], axis=0)
            in_ic1 = np.take_along_axis(in_ic1, sorter[:, None], axis=0)

            # Compute the initial states from periodic conditions
            duration = start[0] + period - start
            exp_fading = np.exp(-duration)
            prev_in_ic0 = np.full_like(in_ic0[0], np.nan)
            prev_in_ic1 = np.full_like(in_ic1[0], np.nan)
            while not (
                np.allclose(prev_in_ic0, in_ic0[0])
                and np.allclose(prev_in_ic1, in_ic1[0])
            ):
                np.copyto(prev_in_ic0, in_ic0[0])
                np.copyto(prev_in_ic1, in_ic1[0])

                in_ic0[0] += np.sum(
                    (in_ic0 + in_ic1 * duration[:, None]) * exp_fading[:, None], axis=0
                )
                in_ic1[0] += np.sum(in_ic1 * exp_fading[:, None], axis=0)

                duration += period
                exp_fading *= np.exp(-period)

            # Scan the states in place (forward update)
            fscan_states(start, in_ic0, in_ic1, in_c0, in_c1)

            logger.debug(
                f"The time interval [0, {period}] has been partitioned into {start.size} intervals."
            )

            ## Silent time intervals: z <= z_max
            logger.debug(f"Silent zone is [0, {period}].")
            self.z_start.append(start)
            self.z_length.append(np.diff(start, append=period))
            self.z_lim.append(np.full_like(start, zmax))
            self.z_in_c0.append(in_c0)
            self.z_in_c1.append(in_c1)
            self.z_c0.append(np.full_like(start, np.nan))
            self.z_c1.append(np.full_like(start, np.nan))

        self.model.update()
        logger.info(
            f"Solver initialized with {self.n_constraints} constraints."
        )

    def refine_constraints(self) -> int:
        """
        Refine constraints by adding violated constraint templates.
        
        Checks current solution against potential constraint violations in
        silent and active zones, adding new constraints where violations exceed
        the feasibility tolerance.

        Returns
        -------
        int
            Constraint refinement status:
            * 1 : All constraints satisfied, solution is optimal
            * 0 : New constraints added, continue refinement
            * -1 : Refinement failed (currently unused)
            
        Notes
        -----
        The method:
        - Evaluates membrane potential using current optimal weights
        - Searches for maximum violations in silent zones (z <= zmax)
        - Searches for maximum violations in active zones (dz >= dzmin)  
        - Adds new constraints for violations exceeding FEAS_TOL
        - Uses the find_maximum_violation utility for efficient violation detection
        """
        flag = False  # Flag to indicate if a new constraint was added

        # self.weight = self.solver.x_value()
        for n in range(len(self.z_start)):
            # Potential constraints
            self.z_c0[n] = (
                np.inner(self.z_in_c0[n][:, :-1], self.weights.X)
                + self.z_in_c0[n][:, -1] * REFRACTORY_RESET
            )
            self.z_c1[n] = (
                np.inner(self.z_in_c1[n][:, :-1], self.weights.X)
                + self.z_in_c1[n][:, -1] * REFRACTORY_RESET
            )

            res = find_maximum_violation(
                self.z_c0[n], self.z_c1[n], self.z_length[n], self.z_lim[n]
            )

            if res is not None:
                vmax, imax, dtmax = res
                if vmax > FEAS_TOL:
                    self.model.addConstr(
                        (
                            self.z_in_c0[n][imax, :-1]
                            + dtmax * self.z_in_c1[n][imax, :-1]
                        )
                        * np.exp(-dtmax)
                        @ self.weights
                        <= self.z_lim[n][imax]
                        - REFRACTORY_RESET
                        * (
                            self.z_in_c0[n][imax, -1]
                            + dtmax * self.z_in_c1[n][imax, -1]
                        )
                        * np.exp(-dtmax),
                        name=f"zmax_{self.n_constraints + 1}",
                    )

                    logger.debug(
                        f"Constraint refinement: a violation of the potential constraint has been detected on the {n}th interval at t={self.z_start[n][imax] + dtmax}."
                    )
                    flag = True

        for n in range(len(self.dz_start)):
            # Potential derivative constraints
            self.dz_c0[n] = (
                np.inner(self.dz_in_c0[n][:, :-1], self.weights.X)
                + self.dz_in_c0[n][:, -1] * REFRACTORY_RESET
            )
            self.dz_c1[n] = (
                np.inner(self.dz_in_c1[n][:, :-1], self.weights.X)
                + self.dz_in_c1[n][:, -1] * REFRACTORY_RESET
            )

            res = find_maximum_violation(
                self.dz_c0[n], self.dz_c1[n], self.dz_length[n], self.dz_lim[n]
            )
            # dz_dvmax, dz_imax, dz_dtmax =
            if res is not None:
                vmax, imax, dtmax = res
                if vmax > FEAS_TOL:
                    self.model.addConstr(
                        (
                            self.dz_in_c0[n][imax, :-1]
                            + dtmax * self.dz_in_c1[n][imax, :-1]
                        )
                        * np.exp(-dtmax)
                        @ self.weights
                        <= self.dz_lim[n][imax]
                        - REFRACTORY_RESET
                        * (
                            self.dz_in_c0[n][imax, -1]
                            + dtmax * self.dz_in_c1[n][imax, -1]
                        )
                        * np.exp(-dtmax),
                        name=f"dzmin_{self.n_constraints + 1}",
                    )

                    logger.debug(
                        f"Constraint refinement: a violation of the potential derivative constraint has been detected on the {n}th interval at t={self.dz_start[n][imax] + dtmax}."
                    )
                    flag = True

        if not flag:
            logger.debug("Constraint refinement: all constraints are satisfied!")
            return 1

        return 0