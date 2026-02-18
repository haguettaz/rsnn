import gurobipy as gp
import numpy as np
import polars as pl

from rsnn import FIRING_THRESHOLD
from rsnn.log import setup_logging
from rsnn.optim import GUROBI_STATUS

# from rsnn.optim.nrg import get_nrg_matrix
from rsnn.optim.utils import (
    compute_1d_linear_map,
    compute_nd_linear_map,
    find_max_violations,
    get_nrg_matrix,
    scan_with_weights,
)


def compute_template(
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    out_spikes: pl.DataFrame,
    threshold: float,
    dzmin: float,
    zmax: float,
) -> pl.DataFrame:
    """Compute neuronal states for SVM optimization.

    Calculates the complete set of neuronal states including synaptic transmission events, refractory periods, and firing events.
    Combines all state types without temporal scanning.

    Args:
        syn_states (pl.DataFrame): Synaptic transmission states with columns 'f_index', 'start', and 'in_index'.
        rec_states (pl.DataFrame): Refractory states with columns 'f_index', 'start', and 'weight'.
        out_spikes (pl.DataFrame): Output spike data with columns 'index', 'f_index', 'time'.
        threshold (float): Neuronal firing threshold.
        dzmin (float): Minimum derivative at firing time.
        zmax (float): Maximum membrane potential at rest.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
            - syn_states: Synaptic transmission states with columns 'f_index', 'in_index', 'start'
            - rec_states: Refractory states with columns 'f_index', 'start'
            - states: All states with columns 'f_index', 'start', 'in_index', 'weight', 'in_coef_0', 'in_coef_1', 'active', 'length', and sorted by start time over each firing index group.
    """
    # Firing states
    f_states = out_spikes.select(
        pl.col("f_index"),
        pl.col("time").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(None, pl.Float64).alias("weight"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
        pl.lit(threshold, pl.Float64).alias("rhs_coef_0"),
        pl.lit(dzmin, pl.Float64).alias("rhs_coef_1"),
    )

    before_f_states = out_spikes.select(
        pl.col("f_index"),
        (pl.col("time") - (threshold - zmax) / dzmin).alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(None, pl.Float64).alias("weight"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
        pl.lit(zmax, pl.Float64).alias("rhs_coef_0"),
        pl.lit(dzmin, pl.Float64).alias("rhs_coef_1"),
    )

    rec_states = rec_states.select(
        pl.col("f_index"),
        pl.col("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.col("weight"),
        pl.lit(1.0, pl.Float64).alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    )
    syn_states = syn_states.select(
        pl.col("f_index"),
        pl.col("start"),
        pl.col("in_index"),
        pl.lit(None, pl.Float64).alias("weight"),
        pl.lit(0.0, pl.Float64).alias("in_coef_0"),
        pl.lit(1.0, pl.Float64).alias("in_coef_1"),
    )
    in_states = pl.concat([rec_states, syn_states]).with_columns(
        pl.lit(None, pl.Float64).alias("rhs_coef_0"),
        pl.lit(None, pl.Float64).alias("rhs_coef_1"),
    )

    states = (
        in_states.vstack(before_f_states)
        .vstack(f_states)
        .sort("start", maintain_order=True)
        .with_columns(
            pl.col("rhs_coef_1").forward_fill().fill_null(0.0).over("f_index"),
            pl.col("start")
            .diff()
            .shift(-1, fill_value=0.0)
            .over("f_index")
            .alias("length"),
        )
        .with_columns(
            (
                threshold
                - (pl.col("length") * pl.col("rhs_coef_1")).cum_sum(reverse=True)
            )
            .over("f_index")
            .alias("rhs_coef_0"),
        )
    )

    return states


def optimize(
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    out_spikes: pl.DataFrame,
    syn_params: pl.DataFrame,
    threshold: float = FIRING_THRESHOLD,
    dzmin: float = 2.0,
    zmax: float = 0.5,
    nrg: bool = False,
    diag: bool = True,
    n_iter: int = 1000,
    logger=None,
    return_model: bool = False,
    return_states: bool = False,
) -> (
    pl.DataFrame
    | tuple[pl.DataFrame, gp.Model]
    | tuple[pl.DataFrame, pl.DataFrame]
    | tuple[pl.DataFrame, gp.Model, pl.DataFrame]
):
    """Optimize synaptic weights using template-based iterative constraint refinement.

    Performs iterative optimization of synaptic weights by alternating between weight optimization and constraint refinement.
    Uses linear programming with template-based constraints to ensure proper firing behavior.

    Args:
        syn_states (pl.DataFrame): Synaptic transmission states with columns 'f_index', 'start', and 'in_index'.
        rec_states (pl.DataFrame): Refractory states with columns 'f_index', 'start', and 'weight'.
        out_spikes (pl.DataFrame): Output spike data with columns 'f_index', 'time'. Must be sorted by time within each neuron group.
        syn_params (pl.DataFrame): Synaptic parameters with columns 'in_index', 'min_weight', 'max_weight', and 'l2_reg' (optional).
        threshold (float, optional): Neuronal firing threshold. Defaults to FIRING_THRESHOLD.
        dzmin (float, optional): Minimum derivative at firing time. Defaults to 2.0.
        zmax (float, optional): Maximum membrane potential at rest. Defaults to 0.0.
        nrg (bool, optional): Whether to use energy-based objective. Defaults to False.
        n_iter (int, optional): Maximum number of constraints adaptation iterations. Defaults to 1000.
        logger (optional): Logger for logging messages. If None, a default logger is created.
        return_model (bool, optional): Whether to return the Gurobi model. Defaults to False.
        return_states (bool, optional): Whether to return the final neuronal states. Defaults to False.

    Returns:
        Optimized synapses as a DataFrame with columns 'in_index' and 'weight'. If optimization fails, returns DataFrame with 'weight' set to None.
            If return_model is True, returns a tuple (synapses, model).
            If return_states is True, returns a tuple (synapses, states).
            If both return_model and return_states are True, returns a tuple (synapses, model, states).

    Raises:
        ValueError: zmax > FIRING_THRESHOLD or dzmin <= 0.

    Notes:
        The optimization alternates between:
        1. Minimizing the objective (quadratic) of weights subject to current constraints.
        2. Adding new constraints for detected violations in silent/active regions.
        Converges when no more constraint violations are found.
    """
    if logger is None:
        logger = setup_logging(
            __name__,
            console_level="INFO",
            file_level="DEBUG",
            file_path="solver.log",
        )

    if dzmin <= 0:
        raise ValueError("dzmin must be positive.")

    if zmax >= threshold:
        raise ValueError("zmax must be lower than the threshold.")

    syn_params = syn_params.sort("in_index")
    n_synapses = syn_params.height

    ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
    model = gp.Model("qp-model")
    model.setParam("OutputFlag", 0)  # Disable output

    # Setup variables to be optimized = the synaptic weights
    weights = model.addMVar(
        shape=syn_params.height,
        lb=syn_params.select("min_weight").to_numpy().flatten(),
        ub=syn_params.select("max_weight").to_numpy().flatten(),
    )
    logger.debug("Learnable weights initialized.")

    if syn_params.columns.__contains__("l2_reg"):
        reg_matrix = np.diag(syn_params.select("l2_reg").to_numpy().flatten())  # type: ignore
    else:
        reg_matrix = None

    # Objective function
    if nrg:
        nrg_matrix = get_nrg_matrix(
            n_synapses,
            syn_states.join(
                out_spikes.select(pl.col("f_index"), pl.col("time").alias("end")),
                on="f_index",
            ),
            diag,
        )
        if reg_matrix is not None:
            model.setObjective(
                weights @ (nrg_matrix + reg_matrix) @ weights, sense=gp.GRB.MINIMIZE
            )
        else:
            model.setObjective(weights @ nrg_matrix @ weights, sense=gp.GRB.MINIMIZE)

    elif reg_matrix is not None:
        model.setObjective(weights @ reg_matrix @ weights, sense=gp.GRB.MINIMIZE)
    else:
        model.setObjective(weights @ weights, sense=gp.GRB.MINIMIZE)

    logger.debug("Objective function set.")

    # Firing time constraints
    if return_model:
        for time in out_spikes.select("f_index", "time").iter_rows(named=True):
            # Threshold crossing
            lin_map = compute_1d_linear_map(
                n_synapses,
                syn_states,
                rec_states,
                time,
            )
            model.addConstr(
                lin_map(weights) == threshold,
                name=f"ft-{time['f_index']}-{time['time']}",
            )
            # With minimum slope
            lin_map = compute_1d_linear_map(
                n_synapses,
                syn_states,
                rec_states,
                time,
                deriv=1,
            )
            model.addConstr(
                lin_map(weights) >= dzmin,
                name=f"dft-{time['f_index']}-{time['time']}",
            )
    else:
        # Threshold crossing
        lin_map = compute_nd_linear_map(
            n_synapses,
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time").with_row_index("t_index"),
        )
        model.addConstr(lin_map(weights) == threshold)
        # With minimum slope
        lin_map = compute_nd_linear_map(
            n_synapses,
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time").with_row_index("t_index"),
            deriv=1,
        )
        model.addConstr(lin_map(weights) >= dzmin)
    logger.debug("Firing time constraints added.")

    states = compute_template(
        syn_states, rec_states, out_spikes, threshold, dzmin, zmax
    )

    for i in range(n_iter):
        # 1. Optimize weights
        model.optimize()

        if model.status not in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL}:
            logger.debug(model.status)
            logger.error(
                f"Iteration {i}. Optimization failed: {GUROBI_STATUS[model.status]}"
            )
            break

        if model.status == gp.GRB.SUBOPTIMAL:
            logger.warning(f"Iteration {i}. Optimization suboptimal.")

        states = scan_with_weights(states, weights.X)  # type: ignore
        # logger.info(states.filter(pl.col("coef_0") > FIRING_THRESHOLD))

        # 2. Refine constraints
        max_violations = find_max_violations(states)
        logger.debug(f"Max violations:\n{max_violations}")

        if max_violations.height > 0:
            logger.info(
                f"Iteration {i}. The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints; still {max_violations.height} violations to resolve."
            )
            max_violations = max_violations.rename({"tmax": "time"})

            if return_model:
                for violation in max_violations.iter_rows(named=True):
                    lin_map = compute_1d_linear_map(
                        n_synapses,
                        syn_states,
                        rec_states,
                        violation,
                    )
                    model.addConstr(
                        lin_map(weights) <= violation["bound"],
                        name=f"mv-{violation["f_index"]}-{violation["time"]}-{violation["bound"]}",
                    )

            else:
                lin_map = compute_nd_linear_map(
                    n_synapses,
                    syn_states,
                    rec_states,
                    max_violations.with_row_index("t_index"),
                )
                model.addConstr(lin_map(weights) <= max_violations["bound"].to_numpy())  # type: ignore

            logger.debug("Maximum violation constraints added.")

        else:
            logger.info(
                f"Iteration {i}. The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints. No more violations."
            )

            synapses = pl.DataFrame(
                data={
                    "in_index": np.arange(n_synapses, dtype=np.uint32),
                    "weight": weights.X,
                },
                schema={"in_index": pl.UInt32, "weight": pl.Float64},
            )

            if return_model and return_states:
                return synapses, model, states

            if return_model:
                return synapses, model

            if return_states:
                return synapses, states

            return synapses

    synapses = pl.DataFrame(schema={"in_index": pl.UInt32, "weight": pl.Float64})

    if return_model and return_states:
        return synapses, model, states

    if return_model:
        return synapses, model

    if return_states:
        return synapses, states

    return synapses


###


# def compute_states(
#     in_spikes: pl.DataFrame,
#     out_spikes: pl.DataFrame,
#     threshold: float,
#     reset: float,
#     dzmin: float,
# ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
#     """Compute neuronal states for template-based optimization.

#     Calculates different types of states needed for the template-based optimization: firing states, synaptic transmission states, and refractory states. Also computes before-firing states for constraint violations.

#     Args:
#         synapses (pl.DataFrame): Synaptic connections with columns including 'source', 'target', 'delay'.
#         out_spikes (pl.DataFrame): Output spike times with columns 'index', 'period', 'neuron', 'f_index', 'time', 'time_prev'. Must be sorted by time within each (index, neuron) group.
#         src_spikes (pl.DataFrame): Source spike times with columns 'index', 'period', 'neuron', 'time'.
#         eps (float): Epsilon parameter for before-firing state timing.

#     Returns:
#         tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
#             - syn_states: Synaptic transmission states with columns 'f_index', 'in_index', 'start'
#             - rec_states: Refractory states with columns 'f_index', 'start'
#             - states: All states with columns 'f_index', 'start', 'in_index', 'weight', 'in_coef_0', 'in_coef_1', 'active', 'length', and sorted by start time over each firing index group.
#     """

#     if dzmin <= 0.0:
#         raise ValueError("dzmin must be positive.")

#     # Firing states
#     f_states = out_spikes.select(
#         pl.col("f_index"),
#         pl.col("time").alias("start"),
#         pl.lit(None, pl.UInt32).alias("in_index"),
#         pl.lit(None, pl.Float64).alias("weight"),
#         pl.lit(None, pl.Float64).alias("in_coef_0"),
#         pl.lit(None, pl.Float64).alias("in_coef_1"),
#         # pl.lit(False, pl.Boolean).alias("constr"),
#     )

#     before_f_states = out_spikes.select(
#         pl.col("f_index"),
#         (pl.col("time") - FIRING_THRESHOLD / dzmin)
#         .clip(pl.col("time_prev"))
#         .alias("start"),
#         pl.lit(None, pl.UInt32).alias("in_index"),
#         pl.lit(None, pl.Float64).alias("weight"),
#         pl.lit(None, pl.Float64).alias("in_coef_0"),
#         pl.lit(None, pl.Float64).alias("in_coef_1"),
#         # pl.lit(True, pl.Boolean).alias("active"),
#         pl.lit(None, pl.Float64).alias("rhs_coef_0"),
#         pl.lit(dzmin, pl.Float64).alias("rhs_coef_1"),
#     )

#     in_states = in_spikes.select(
#         pl.col("f_index"),
#         pl.col("in_index"),
#         pl.col("time").alias("start"),
#         pl.when(pl.col("in_index").is_null())
#         .then(reset)
#         .otherwise(None)
#         .alias("weight"),
#         pl.when(pl.col("in_index").is_null())
#         .then(1.0)
#         .otherwise(0.0)
#         .alias("in_coef_0"),
#         pl.when(pl.col("in_index").is_null())
#         .then(0.0)
#         .otherwise(1.0)
#         .alias("in_coef_1"),
#     )

#     states = (
#         in_states.with_columns(
#             pl.lit(None, pl.Float64).alias("rhs_coef_0"),
#             pl.lit(None, pl.Float64).alias("rhs_coef_1"),
#         )
#         .vstack(before_f_states)
#         .vstack(
#             f_states.with_columns(
#                 pl.lit(None, pl.Float64).alias("rhs_coef_0"),
#                 pl.lit(None, pl.Float64).alias("rhs_coef_1"),
#             )
#         )
#         .sort("start")
#         .with_columns(
#             pl.col("rhs_coef_1").forward_fill().fill_null(0.0).over("f_index"),
#             pl.col("start")
#             .diff()
#             .shift(-1, fill_value=0.0)
#             .over("f_index")
#             .alias("length"),
#         )
#         .with_columns(
#             (
#                 FIRING_THRESHOLD
#                 - (pl.col("length") * pl.col("rhs_coef_1")).cum_sum(reverse=True)
#             )
#             .over("f_index")
#             .alias("rhs_coef_0"),
#         )
#     )

#     return (
#         in_states.filter(pl.col("in_index").is_not_null()).select(
#             "f_index", "start", "in_index"
#         ),
#         in_states.filter(pl.col("in_index").is_null()).select("f_index", "start"),
#         states,  # .filter(pl.col("length") > 0.0),
#     )


# def optimize(
#     in_spikes: pl.DataFrame,
#     out_spikes: pl.DataFrame,
#     threshold: float = FIRING_THRESHOLD,
#     reset: float = REFRACTORY_RESET,
#     wmin: float = float("-inf"),
#     wmax: float = float("inf"),
#     dzmin: float = 1.0,
#     n_iter: int = 1000,
#     logger=None,
#     return_support: bool = False,
# ) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
#     """Optimize synaptic weights using template-based iterative constraint refinement.

#     Performs iterative optimization of synaptic weights by alternating between weight optimization and constraint refinement.
#     Uses linear programming with template-based constraints to ensure proper firing behavior.

#     Args:
#         synapses (pl.DataFrame): Synaptic connections with columns 'source', 'target', 'delay', 'weight'.
#         spikes (pl.DataFrame): Spike train data with columns 'index', 'period', 'neuron', 'time'.
#         wmin (float, optional): Minimum synaptic weight bound. Defaults to -inf.
#         wmax (float, optional): Maximum synaptic weight bound. Defaults to inf.
#         eps (float, optional): Epsilon parameter for before-firing constraints. Defaults to 0.2.
#         zmax (float, optional): Maximum membrane potential in silent regions. Defaults to 0.0.
#         dzmin (float, optional): Minimum derivative in active regions for sufficient rise. Defaults to 1.0.
#         n_iter (int, optional): Maximum number of constraint generation iterations. Defaults to 1000.
#         feas_tol (float, optional): Feasibility tolerance for constraint violations. Defaults to 1e-5.

#     Returns:
#         pl.DataFrame: Optimized synapses as a DataFrame with columns 'source', 'target', 'delay', 'weight'.
#             If optimization fails for any neuron, returns synapses with 'weight' set to None.

#     Raises:
#         ValueError: If eps < 0, zmax > FIRING_THRESHOLD, dzmin < 0, or wmin >= wmax.

#     Notes:
#         The optimization alternates between:
#         1. Minimizing L2 norm of weights subject to current constraints
#         2. Adding new constraints for detected violations in silent/active regions
#         Converges when no more constraint violations are found.
#     """
#     if logger is None:
#         logger = setup_logging(
#             __name__,
#             console_level="INFO",
#             file_level="DEBUG",
#             file_path="optim-svm.log",
#         )

#     if wmin >= wmax:
#         raise ValueError("wmin must be less than wmax.")

#     synapses_lst = []

#     for (neuron,), in_synapses in synapses.partition_by(
#         "target", maintain_order=False, as_dict=True
#     ).items():
#         logger.debug(f"Optimizing neuron {neuron}...")
#         # Prepare synapses and output spikes for the current neuron
#         in_synapses = init_synapses(in_synapses, spikes.select("neuron"))
#         out_spikes = init_out_spikes(spikes.filter(pl.col("neuron") == neuron))

#         ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
#         model = gp.Model("model")
#         model.setParam("OutputFlag", 0)  # Disable output

#         # Setup variables to be optimized = the synaptic weights
#         weights = model.addMVar(shape=in_synapses.height, lb=wmin, ub=wmax)
#         logger.debug(f"Neuron {neuron}. Learnable weights initialized.")

#         # Objective function - activity-agnostic
#         model.setObjective(
#             weights @ weights,
#             sense=gp.GRB.MINIMIZE,
#         )
#         logger.debug(f"Neuron {neuron}. Objective function set.")

#         # Compute states for linear constraints
#         syn_states, rec_states, states = compute_states(
#             in_synapses, out_spikes, spikes, dzmin
#         )

#         # Firing time constraints
#         lin_map = compute_linear_map(
#             in_synapses.height,
#             syn_states,
#             rec_states,
#             out_spikes.select("f_index", "time").with_row_index("t_index"),
#         )
#         model.addConstr(lin_map(weights) == FIRING_THRESHOLD)

#         lin_map = compute_linear_map(
#             in_synapses.height,
#             syn_states,
#             rec_states,
#             out_spikes.select("f_index", "time").with_row_index("t_index"),
#             deriv=1,
#         )
#         model.addConstr(lin_map(weights) >= dzmin)  # type: ignore

#         logger.debug(f"Neuron {neuron}. Firing time constraints added.")

#         for i in range(n_iter):
#             # 1. Optimize weights
#             model.optimize()

#             if model.status != gp.GRB.OPTIMAL:
#                 logger.error(
#                     f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
#                 )
#                 return synapses.with_columns(pl.lit(None, pl.Float64).alias("weight"))

#             states = scan_with_new_weights(states, weights.X)  # type: ignore
#             # logger.info(states.filter(pl.col("coef_0") > FIRING_THRESHOLD))

#             # 2. Refine constraints
#             max_violations = find_max_violations(states)
#             logger.info(
#                 f"Neuron {neuron}: iteration {i}. The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints; still {max_violations.height} violations to resolve."
#             )
#             # logger.info(max_violations)

#             if max_violations.height > 0:
#                 max_violations = max_violations.with_row_index("t_index").with_columns(
#                     pl.col("tmax").alias("time")
#                 )
#                 lin_map = compute_linear_map(
#                     in_synapses.height,
#                     syn_states,
#                     rec_states,
#                     max_violations,
#                 )
#                 zmax = max_violations.get_column("bound").to_numpy()  # type: ignore
#                 model.addConstr(lin_map(weights) <= zmax)  # type: ignore

#             else:
#                 logger.info(
#                     f"Neuron {neuron}. Optimization successful (in {i} iterations)"
#                 )
#                 synapses_lst.append(
#                     in_synapses.update(
#                         pl.DataFrame({"weight": weights.X}).with_row_index("in_index"),
#                         on="in_index",
#                     ).drop("in_index")
#                 )
#                 break

#     return pl.concat(synapses_lst)
