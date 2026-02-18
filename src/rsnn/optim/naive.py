import gurobipy as gp
import numpy as np
import polars as pl

from rsnn import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.log import setup_logging
from rsnn.optim import GUROBI_STATUS
from rsnn.optim.utils import (
    compute_1d_linear_map,
    compute_nd_linear_map,
    find_max_violations,
    scan_with_weights,
)

# SLOPE_MARGIN = 1e-6  # Minimum slope at firing times
# LEVEL_MARGIN = 1e-6  # Maximum level when not firing


def compute_template(
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    zmax: float,
) -> pl.DataFrame:
    """Compute neuronal states for naive optimization. Note that the last state before each firing is not included as not needed for the optimization process.

    Args:
        syn_states (pl.DataFrame): Synaptic transmission states with columns 'f_index', 'start', and 'in_index'.
        rec_states (pl.DataFrame): Refractory states with columns 'f_index', 'start', and 'weight'.
        out_spikes (pl.DataFrame): Output spike data with columns 'index', 'f_index', 'time'.
        threshold (float): Neuronal firing threshold.
        dzmin (float): Minimum derivative at firing time.
        zmax (float): Maximum membrane potential at rest.

    Returns:
        pl.DataFrame: Neuronal states with columns 'f_index', 'in_index', 'start', 'length', 'weight', 'in_coef_0', 'in_coef_1', 'rhs_coef_0', and 'rhs_coef_1'. The DataFrame is sorted by time within each firing index group.
    """
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
    in_states = (
        pl.concat([rec_states, syn_states])
        .sort("start")
        .with_columns(
            pl.col("start").diff().shift(-1).over("f_index").alias("length"),
            pl.lit(zmax, pl.Float64).alias("rhs_coef_0"),
            pl.lit(0.0, pl.Float64).alias("rhs_coef_1"),
        )
        .drop_nulls("length")
    )

    return in_states

    # in_states = in_spikes.sort("time").select(
    #     pl.col("f_index"),
    #     pl.col("in_index"),
    #     pl.col("time").alias("start"),
    #     pl.col("time").diff().shift(-1).over("f_index").alias("length"),
    #     pl.when(pl.col("in_index").is_null())
    #     .then(reset)
    #     .otherwise(None)
    #     .alias("weight"),
    #     pl.when(pl.col("in_index").is_null())
    #     .then(1.0)
    #     .otherwise(0.0)
    #     .alias("in_coef_0"),
    #     pl.when(pl.col("in_index").is_null())
    #     .then(0.0)
    #     .otherwise(1.0)
    #     .alias("in_coef_1"),
    #     pl.lit(threshold - LEVEL_MARGIN, pl.Float64).alias("rhs_coef_0"),
    #     pl.lit(0.0, pl.Float64).alias("rhs_coef_1"),
    # )

    # return in_states.select(
    #     pl.all()
    #     .gather(pl.int_range(pl.len() - 1))
    #     .over("f_index", mapping_strategy="explode")
    # )


# def optimize(
#     neurons: pl.DataFrame,
#     synapses: pl.DataFrame,
#     spikes: pl.DataFrame,
#     periods: pl.DataFrame,
#     wmin: float = float("-inf"),
#     wmax: float = float("inf"),
#     n_iter: int = 1000,
#     logger=None,
# ) -> pl.DataFrame:
#     """Optimize synaptic weights using template-based iterative constraint refinement.

#     Performs iterative optimization of synaptic weights by alternating between weight optimization and constraint refinement.
#     Uses linear programming with template-based constraints to ensure proper firing behavior.

#     Args:
#         neurons (pl.DataFrame): Neuron parameters with columns 'neuron', 'threshold', 'reset'.
#         synapses (pl.DataFrame): Synaptic connections with columns 'source', 'target', 'delay', 'weight'.
#         spikes (pl.DataFrame): Spike train data with columns 'index', 'period', 'neuron', 'time'.
#         periods (pl.DataFrame): Period information with columns 'index', 'period'.
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
#             file_path="optim-naive.log",
#         )

#     if wmin >= wmax:
#         raise ValueError("wmin must be less than wmax.")

#     if neurons.height != neurons.unique("neuron").height:
#         raise ValueError("Neurons DataFrame contains duplicate neuron entries.")

#     synapses_lst = []

#     # for (neuron,), in_synapses in synapses.partition_by(
#     #     "target", maintain_order=False, as_dict=True
#     # ).items():
#     for id in neurons["neuron"]:
#         logger.debug(f"Optimizing neuron {id}...")

#         # Extract neuron information
#         neuron = neurons.filter(pl.col("neuron") == id)
#         threshold = neuron.select("threshold").item()
#         reset = neuron.select("reset").item()

#         out_spikes = spikes.filter(pl.col("neuron") == id).with_columns(
#             pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index")
#         )
#         in_synapses = synapses.filter(pl.col("target") == id).with_columns(
#             pl.int_range(pl.len(), dtype=pl.UInt32).alias("in_index")
#         )
#         in_spikes = spikes.join(
#             in_synapses, left_on="neuron", right_on="source"
#         ).select(
#             pl.col("index"),
#             pl.col("in_index"),
#             pl.col("time") + pl.col("delay"),
#             pl.col("weight"),
#         )

#         # init_synapses(synapses.filter(pl.col("target") == id), spikes.select("neuron"))
#         # out_spikes = init_out_spikes(spikes.filter(pl.col("neuron") == id))

#         ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
#         model = gp.Model("model")
#         model.setParam("OutputFlag", 0)  # Disable output

#         # Setup variables to be optimized = the synaptic weights
#         weights = model.addMVar(shape=in_synapses.height, lb=wmin, ub=wmax)
#         logger.debug(f"Neuron {id}. Learnable weights initialized.")

#         # Objective function - activity-agnostic
#         model.setObjective(
#             weights @ weights,
#             sense=gp.GRB.MINIMIZE,
#         )
#         logger.debug(f"Neuron {id}. Objective function set.")

#         ### TO CONTINUE HERE.-..

#         # Compute states for linear constraints
#         syn_states, rec_states, states = compute_states(
#             out_spikes, in_spikes, threshold, reset, periods
#         )

#         # Firing time constraints
#         lin_map = compute_linear_map(
#             in_synapses.height,
#             syn_states,
#             rec_states,
#             out_spikes.select("f_index", "time").with_row_index("t_index"),
#         )
#         model.addConstr(lin_map(weights) == threshold)
#         lin_map = compute_linear_map(
#             in_synapses.height,
#             syn_states,
#             rec_states,
#             out_spikes.select("f_index", "time").with_row_index("t_index"),
#             deriv=1,
#         )
#         model.addConstr(lin_map(weights) >= SLOPE_MARGIN)  # type: ignore
#         logger.debug(f"Neuron {id}. Firing time constraints added.")

#         for i in range(n_iter):
#             # 1. Optimize weights
#             model.optimize()

#             if model.status != gp.GRB.OPTIMAL:
#                 logger.error(
#                     f"Neuron {id}. Optimization failed: {GUROBI_STATUS[model.status]}"
#                 )
#                 return synapses.with_columns(pl.lit(None, pl.Float64).alias("weight"))

#             states = scan_with_new_weights(states, weights.X)  # type: ignore
#             # logger.info(states.filter(pl.col("coef_0") > FIRING_THRESHOLD))

#             # 2. Refine constraints
#             max_violations = find_max_violations(states)

#             if max_violations.height > 0:
#                 logger.info(
#                     f"Neuron {id}: iteration {i}. The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints; still {max_violations.height} violations to resolve."
#                 )
#                 max_violations = max_violations.with_row_index("t_index").with_columns(
#                     pl.col("tmax").alias("time")
#                 )
#                 logger.debug(
#                     f"\tNew support constraints at time(s): {max_violations.get_column('time').to_list()}."
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
#                     f"Neuron {id}: iteration {i}. The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints. No more violations."
#                 )
#                 synapses_lst.append(
#                     in_synapses.update(
#                         pl.DataFrame({"weight": weights.X}).with_row_index("in_index"),
#                         on="in_index",
#                     ).select(
#                         pl.lit(id).alias("source"),
#                         pl.col("target"),
#                         pl.col("delay"),
#                         pl.col("weight"),
#                     )
#                 )
#                 break

#     return pl.concat(synapses_lst)


def optimize(
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    out_spikes: pl.DataFrame,
    threshold: float = FIRING_THRESHOLD,
    # reset: float = REFRACTORY_RESET,
    wmin: float = float("-inf"),
    wmax: float = float("inf"),
    dztol: float = 1e-6,
    ztol: float = 1e-6,
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
        out_spikes (pl.DataFrame): Output spike data with columns 'f_index', 'time'. Must be sorted by time within each neuron group.
        in_spikes (pl.DataFrame): Input spike data with columns 'neuron', 'time', 'weight'. Must be sorted by time within each neuron group.
        periods (pl.DataFrame): Period information with columns 'index', 'period'.
        wmin (float, optional): Minimum synaptic weight bound. Defaults to -inf.
        wmax (float, optional): Maximum synaptic weight bound. Defaults to inf.
        n_iter (int, optional): Maximum number of constraint generation iterations. Defaults to 1000.
        logger (optional): Logger for logging messages. If None, a default logger is created.

    Returns:
        pl.DataFrame: Optimized synapses as a DataFrame with columns 'in_index' and 'weight'. If optimization fails, returns DataFrame with 'weight' set to None.

    Raises:
        ValueError: If eps < 0, zmax > FIRING_THRESHOLD, dzmin < 0, or wmin >= wmax.

    Notes:
        The optimization alternates between:
        1. Minimizing L2 norm of weights subject to current constraints
        2. Adding new constraints for detected violations in silent/active regions
        Converges when no more constraint violations are found.
    """
    if logger is None:
        logger = setup_logging(
            __name__,
            console_level="INFO",
            file_level="DEBUG",
            file_path="naive.log",
        )

    if wmin >= wmax:
        raise ValueError("wmin must be less than wmax.")

    if dztol < 0:
        raise ValueError("dztol must be non-negative.")

    if ztol < 0:
        raise ValueError("ztol must be non-negative.")

    n_synapses = syn_states.select(pl.col("in_index")).max().item() + 1

    # init_synapses(synapses.filter(pl.col("target") == id), spikes.select("neuron"))
    # out_spikes = init_out_spikes(spikes.filter(pl.col("neuron") == id))

    ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
    model = gp.Model("model")
    model.setParam("OutputFlag", 0)  # Disable output

    # Setup variables to be optimized = the synaptic weights
    weights = model.addMVar(shape=n_synapses, lb=wmin, ub=wmax)
    logger.debug("Learnable weights initialized.")

    # Objective function - activity-agnostic
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
                lin_map(weights) >= dztol,
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
        model.addConstr(lin_map(weights) >= dztol)
    logger.debug("Firing time constraints added.")

    states = compute_template(syn_states, rec_states, threshold - ztol)
    # init_states(in_spikes, threshold, reset)

    for i in range(n_iter):
        # 1. Optimize weights
        model.optimize()

        if model.status != gp.GRB.OPTIMAL:
            logger.error(
                f"Iteration {i}. Optimization failed: {GUROBI_STATUS[model.status]}"
            )
            break

        states = scan_with_weights(states, weights.X)  # type: ignore
        # logger.info(states.filter(pl.col("coef_0") > FIRING_THRESHOLD))

        # 2. Refine constraints
        max_violations = find_max_violations(states)

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
                model.addConstr(lin_map(weights) <= max_violations.bound.to_numpy())  # type: ignore

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
