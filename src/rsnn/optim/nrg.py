from typing import Callable

import gurobipy as gp
import numpy as np
import polars as pl

import rsnn_plugin as rp
from rsnn import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.log import setup_logging
from rsnn.optim import GUROBI_STATUS
from rsnn.optim.utils import compute_1d_linear_map  # init_out_spikes,
from rsnn.optim.utils import (
    compute_nd_linear_map,
    dataframe_to_1d_array,
    dataframe_to_2d_array,
    init_synapses,
    modulo_with_offset,
)
from rsnn.rand import synapses


def compute_states(synapses, out_spikes, src_spikes):
    """Compute input states for synaptic transmission and refractoriness.

    Calculates state transitions for neuronal dynamics including synaptic inputs and refractory periods. The function processes synaptic transmission events and refractory reset states separately, then combines them.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns including 'in_index', 'source', 'target', 'delay', 'weight'.
        out_spikes (pl.DataFrame): Output spike times with columns including 'index', 'period', 'neuron', 'f_index', 'time', 'time_prev'. Must be sorted by time within each (index, neuron) group.
        src_spikes (pl.DataFrame): Source spike times with columns including 'index', 'period', 'neuron', 'time'.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
            - syn_states: Synaptic input states with columns 'f_index', 'in_index', 'start'
            - rec_states: Refractory states with columns 'f_index', 'start'
    """
    # Set the time origins per index
    origins = out_spikes.group_by(["index", "neuron"]).agg(
        pl.first("time_prev").alias("time_origin")
    )

    # Refractoriness
    rec_states = out_spikes.select(
        pl.col("index"),
        pl.col("neuron"),
        pl.col("f_index"),
        pl.col("time_prev").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
    )

    # Synaptic transmission
    syn_states = (
        origins.join(synapses, left_on="neuron", right_on="target")
        .join(src_spikes, left_on=["index", "source"], right_on=["index", "neuron"])
        .select(
            pl.col("index"),
            pl.col("neuron"),
            pl.lit(None, pl.UInt32).alias("f_index"),
            modulo_with_offset(
                pl.col("time") + pl.col("delay"),
                pl.col("period"),
                pl.col("time_origin"),
            ).alias("start"),
            pl.col("in_index"),
        )
    )

    # All input states
    in_states = (
        syn_states.extend(rec_states)
        .sort("index", "neuron", "start")
        .select(
            pl.col("f_index").forward_fill().over(["index", "neuron"]),
            pl.col("start"),
            pl.col("in_index"),
        )
    )

    return (
        in_states.filter(pl.col("in_index").is_not_null()),
        in_states.filter(pl.col("in_index").is_null()).drop("in_index"),
    )


# def get_diag_nrg_1st(syn_states: pl.DataFrame, last_only: bool):
#     """
#     Args:
#         syn_states (pl.DataFrame): with column 'f_index' containing the next firing time (end integration time)
#         which (_type_, optional): _description_. Defaults to None.

#     Raises:
#         ValueError: _description_
#     """
#     if last_only:
#         syn_states = (
#             syn_states.group_by(["f_index", "end", "in_index"])
#             .agg(pl.all().top_k_by("start", 1))
#             .explode(pl.exclude(["f_index", "end", "in_index"]))  # type: ignore
#         )

#     return (
#         syn_states.join(syn_states, on=["f_index", "end", "in_index"])
#         .group_by("in_index")
#         .agg(
#             rp.inner_syn_1st(pl.col("start"), pl.col("start_right"), pl.col("end"))
#             .sum()
#             .alias("energy")
#         )
#         .with_columns(pl.col("in_index").alias("in_index_right"))
#     )


# def get_diag_nrg_2nd(syn_states: pl.DataFrame, last_only: bool):
#     """
#     Args:
#         syn_states (pl.DataFrame): with column 'f_index' containing the next firing time (end integration time)
#         which (_type_, optional): _description_. Defaults to None.

#     Raises:
#         ValueError: _description_
#     """
#     if last_only:
#         syn_states = (
#             syn_states.group_by(["f_index", "end", "in_index"])
#             .agg(pl.all().top_k_by("start", 1))
#             .explode(pl.exclude(["f_index", "end", "in_index"]))  # type: ignore
#         )

#     return (
#         syn_states.join(syn_states, on=["f_index", "end", "in_index"])
#         .group_by("in_index")
#         .agg(
#             rp.inner_syn_2nd(pl.col("start"), pl.col("start_right"), pl.col("end"))
#             .sum()
#             .alias("energy")
#         )
#         .with_columns(pl.col("in_index").alias("in_index_right"))
#     )


# def get_full_nrg_1st(syn_states: pl.DataFrame, last_only: bool):
#     """
#     Args:
#         syn_states (pl.DataFrame): with column 'end' and 'neuron'
#         last_only

#     Raises:
#         ValueError: _description_
#     """
#     if last_only:
#         # syn_states = syn_states.group_by(["f_index", "end", "in_index"]).agg(
#         #     pl.max("start").alias("start")
#         # )

#         syn_states = (
#             syn_states.group_by(["f_index", "end", "in_index"])
#             .agg(pl.all().top_k_by("start", 1))
#             .explode(pl.exclude(["f_index", "end", "in_index"]))  # type: ignore
#         )

#     return (
#         syn_states.join(syn_states, on=["f_index", "end"])
#         .group_by(["in_index", "in_index_right"])
#         .agg(
#             rp.inner_syn_1st(pl.col("start"), pl.col("start_right"), pl.col("end"))
#             .sum()
#             .alias("energy")
#         )
#     )


# def get_full_nrg_2nd(syn_states: pl.DataFrame, last_only: bool):
#     """
#     Args:
#         syn_states (pl.DataFrame): with column 'f_index' containing the next firing time (end integration time)
#         which (_type_, optional): _description_. Defaults to None.

#     Warnings:
#         Only one end per f_index

#     Raises:
#         ValueError: _description_
#     """
#     if last_only:
#         syn_states = (
#             syn_states.group_by(["f_index", "in_index"])
#             .agg(pl.all().top_k_by("start", 1))
#             .explode(pl.exclude(["f_index", "in_index"]))  # type: ignore
#         )

#     return (
#         syn_states.join(syn_states, on=["f_index", "end"])
#         .group_by(["in_index", "in_index_right"])
#         .agg(
#             rp.inner_syn_2nd(pl.col("start"), pl.col("start_right"), pl.col("end"))
#             .sum()
#             .alias("energy")
#         )
#     )


def get_nrg_matrix(
    n_synapses: int,
    syn_states: pl.DataFrame,
    l2_reg: float,
    # first_order: bool = False,
    # full: bool = True,
    # last_only: bool = False,
) -> Callable[[gp.MVar], gp.MQuadExpr]:
    """Compute energy-based metrics for synaptic weights optimization.

    Calculates the weighted mean (linear term) and precision matrix (quadratic term) for the energy-based objective function.
    The metric can be computed either synapse-locally (diagonal) or neuron-locally (full matrix).

    Args:
        n_synapses (int): Number of synapses to optimize.
        out_spikes (pl.DataFrame): Output spikes containing 'f_index', 'time'.
        syn_states (pl.DataFrame): Synaptic states containing 'f_index', 'in_index', 'start', and 'end'.
        diag (bool, optional): If True, use diagonal (synapse-local) quadratic term in the objective. Otherwise, use full quadratic term. Defaults to False.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 1.0.

    Returns:
        Callable[[gp.MVar], gp.MQuadExpr]: A quadratic function of the synaptic weights.
    """

    syn_energies = (
        syn_states.join(syn_states, on=["f_index", "end"])
        .group_by(["in_index", "in_index_right"])
        .agg(
            rp.inner_syn_2nd(pl.col("start"), pl.col("start_right"), pl.col("end"))
            .sum()
            .alias("energy")
        )
    )
    Q = dataframe_to_2d_array(
        syn_energies,
        "in_index",
        "in_index_right",
        "energy",
        (n_synapses, n_synapses),
    ) + l2_reg * np.eye(n_synapses)

    # else:
    #     if first_order:
    #         syn_energies = get_diag_nrg_1st(syn_states, last_only=last_only)
    #     else:
    #         syn_energies = get_diag_nrg_2nd(syn_states, last_only=last_only)

    #     Q = dataframe_to_2d_array(
    #         syn_energies,
    #         "in_index",
    #         "in_index",
    #         "energy",
    #         (n_synapses, n_synapses),
    #     )

    return lambda x_: x_ @ Q @ x_


def optimize(
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    out_spikes: pl.DataFrame,
    syn_bounds: pl.DataFrame,
    threshold: float = FIRING_THRESHOLD,
    # first_order=False,
    # last_only=False,
    # full=True,
    l2_reg=1e-3,
    dzmin=float("-inf"),
    logger=None,
    return_model: bool = False,
):
    """Optimize synaptic weights using energy-based quadratic programming.

    Performs constrained optimization of synaptic weights to minimize an energy-based
    objective function while satisfying firing constraints. The optimization can use
    different energy metrics based on the specified parameters.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns including 'source', 'target', 'delay'.
        spikes (pl.DataFrame): Spike train data with columns 'index', 'neuron', 'time'.
        wmin (float, optional): Minimum synaptic weight bound. Defaults to -inf.
        wmax (float, optional): Maximum synaptic weight bound. Defaults to inf.
        diag_obj (bool, optional): If True, use diagonal (synapse-local) quadratic term in the objective. Otherwise, use full quadratic term. Defaults to False.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 1e-1.
        dzmin (float, optional): Minimum derivative constraint for sufficient rise at firing times. Defaults to 1e-6.

    Returns:
        pl.DataFrame: Optimized synapses as a DataFrame with columns 'source', 'target', 'delay', 'weight'. If optimization fails for any neuron, returns original synapses with 'weight' set to None.

    Raises:
        ValueError: If eps < 0, zmax > FIRING_THRESHOLD, dzmin < 0, or wmin >= wmax.
    """

    if logger is None:
        logger = setup_logging(
            __name__,
            console_level="INFO",
            file_level="DEBUG",
            file_path="nrg.log",
        )

    syn_bounds = syn_bounds.sort("in_index")
    n_synapses = syn_bounds.height

    ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
    model = gp.Model("qp-model")
    model.setParam("OutputFlag", 0)  # Disable output

    # Setup variables to be optimized = the synaptic weights
    weights = model.addMVar(
        shape=syn_bounds.height,
        lb=syn_bounds.select("min_weight").to_numpy().flatten(),
        ub=syn_bounds.select("max_weight").to_numpy().flatten(),
    )
    logger.debug("Learnable weights initialized.")

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

    # Objective: minimizse the norm induced by the energy metric
    quad_objective = get_nrg_matrix(
        n_synapses,
        syn_states.join(
            out_spikes.select(pl.col("f_index"), pl.col("time").alias("end")),
            on="f_index",
        ),
        l2_reg,
    )
    logger.debug("Energy metric computed.")
    model.setObjective(quad_objective(weights), sense=gp.GRB.MINIMIZE)
    logger.debug("Objective function set.")

    model.optimize()

    if model.status != gp.GRB.OPTIMAL:
        logger.error(f"Optimization failed: {GUROBI_STATUS[model.status]}")
        synapses = pl.DataFrame(schema={"in_index": pl.UInt32, "weight": pl.Float64})

    else:
        logger.info(
            f"The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints."
        )

        synapses = pl.DataFrame(
            data={
                "in_index": np.arange(n_synapses, dtype=np.uint32),
                "weight": weights.X,
            },
            schema={"in_index": pl.UInt32, "weight": pl.Float64},
        )

    if return_model:
        return synapses, model

    return synapses


def q_optimize(
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    out_spikes: pl.DataFrame,
    threshold: float = FIRING_THRESHOLD,
    wb: float = float("inf"),
    n_lvl: int = 3,
    l2_reg=1e-1,
    dzmin=float("-inf"),
    logger=None,
    return_model: bool = False,
):
    """Optimize synaptic weights using energy-based quadratic programming.

    Performs constrained optimization of synaptic weights to minimize an energy-based
    objective function while satisfying firing constraints. The optimization can use
    different energy metrics based on the specified parameters.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns including 'source', 'target', 'delay'.
        spikes (pl.DataFrame): Spike train data with columns 'index', 'neuron', 'time'.
        wmin (float, optional): Minimum synaptic weight bound. Defaults to -inf.
        wmax (float, optional): Maximum synaptic weight bound. Defaults to inf.
        diag_obj (bool, optional): If True, use diagonal (synapse-local) quadratic term in the objective. Otherwise, use full quadratic term. Defaults to False.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 1e-1.
        dzmin (float, optional): Minimum derivative constraint for sufficient rise at firing times. Defaults to 1e-6.

    Returns:
        pl.DataFrame: Optimized synapses as a DataFrame with columns 'source', 'target', 'delay', 'weight'. If optimization fails for any neuron, returns original synapses with 'weight' set to None.

    Raises:
        ValueError: If eps < 0, zmax > FIRING_THRESHOLD, dzmin < 0, or wmin >= wmax.
    """

    if logger is None:
        logger = setup_logging(
            __name__,
            console_level="INFO",
            file_level="DEBUG",
            file_path="svm.log",
        )

    if wb < 0.0:
        raise ValueError("wb must be non-negative.")

    if n_lvl % 2 == 0 or n_lvl < 3:
        raise ValueError("n_lvl must be an odd integer greater than or equal to 3.")

    n_synapses = syn_states.select(pl.col("in_index")).max().item() + 1

    ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
    model = gp.Model("qp-model")
    model.setParam("OutputFlag", 0)  # Disable output

    # Setup variables to be optimized = the synaptic weights
    scale = 2.0 * wb / (n_lvl - 1)
    qwb = n_lvl // 2
    weights = model.addMVar(shape=n_synapses, lb=-qwb, ub=qwb, vtype=gp.GRB.INTEGER)
    logger.debug("Learnable weights initialized.")

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
                lin_map(weights * scale) >= threshold,
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
                lin_map(weights * scale) >= dzmin,
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
        model.addConstr(lin_map(weights * scale) >= threshold)
        # With minimum slope
        lin_map = compute_nd_linear_map(
            n_synapses,
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time").with_row_index("t_index"),
            deriv=1,
        )
        model.addConstr(lin_map(weights * scale) >= dzmin)
    logger.debug("Firing time constraints added.")

    # Objective: minimizse the norm induced by the energy metric
    quad_objective = get_nrg_matrix(
        n_synapses,
        syn_states.join(
            out_spikes.select(pl.col("f_index"), pl.col("time").alias("end")),
            on="f_index",
        ),
        l2_reg,
    )
    logger.debug("Energy metric computed.")
    model.setObjective(quad_objective(weights), sense=gp.GRB.MINIMIZE)
    logger.debug("Objective function set.")

    model.optimize()

    if model.status != gp.GRB.OPTIMAL:
        logger.error(f"Optimization failed: {GUROBI_STATUS[model.status]}")
        synapses = pl.DataFrame(schema={"in_index": pl.UInt32, "weight": pl.Float64})

    else:
        logger.info(
            f"The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints."
        )

        synapses = pl.DataFrame(
            data={
                "in_index": np.arange(n_synapses, dtype=np.uint32),
                "weight": weights.X * scale,
            },
            schema={"in_index": pl.UInt32, "weight": pl.Float64},
        )

    if return_model:
        return synapses, model

    return synapses

    # lin_map = compute_linear_map(
    #     n_synapses,
    #     syn_states,
    #     rec_states,
    #     out_spikes.select("f_index", "time").with_row_index("t_index"),
    # )
    # model.addConstr(lin_map(weights) == FIRING_THRESHOLD)

    # if dzmin is not None:
    #     lin_map = compute_linear_map(
    #         n_synapses,
    #         syn_states,
    #         rec_states,
    #         out_spikes.select("f_index", "time").with_row_index("t_index"),
    #         deriv=1,
    #     )
    #     model.addConstr(lin_map(weights) >= dzmin)  # type: ignore
    #     logger.debug("Firing constraints added.")

    # for (neuron,), in_synapses in synapses.partition_by("target", as_dict=True).items():
    #     # Prepare synapses and output spikes for the current neuron
    #     in_synapses = init_synapses(in_synapses, spikes.select("neuron"))
    #     out_spikes = init_out_spikes(spikes.filter(pl.col("neuron") == neuron))

    #     syn_states, rec_states = compute_states(in_synapses, out_spikes, spikes)

    #     # Objective: to minimize an energy-based metric -- todo: optimize!!
    #     quad_objective = get_nrg_matrix(
    #         in_synapses.height,
    #         syn_states.join(
    #             out_spikes.select(pl.col("f_index"), pl.col("time").alias("end")),
    #             on="f_index",
    #         ),
    #         first_order,
    #         full,
    #         last_only,
    #         l2_reg,
    #     )
    #     logger.debug(f"Neuron {neuron}. Energy metrics computed.")
    #     model.setObjective(
    #         quad_objective(weights),
    #         sense=gp.GRB.MINIMIZE,
    #     )
    #     logger.debug(f"Neuron {neuron}. Objective function set.")

    #     # Firing constraints: exact crossing and sufficient rise
    #     lin_map = compute_linear_map(
    #         in_synapses.height,
    #         syn_states,
    #         rec_states,
    #         out_spikes.select("f_index", "time").with_row_index("t_index"),
    #     )
    #     model.addConstr(lin_map(weights) == FIRING_THRESHOLD)

    #     lin_map = compute_linear_map(
    #         in_synapses.height,
    #         syn_states,
    #         rec_states,
    #         out_spikes.select("f_index", "time").with_row_index("t_index"),
    #         deriv=1,
    #     )
    #     model.addConstr(lin_map(weights) >= dzmin)  # type: ignore
    #     logger.debug(f"Neuron {neuron}. Firing constraints added.")

    #     model.optimize()
    #     if model.status != gp.GRB.OPTIMAL:
    #         logger.error(
    #             f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
    #         )
    #         return synapses.with_columns(pl.lit(None, pl.Float64).alias("weight"))

    #     logger.info(f"Neuron {neuron}. Optimization completed!")
    #     synapses_lst.append(
    #         in_synapses.update(
    #             pl.DataFrame({"weight": weights.X}).with_row_index("in_index"),
    #             on="in_index",
    #         ).drop("in_index")
    #     )

    # return pl.concat(synapses_lst)
