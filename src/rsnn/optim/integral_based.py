from typing import Callable

import gurobipy as gp
import numpy as np
import polars as pl

import rsnn_plugin as rp
from rsnn import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.log import setup_logging
from rsnn.optim import GUROBI_STATUS
from rsnn.optim.utils import (
    compute_linear_map,
    dataframe_to_1d_array,
    init_out_spikes,
    init_synapses,
    modulo_with_offset,
)

logger = setup_logging(__name__, console_level="INFO", file_level="DEBUG")


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


def compute_objective(
    n_synapses: int,
    out_spikes: pl.DataFrame,
    syn_states: pl.DataFrame,
    l2_reg: float = 1.0,
) -> Callable[[gp.MVar], gp.MQuadExpr]:
    """Compute energy-based metrics for synaptic weights optimization.

    Calculates the weighted mean (linear term) and precision matrix (quadratic term) for the energy-based objective function. The metric can be computed either synapse-locally (diagonal) or neuron-locally (full matrix).

    Args:
        n_synapses (int): Number of synapses to optimize.
        out_spikes (pl.DataFrame): Output spikes containing 'f_index', 'time'.
        syn_states (pl.DataFrame): Synaptic states containing 'f_index', 'in_index', 'start'.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 1.0.

    Returns:
        Callable[[gp.MVar], gp.MQuadExpr]: A quadratic function of the synaptic weights.
    """
    lin_map = dataframe_to_1d_array(
        syn_states.join(out_spikes, on="f_index")
        .group_by("in_index")
        .agg(
            rp.integrate_synaptic_input(pl.col("time") - pl.col("start"))
            .sum()
            .alias("value")
        ),
        "in_index",
        "value",
        n_synapses,
    )

    return lambda x_: lin_map @ x_ + l2_reg * (x_ @ x_)


def optimize(
    synapses,
    spikes,
    wmin=float("-inf"),
    wmax=float("inf"),
    l2_reg=1e-1,
    dzmin=1e-6,
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
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 1e-1.
        dzmin (float, optional): Minimum derivative constraint for sufficient rise at firing times. Defaults to 1e-6.

    Returns:
        pl.DataFrame: Optimized synapses as a DataFrame with columns 'source', 'target', 'delay', 'weight'. If optimization fails for any neuron, returns original synapses with 'weight' set to None.

    Raises:
        ValueError: If eps < 0, zmax > FIRING_THRESHOLD, dzmin < 0, or wmin >= wmax.
    """
    if wmin >= wmax:
        raise ValueError("wmin must be less than wmax.")

    synapses_lst = []

    for (neuron,), in_synapses in synapses.partition_by("target", as_dict=True).items():
        # Prepare synapses and output spikes for the current neuron
        in_synapses = init_synapses(in_synapses, spikes.select("neuron"))
        out_spikes = init_out_spikes(spikes.filter(pl.col("neuron") == neuron))

        ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
        model = gp.Model("model")
        model.setParam("OutputFlag", 0)  # Disable output

        # Setup variables to be optimized = the synaptic weights
        weights = model.addMVar(shape=in_synapses.height, lb=wmin, ub=wmax)
        logger.debug(f"Neuron {neuron}. Learnable weights initialized.")

        syn_states, rec_states = compute_states(in_synapses, out_spikes, spikes)

        # Objective: to minimize an energy-based metric -- todo: optimize!!
        quad_obj = compute_objective(in_synapses.height, out_spikes, syn_states, l2_reg)
        model.setObjective(quad_obj(weights), sense=gp.GRB.MINIMIZE)
        logger.debug(f"Neuron {neuron}. Objective function set.")

        # Firing constraints: exact crossing and sufficient rise
        lin_map = compute_linear_map(
            in_synapses.height,
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time"),
        )
        model.addConstr(lin_map(weights) == FIRING_THRESHOLD)

        lin_map = compute_linear_map(
            in_synapses.height,
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time"),
            deriv=1,
        )
        model.addConstr(lin_map(weights) >= dzmin)  # type: ignore
        logger.debug(f"Neuron {neuron}. Firing constraints added.")

        model.optimize()
        if model.status != gp.GRB.OPTIMAL:
            logger.error(
                f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
            )
            return synapses.with_columns(pl.lit(None, pl.Float64).alias("weight"))

        logger.info(f"Neuron {neuron}. Optimization completed!")
        synapses_lst.append(
            in_synapses.update(
                pl.DataFrame({"weight": weights.X}).with_row_index("in_index"),
                on="in_index",
            ).drop("in_index")
        )

    return pl.concat(synapses_lst)
