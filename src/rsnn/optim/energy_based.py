import gurobipy as gp
import polars as pl

import rsnn_plugin as rp
from rsnn import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.log import setup_logging
from rsnn.optim import GUROBI_STATUS
from rsnn.optim.utils import (
    compute_linear_map,
    dataframe_to_1d_array,
    dataframe_to_sym_2d_array,
    init_out_spikes,
    init_synapses,
    modulo_with_offset,
)

logger = setup_logging(__name__, console_level="INFO", file_level="DEBUG")


def compute_states(synapses, out_spikes, src_spikes):
    """Compute input states for synaptic transmission and refractoriness.

    Calculates state transitions for neuronal dynamics including synaptic inputs
    and refractory periods. The function processes synaptic transmission events
    and refractory reset states separately, then combines them.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns including
            'in_index', 'source', 'target', 'delay', 'weight'.
        out_spikes (pl.DataFrame): Output spike times with columns including 'index',
            'f_index', 'time_prev', 'period'.
        src_spikes (pl.DataFrame): Source spike times with columns including 'index', 'neuron', 'time', 'period'.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
            - syn_states: Synaptic input states with in_index
            - rec_states: Refractory states without in_index
    """
    # Set the time origins per index
    origins = out_spikes.group_by("index").agg(pl.min("time_prev").alias("time_origin"))

    # Refractoriness
    rec_states = out_spikes.select(
        pl.col("index"),
        pl.col("f_index"),
        pl.col("time_prev").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight"),
        pl.lit(1.0, pl.Float64).alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    )

    # Synaptic transmission
    syn_states = (
        synapses.join(src_spikes, left_on="source", right_on="neuron")
        .join(origins, on="index")
        .select(
            pl.col("index"),
            pl.lit(None, pl.UInt32).alias("f_index"),
            modulo_with_offset(
                pl.col("time") + pl.col("delay"),
                pl.col("period"),
                pl.col("time_origin"),
            ).alias("start"),
            pl.col("in_index"),
            pl.col("weight"),
            pl.lit(0.0, pl.Float64).alias("in_coef_0"),
            pl.lit(1.0, pl.Float64).alias("in_coef_1"),
        )
    )

    # All input states
    in_states = syn_states.extend(rec_states).select(
        pl.col("f_index").forward_fill().over("index", order_by="start"),
        pl.col("start"),
        pl.col("in_index"),
        pl.col("weight"),
        pl.col("in_coef_0"),
        pl.col("in_coef_1"),
    )

    return (
        in_states.filter(pl.col("in_index").is_not_null()),
        in_states.filter(pl.col("in_index").is_null()).drop("in_index"),
    )


def compute_energy_metrics(
    n_synapses,
    syn_states,
    rec_states,
    out_spikes,
    syn_loc,
):
    """Compute energy-based metrics for synaptic weights optimization.

    Calculates the weighted mean (linear term) and precision matrix (quadratic term)
    for the energy-based objective function. The metric can be computed either
    synapse-locally (diagonal) or neuron-locally (full matrix).

    Args:
        n_synapses (int): Number of synapses to optimize.
        syn_states (pl.DataFrame): Synaptic states containing 'f_index',
            'in_index', 'start'.
        rec_states (pl.DataFrame): Refractory states containing 'f_index',
            'start', 'weight'.
        out_spikes (pl.DataFrame): Output spikes containing 'f_index', 'time'.
        syn_loc (bool): If True, use synapse-local (diagonal) metric.
            If False, use neuron-local (full) metric.

    Returns:
        tuple[np.ndarray | None, np.ndarray]: A tuple containing:
            - weighted_mean: Linear term coefficients or None if rec_states is None
            - precision: Quadratic term precision matrix
    """
    if syn_loc:
        syn_to_syn_metric = (
            syn_states.join(syn_states, on=["f_index", "in_index"])
            .join(out_spikes, on="f_index")
            .group_by("in_index")
            .agg(
                rp.energy_syn_to_syn_metric(
                    (pl.col("start") - pl.col("start_right")).abs(),
                    pl.col("time")
                    - pl.max_horizontal(pl.col("start"), pl.col("start_right")),
                )
                .sum()
                .alias("coef")
            )
        )
        precision = dataframe_to_sym_2d_array(
            syn_to_syn_metric,
            "in_index",
            "in_index",
            "coef",
            (n_synapses, n_synapses),
        )
    else:
        syn_to_syn_metric = (
            syn_states.join(syn_states, on="f_index")
            .filter(pl.col("in_index") <= pl.col("in_index_right"))
            .join(out_spikes, on="f_index")
            .group_by(["in_index", "in_index_right"])
            .agg(
                rp.energy_syn_to_syn_metric(
                    (pl.col("start") - pl.col("start_right")).abs(),
                    pl.col("time")
                    - pl.max_horizontal(pl.col("start"), pl.col("start_right")),
                )
                .sum()
                .alias("coef")
            )
        )

        precision = dataframe_to_sym_2d_array(
            syn_to_syn_metric,
            "in_index",
            "in_index_right",
            "coef",
            (n_synapses, n_synapses),
        )

    if rec_states is None:
        weighted_mean = None
    else:
        rec_to_syn_metric = (
            syn_states.join(rec_states, on="f_index")
            .join(out_spikes, on="f_index")
            .group_by("in_index")
            .agg(
                (
                    rp.energy_rec_to_syn_metric(
                        pl.col("start") - pl.col("start_right"),
                        pl.col("time") - pl.col("start"),
                    )
                    * pl.col("weight")
                )
                .sum()
                .alias("coef")
            )
        )
        weighted_mean = dataframe_to_1d_array(
            rec_to_syn_metric, "in_index", "coef", n_synapses
        )

    return weighted_mean, precision


def optimize(
    synapses,
    spikes,
    wmin=float("-inf"),
    wmax=float("inf"),
    syn_loc=True,
    lin_rec=False,
    l2_reg=0.0,
    dzmin=1e-6,
):
    """Optimize synaptic weights using energy-based quadratic programming.

    Performs constrained optimization of synaptic weights to minimize an energy-based
    objective function while satisfying firing constraints. The optimization can use
    different energy metrics based on the specified parameters.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns including
            'source', 'target', 'delay', 'weight'.
        spikes (pl.DataFrame): Spike train data with columns 'index', 'neuron', 'time'.
        wmin (float, optional): Minimum synaptic weight bound. Defaults to -inf.
        wmax (float, optional): Maximum synaptic weight bound. Defaults to inf.
        syn_loc (bool, optional): If True, use synapse-local (diagonal) energy metric.
            If False, use neuron-local (full) energy metric. Defaults to True.
        lin_rec (bool, optional): If True, include linear recovery term in the
            energy metric. Defaults to False.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 0.0.
        dzmin (float, optional): Minimum derivative constraint for sufficient rise
            at firing times. Defaults to 1e-6.

    Returns:
        pl.DataFrame: Optimized synapses as a DataFrame.

    Raises:
        ValueError: If wmin >= wmax.
        RuntimeError: If optimization fails.
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
        weighted_mean, precision = compute_energy_metrics(
            in_synapses.height,
            syn_states,
            rec_states if lin_rec else None,
            out_spikes,
            syn_loc,
        )
        logger.debug(f"Neuron {neuron}. Energy metrics computed.")
        if weighted_mean is None:
            model.setObjective(
                weights @ precision @ weights + l2_reg * (weights @ weights),
                sense=gp.GRB.MINIMIZE,
            )
        else:
            model.setObjective(
                weights @ precision @ weights
                + weighted_mean @ weights
                + l2_reg * (weights @ weights),
                sense=gp.GRB.MINIMIZE,
            )
        logger.debug(f"Neuron {neuron}. Objective function set.")

        # Firing constraints: exact crossing and sufficient rise
        syn_lin_map, rec_lin_offset = compute_linear_map(
            in_synapses.height,
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time"),
        )
        model.addConstr(syn_lin_map @ weights + rec_lin_offset == FIRING_THRESHOLD)

        syn_lin_map, rec_lin_offset = compute_linear_map(
            in_synapses.height,
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time"),
            deriv=1,
        )
        model.addConstr(syn_lin_map @ weights + rec_lin_offset >= dzmin)
        logger.debug(f"Neuron {neuron}. Firing constraints added.")

        model.optimize()
        if model.status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
            )

        logger.info(f"Neuron {neuron}. Optimization completed!")
        synapses_lst.append(
            in_synapses.update(
                pl.DataFrame({"weight": weights.X}).with_row_index("in_index"),
                on="in_index",
            ).drop("in_index")
        )

    return pl.concat(synapses_lst)
