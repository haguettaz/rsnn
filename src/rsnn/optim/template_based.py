import gurobipy as gp
import numpy as np
import polars as pl

import rsnn_plugin as rp
from rsnn import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.log import setup_logging
from rsnn.optim import GUROBI_STATUS
from rsnn.optim.utils import (
    compute_linear_map,
    init_out_spikes,
    init_synapses,
    modulo_with_offset,
)

logger = setup_logging(__name__, console_level="INFO", file_level="DEBUG")


def compute_states(
    synapses: pl.DataFrame,
    out_spikes: pl.DataFrame,
    src_spikes: pl.DataFrame,
    eps: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compute neuronal states for template-based optimization.

    Calculates different types of states needed for the template-based optimization: firing states, synaptic transmission states, and refractory states. Also computes before-firing states for constraint violations.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns including 'source', 'target', 'delay'.
        out_spikes (pl.DataFrame): Output spike times with columns 'index', 'period', 'neuron', 'f_index', 'time', 'time_prev'. Must be sorted by time within each (index, neuron) group.
        src_spikes (pl.DataFrame): Source spike times with columns 'index', 'period', 'neuron', 'time'.
        eps (float): Epsilon parameter for before-firing state timing.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
            - syn_states: Synaptic transmission states with columns 'f_index', 'in_index', 'start'
            - rec_states: Refractory states with columns 'f_index', 'start'
            - states: All states with columns 'f_index', 'start', 'in_index', 'weight', 'in_coef_0', 'in_coef_1', 'active', 'length', and sorted by start time over each firing index group.

    Raises:
        ValueError: If eps is not positive.
    """
    if eps <= 0:
        raise ValueError("epsilon must be positive.")

    # Set the time origins per index
    origins = out_spikes.group_by(["index", "neuron"]).agg(
        pl.first("time_prev").alias("time_origin")
    )

    # Firing states
    f_states = out_spikes.select(
        pl.col("f_index"),
        pl.col("time").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(None, pl.Float64).alias("weight"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
        pl.lit(True, pl.Boolean).alias("active"),
    )

    before_f_states = out_spikes.select(
        pl.col("f_index"),
        (pl.col("time") - eps).clip(pl.col("time_prev")).alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(None, pl.Float64).alias("weight"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
        pl.lit(True, pl.Boolean).alias("active"),
    )

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
        origins.join(synapses, left_on="neuron", right_on="target")
        .join(src_spikes, left_on=["index", "source"], right_on=["index", "neuron"])
        .select(
            pl.col("index"),
            pl.lit(None, pl.UInt32).alias("f_index"),
            modulo_with_offset(
                pl.col("time") + pl.col("delay"),
                pl.col("period"),
                pl.col("time_origin"),
            ).alias("start"),
            pl.col("in_index"),
            pl.lit(None, pl.Float64).alias("weight"),
            pl.lit(0.0, pl.Float64).alias("in_coef_0"),
            pl.lit(1.0, pl.Float64).alias("in_coef_1"),
        )
    )
    in_states = (
        syn_states.sort("start")
        .merge_sorted(rec_states, key="start")
        .select(
            pl.col("f_index").forward_fill().over("index"),
            pl.col("start"),
            pl.col("in_index"),
            pl.col("weight"),
            pl.col("in_coef_0"),
            pl.col("in_coef_1"),
        )
    )

    states = (
        in_states.with_columns(pl.lit(None, pl.Boolean).alias("active"))
        .merge_sorted(before_f_states, key="start")
        .merge_sorted(f_states, key="start")
        .with_columns(
            pl.col("active").forward_fill().fill_null(False).over("f_index"),
            pl.col("start")
            .diff()
            .shift(-1, fill_value=0.0)
            .over("f_index")
            .alias("length"),
        )
    )

    return (
        in_states.filter(pl.col("in_index").is_not_null()).select(
            "f_index", "start", "in_index"
        ),
        in_states.filter(pl.col("in_index").is_null()).select("f_index", "start"),
        states,
    )


def optimize(
    synapses: pl.DataFrame,
    spikes: pl.DataFrame,
    wmin: float = float("-inf"),
    wmax: float = float("inf"),
    eps: float = 0.2,
    zmax: float = 0.0,
    dzmin: float = 2.0,
    n_iter: int = 1000,
    feas_tol: float = 1e-5,
) -> pl.DataFrame:
    """Optimize synaptic weights using template-based iterative constraint refinement.

    Performs iterative optimization of synaptic weights by alternating between weight optimization and constraint refinement.
    Uses linear programming with template-based constraints to ensure proper firing behavior.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns 'source', 'target', 'delay', 'weight'.
        spikes (pl.DataFrame): Spike train data with columns 'index', 'period', 'neuron', 'time'.
        wmin (float, optional): Minimum synaptic weight bound. Defaults to -inf.
        wmax (float, optional): Maximum synaptic weight bound. Defaults to inf.
        eps (float, optional): Epsilon parameter for before-firing constraints. Defaults to 0.2.
        zmax (float, optional): Maximum membrane potential in silent regions. Defaults to 0.0.
        dzmin (float, optional): Minimum derivative in active regions for sufficient rise. Defaults to 1.0.
        n_iter (int, optional): Maximum number of constraint generation iterations. Defaults to 1000.
        feas_tol (float, optional): Feasibility tolerance for constraint violations. Defaults to 1e-5.

    Returns:
        pl.DataFrame: Optimized synapses as a DataFrame with columns 'source', 'target', 'delay', 'weight'.
            If optimization fails for any neuron, returns synapses with 'weight' set to None.

    Raises:
        ValueError: If eps < 0, zmax > FIRING_THRESHOLD, dzmin < 0, or wmin >= wmax.

    Notes:
        The optimization alternates between:
        1. Minimizing L2 norm of weights subject to current constraints
        2. Adding new constraints for detected violations in silent/active regions
        Converges when no more constraint violations are found.
    """
    if eps <= 0:
        raise ValueError("epsilon must be positive.")

    if zmax > FIRING_THRESHOLD:
        raise ValueError("zmax must be less than or equal to the firing threshold.")

    if dzmin < 0:
        raise ValueError("dzmin must be non-negative.")

    if wmin >= wmax:
        raise ValueError("wmin must be less than wmax.")

    synapses_lst = []

    for (neuron,), in_synapses in synapses.partition_by(
        "target", maintain_order=False, as_dict=True
    ).items():
        logger.debug(f"Optimizing neuron {neuron}...")
        # Prepare synapses and output spikes for the current neuron
        in_synapses = init_synapses(in_synapses, spikes.select("neuron"))
        out_spikes = init_out_spikes(spikes.filter(pl.col("neuron") == neuron))

        ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
        model = gp.Model("model")
        model.setParam("OutputFlag", 0)  # Disable output

        # Setup variables to be optimized = the synaptic weights
        weights = model.addMVar(shape=in_synapses.height, lb=wmin, ub=wmax)
        logger.debug(f"Neuron {neuron}. Learnable weights initialized.")

        # Objective function - activity-agnostic
        model.setObjective(
            weights @ weights,
            sense=gp.GRB.MINIMIZE,
        )
        logger.debug(f"Neuron {neuron}. Objective function set.")

        # Compute states for linear constraints
        syn_states, rec_states, states = compute_states(
            in_synapses, out_spikes, spikes, eps
        )

        # Firing time constraints
        syn_lin_map, rec_lin_offset = compute_linear_map(
            in_synapses.height,
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time"),
        )
        model.addConstr(syn_lin_map @ weights + rec_lin_offset == FIRING_THRESHOLD)
        logger.debug(f"Neuron {neuron}. Firing time constraints added.")

        for i in range(n_iter):
            # 1. Optimize weights
            model.optimize()

            if model.status != gp.GRB.OPTIMAL:
                logger.error(
                    f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
                )
                return synapses.with_columns(pl.lit(None, pl.Float64).alias("weight"))

            states = scan_with_new_weights(states, weights.X)  # type: ignore

            # 2. Refine constraints
            silent_violations = (
                states.filter(pl.col("active") == False)
                .group_by("f_index")
                .agg(
                    rp.max_violation(
                        pl.col("start"),
                        pl.col("length"),
                        pl.col("coef_0"),
                        pl.col("coef_1"),
                        vmax=feas_tol + zmax,
                    ).alias("time")
                )
                .drop_nulls()
            )
            if silent_violations.height > 0:
                syn_lin_map, rec_lin_offset = compute_linear_map(
                    in_synapses.height,
                    syn_states,
                    rec_states,
                    silent_violations,
                )
                model.addConstr(syn_lin_map @ weights + rec_lin_offset <= zmax)  # type: ignore

            active_violations = (
                states.filter(pl.col("active") == True)
                .group_by("f_index")
                .agg(
                    rp.max_violation(
                        pl.col("start"),
                        pl.col("length"),
                        (pl.col("coef_0") - pl.col("coef_1")),
                        pl.col("coef_1"),
                        vmax=feas_tol - dzmin,
                    ).alias("time")
                )
                .drop_nulls()
            )
            if active_violations.height > 0:
                syn_lin_map, rec_lin_offset = compute_linear_map(
                    in_synapses.height,
                    syn_states,
                    rec_states,
                    active_violations,
                    deriv=1,
                )
                model.addConstr(syn_lin_map @ weights + rec_lin_offset >= dzmin)  # type: ignore

            n_violations = active_violations.height + silent_violations.height

            logger.debug(
                f"Neuron {neuron}: iteration {i}. The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints. {n_violations} new linear constraints to add."
            )

            if n_violations == 0:
                logger.info(
                    f"Neuron {neuron}. Optimization successful (in {i} iterations)"
                )
                synapses_lst.append(
                    in_synapses.update(
                        pl.DataFrame({"weight": weights.X}).with_row_index("in_index"),
                        on="in_index",
                    ).drop("in_index")
                )
                break

    return pl.concat(synapses_lst)


def scan_with_new_weights(states: pl.DataFrame, weights: np.ndarray) -> pl.DataFrame:
    """Update state coefficients with new synaptic weights using cumulative scanning.

    Recalculates the membrane potential coefficients (coef_0, coef_1) for all states after updating synaptic weights. Uses temporal scanning to accumulate the effects of synaptic inputs over time with exponential decay.

    Args:
        states (pl.DataFrame): Neuronal states with columns 'f_index', 'start', 'length', 'in_index', 'in_coef_0', 'in_coef_1', 'weight'. Must be sorted by start time within each f_index group
        weights (np.ndarray): New synaptic weight values to apply.

    Returns:
        pl.DataFrame: Updated states with recalculated 'weight', 'coef_0', and 'coef_1' columns.

    Notes:
        - Uses exponential decay scanning for temporal dynamics
        - Updates both constant (coef_0) and linear (coef_1) membrane potential terms
    """
    return (
        states.update(
            pl.DataFrame({"weight": weights}).with_row_index("in_index"),
            on="in_index",
        )
        .with_columns(
            rp.scan_coef_1(
                pl.col("length").shift(), pl.col("in_coef_1") * pl.col("weight")
            )
            .over("f_index")
            .alias("coef_1")
        )
        .with_columns(
            rp.scan_coef_0(
                pl.col("length").shift(),
                pl.col("coef_1").shift(),
                pl.col("in_coef_0") * pl.col("weight"),
            )
            .over("f_index")
            .alias("coef_0")
        )
    )
