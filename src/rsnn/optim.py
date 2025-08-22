from collections import defaultdict

import gurobipy as gp
import polars as pl
from scipy import sparse
from tqdm import trange

from .channels import new_channels, transfer_through_channels
from .constants import FIRING_THRESHOLD, REFRACTORY_PERIOD, REFRACTORY_RESET
from .log import setup_logging
from .spikes import *
from .states import compute_maxima, update_coef, update_length

logger = setup_logging(__name__, console_level="DEBUG", file_level="INFO")


GUROBI_STATUS = defaultdict(lambda: "unknown status")
GUROBI_STATUS[gp.GRB.OPTIMAL] = "optimal"
GUROBI_STATUS[gp.GRB.INFEASIBLE] = "infeasible"
GUROBI_STATUS[gp.GRB.UNBOUNDED] = "unbounded"
GUROBI_STATUS[gp.GRB.INF_OR_UNBD] = "infeasible or unbounded"
GUROBI_STATUS[gp.GRB.NUMERIC] = "numeric error"

STATES_COLS = [
    "neuron",
    "f_index",
    "start",
    "length",
    "reset",
    "in_index",
    "w0",
    "w1",
    "active",
]


def create_recovery_states(spikes):
    return spikes.with_columns(
        pl.lit(None, pl.Boolean).alias("active"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.col("time").alias("start"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("w0"),
        pl.lit(0.0, pl.Float64).alias("w1"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("c0"),
        pl.lit(0.0, pl.Float64).alias("c1"),
    ).select(
        "index",
        "neuron",
        "f_index",
        "active",
        "start",
        "in_index",
        "w0",
        "w1",
        "c0",
        "c1",
    )


def create_synaptic_states(spikes, synapses, origins):
    return (
        synapses.join(spikes, left_on="source", right_on="neuron")
        .join(origins, left_on=["index", "target"], right_on=["index", "neuron"])
        .with_columns(
            pl.col("target").alias("neuron"),
            pl.lit(None, pl.UInt32).alias("f_index"),
            pl.lit(None, pl.Boolean).alias("active"),
            modulo_with_offset(
                pl.col("time") + pl.col("delay"),
                pl.col("period"),
                pl.col("time_origin"),
            ).alias("start"),
            pl.lit(None, pl.Float64).alias("c0"),
            pl.lit(None, pl.Float64).alias("c1"),
        )
        .select(
            "index",
            "neuron",
            "f_index",
            "active",
            "start",
            "in_index",
            "w0",
            "w1",
            "c0",
            "c1",
        )
    )


def create_offline_states(spikes, synapses, eps=0.2):
    """Returns states with transformed spikes and synapses."""
    if eps < 0:
        raise ValueError("epsilon must be positive.")

    spikes = extend_with_time_prev(spikes, over=["index", "neuron"])  # result is sorted
    spikes = spikes.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index")
    )

    ## Refractoriness
    rec_states = create_recovery_states(
        spikes.drop("time").rename({"time_prev": "time"})
    )

    ## Synaptic transmission
    # 0. Compute origins for each neuron
    origins = spikes.group_by("index", "neuron").agg(
        pl.min("time_prev").alias("time_origin")
    )

    # 1. Extract synapses to spiking neurons (the other synapses can be ignored, i.e., have weights = 0.0)
    synapses = synapses.join(spikes, left_on="target", right_on="neuron", how="semi")
    synapses = synapses.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("in_index")
    )

    # 2. Create synaptic states
    syn_states = create_synaptic_states(spikes, synapses, origins)

    ## Virtual states for specific time marks
    v_states_1 = spikes.with_columns(
        pl.lit(True, pl.Boolean).alias("active"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.col("time").alias("start"),
        pl.lit(0.0, pl.Float64).alias("w0"),
        pl.lit(0.0, pl.Float64).alias("w1"),
        pl.lit(None, pl.Float64).alias("c0"),
        pl.lit(None, pl.Float64).alias("c1"),
    )
    v_states_2 = spikes.with_columns(
        pl.lit(True, pl.Boolean).alias("active"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        (pl.col("time") - eps).clip(pl.col("time_prev")).alias("start"),
        pl.lit(0.0, pl.Float64).alias("w0"),
        pl.lit(0.0, pl.Float64).alias("w1"),
        pl.lit(None, pl.Float64).alias("c0"),
        pl.lit(None, pl.Float64).alias("c1"),
    )
    v_states = pl.concat([v_states_1, v_states_2]).select(
        "index",
        "neuron",
        "f_index",
        "active",
        "start",
        "in_index",
        "w0",
        "w1",
        "c0",
        "c1",
    )

    # Merge and sort (grouped by index and neuron)
    states = pl.concat([rec_states, syn_states, v_states])
    states = states.sort(["start", "f_index"])
    states = states.with_columns(
        pl.col("f_index").forward_fill().over(["index", "neuron"])
    )
    states = states.with_columns(
        pl.col("active").forward_fill().fill_null(False).over("f_index")
    )
    states = update_length(states, over="f_index", fill_value=0.0)

    return spikes, synapses, states


def optimize_offline(
    spikes,
    synapses,
    states,
    wmin=float("-inf"),
    wmax=float("inf"),
    zmax=0.0,
    dzmin=1.0,
    n_iter=1000,
    feas_tol=1e-5,
):
    # spikes is a dataframe with columns: index, period, neuron, time, time_prev
    # synapses is a dataframe with columns: in_index, source, target, delay, w0, w1
    # states is a dataframe with columns: index, neuron, active, start, length, in_index, w0, w1, c0, c1

    if zmax > FIRING_THRESHOLD:
        raise ValueError("zmax must be less than or equal to the firing threshold.")

    if dzmin < 0:
        raise ValueError("dzmin must be non-negative.")

    if wmin >= wmax:
        raise ValueError("wmin must be less than wmax.")

    ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
    model = gp.Model("qp_model")
    model.setParam("OutputFlag", 0)  # Disable output
    weights = model.addMVar(shape=synapses.height, lb=wmin, ub=wmax)
    model.setObjective(weights @ weights, sense=gp.GRB.MINIMIZE)  # to be adapted
    A, b = compute_linear_cstr(spikes, states, synapses.height)
    model.addConstr(A @ weights + b == FIRING_THRESHOLD)

    # for i in trange(n_iter):
    for i in range(n_iter):
        # 1. Optimize weights
        model.optimize()

        if model.status != gp.GRB.OPTIMAL:
            logger.warning(f"Optimization failed: {GUROBI_STATUS[model.status]}")
            return synapses.drop("in_index")

        synapses = update_synapses_from_weights(synapses, weights.X)
        states = update_states_from_weights(states, weights.X)

        # 2. Refine constraints
        states_silent = states.filter(~pl.col("active"))
        max_violations_silent = compute_maxima(
            states_silent, feas_tol + zmax, k=1, by="f_index"
        )

        if max_violations_silent.height > 0:
            A, b = compute_linear_cstr(
                max_violations_silent.select("f_index", "time"),
                states,
                synapses.height,
            )
            model.addConstr(A @ weights + b <= zmax)  # silent area: z <= zmax

        states_active = states.filter(pl.col("active")).with_columns(
            (pl.col("c0") - pl.col("c1")).alias("c0")
        )
        max_violations_active = compute_maxima(
            states_active, feas_tol - dzmin, k=1, by="f_index"
        )

        if max_violations_active.height > 0:
            A, b = compute_linear_cstr(
                max_violations_active.select("f_index", "time"),
                states,
                synapses.height,
                deriv=1,
            )
            model.addConstr(A @ weights + b >= dzmin)  # active area: dz >= dzmin

        # Prune neuron states with no more violations (their synaptic weights are optimal)
        max_violations = max_violations_active.vstack(max_violations_silent)
        states = states.join(max_violations, on="neuron", how="semi")

        logger.debug(
            f"Iteration {i}. Optimization: the objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints. Constraint refinement: {max_violations.height} linear constraints added over {max_violations.n_unique('neuron')} different neurons."
        )

        if max_violations.height == 0:
            logger.info(f"Optimization successful (in {i} iterations)")
            return synapses.drop("in_index")

    logger.warning(f"Optimization stopped after {n_iter} iterations")
    return synapses.drop("in_index")


def compute_linear_cstr(times, states, n_synapses, deriv=0):
    """Times and states should be indexed by f_index. Returns A and b in A w <= b. Assume recovery and synaptic transmission are zero and first-order fading kernels, respectively."""
    times = times.with_row_index("time_index")
    rec_lin_offset = (
        times.join(
            states.filter(pl.col("in_index").is_null()).select(
                "f_index", "start", "w0", "w1"
            ),
            on="f_index",
            how="left",
        )
        .filter((pl.col("start") <= pl.col("time")))
        .with_columns(
            (pl.col("time") - pl.col("start")).alias("delta"),
        )
        .group_by("time_index")
    )

    syn_lin_map = (
        times.join(
            states.filter(pl.col("in_index").is_not_null()).select(
                "f_index", "in_index", "start"
            ),
            on="f_index",
            how="left",
        )
        .filter((pl.col("start") <= pl.col("time")))
        .with_columns(
            (pl.col("time") - pl.col("start")).alias("delta"),
        )
        .group_by("time_index", "in_index")
    )

    if deriv % 2 == 0:
        rec_lin_offset = rec_lin_offset.agg(
            (
                (pl.col("w0") - deriv * pl.col("w1") + pl.col("w1") * pl.col("delta"))
                * (-pl.col("delta")).exp()
            )
            .sum()
            .alias("coef")
        )
        syn_lin_map = syn_lin_map.agg(
            ((pl.col("delta") - deriv) * (-pl.col("delta")).exp()).sum().alias("coef")
        )
    else:
        rec_lin_offset = rec_lin_offset.agg(
            (
                (deriv * pl.col("w1") - pl.col("w0") - pl.col("w1") * pl.col("delta"))
                * (-pl.col("delta")).exp()
            )
            .sum()
            .alias("coef")
        )
        syn_lin_map = syn_lin_map.agg(
            ((deriv - pl.col("delta")) * (-pl.col("delta")).exp()).sum().alias("coef")
        )

    b = rec_lin_offset.sort("time_index").get_column("coef").to_numpy()
    A = sparse.csr_array(
        (
            (syn_lin_map.get_column("coef").to_numpy()),
            (
                syn_lin_map.get_column("time_index").to_numpy(),
                syn_lin_map.get_column("in_index").to_numpy(),
            ),
        ),
        shape=(b.size, n_synapses),
    )

    return A, b


def update_synapses_from_weights(synapses, weights):
    # update synapses with given weights
    return synapses.update(
        pl.DataFrame({"w1": weights}).with_row_index("in_index"),
        on="in_index",
    )


def update_states_from_weights(states, weights):
    # update synaptic states from given weights
    states = states.update(
        pl.DataFrame({"w1": weights}).with_row_index("in_index"),
        on="in_index",
    )

    states = update_coef(states, over="f_index")
    return states
