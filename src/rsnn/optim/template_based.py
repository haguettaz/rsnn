from collections import defaultdict

import gurobipy as gp
import polars as pl
from scipy import sparse

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


def compute_states(synapses, out_spikes, src_spikes, eps):
    """Returns:
    - f_states: firing states at spike times with columns f_index, start
    - syn_states: synaptic transmission states at synaptic event times
    - rec_states: refractoriness states at refractory event times
    - states: all states sorted by (f_index, start)
    """
    if eps < 0:
        raise ValueError("epsilon must be positive.")

    # Set the time origins per index
    origins = out_spikes.group_by("index").agg(pl.min("time_prev").alias("time_origin"))

    # Firing states
    f_states = out_spikes.select(
        # pl.col("index"),
        pl.col("f_index"),
        pl.col("time").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(None, pl.Float64).alias("weight"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
        pl.lit(True, pl.Boolean).alias("active"),
    )

    before_f_states = out_spikes.select(
        # pl.col("index"),
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
    in_states = syn_states.extend(rec_states).select(
        pl.col("f_index").forward_fill().over("index", order_by="start"),
        pl.col("start"),
        pl.col("in_index"),
        pl.col("weight"),
        pl.col("in_coef_0"),
        pl.col("in_coef_1"),
    )

    states = (
        in_states.with_columns(pl.lit(None, pl.Boolean).alias("active"))
        .extend(before_f_states)
        .extend(f_states)
        .with_columns(
            pl.col("active")
            .forward_fill()
            .fill_null(False)
            .over("f_index", order_by="start")
        )
        .with_columns(
            pl.col("start")
            .diff()
            .shift(-1, fill_value=0.0)
            .over("f_index", order_by="start")
            .alias("length")
        )
    )

    return (
        in_states.filter(pl.col("in_index").is_not_null()),
        in_states.filter(pl.col("in_index").is_null()).drop("in_index"),
        states,
    )


# def event_offline_optimization(spikes, synapses, wmin, wmax):
#     # Implement event-based offline optimization here
#     if wmin >= wmax:
#         raise ValueError("wmin must be less than wmax.")

#     synapses_lst = []

#     for (neuron,), in_synapses in synapses.partition_by("target").items():
#         out_spikes, in_synapses, states = init_event_offline_optimization(
#             spikes, in_synapses
#         )

#         ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
#         model = gp.Model("model")
#         model.setParam("OutputFlag", 0)  # Disable output

#         # Setup variables to be optimized = the synaptic weights
#         weights = model.addMVar(shape=in_synapses.height, lb=wmin, ub=wmax)

#         # Objective function
#         syn_lin_map, _ = compute_linear_map(
#             out_spikes, states, in_synapses.height, deriv=1
#         )
#         syn_lin_map = syn_lin_map.sum(axis=0)
#         model.setObjective(
#             syn_lin_map @ weights, sense=gp.GRB.MAXIMIZE
#         )  # to be adapted

#         # Firing constraints
#         syn_lin_map, rec_lin_offset = compute_linear_map(
#             out_spikes, states, in_synapses.height
#         )
#         model.addConstr(syn_lin_map @ weights + rec_lin_offset == FIRING_THRESHOLD)

#         model.optimize()
#         if model.status != gp.GRB.OPTIMAL:
#             logger.warning(
#                 f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
#             )
#             return

#         logger.info(f"Neuron {neuron}. Optimization successful")
#         in_synapses = update_synapses_from_weights(in_synapses, weights.X)
#         synapses_lst.append(in_synapses.drop("in_index"))

#     return pl.concat(synapses_lst)


def optimize(
    synapses,
    spikes,
    wmin=float("-inf"),
    wmax=float("inf"),
    eps=0.2,
    zmax=0.0,
    dzmin=1.0,
    n_iter=1000,
    feas_tol=1e-5,
):
    # spikes is a dataframe with columns: index, period, neuron, time, time_prev
    # synapses is a dataframe with columns: in_index, source, target, delay, weight_0, weight_1
    # states is a dataframe with columns: index, neuron, active, start, length, in_index, weight_0, weight_1, coef_0, coef_1

    if eps < 0:
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
        in_synapses = init_synapses(in_synapses, spikes)
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
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time"),
            in_synapses.height,
        )
        model.addConstr(syn_lin_map @ weights + rec_lin_offset == FIRING_THRESHOLD)
        logger.debug(f"Neuron {neuron}. Firing time constraints added.")

        for i in range(n_iter):
            # 1. Optimize weights
            model.optimize()

            if model.status != gp.GRB.OPTIMAL:
                logger.warning(
                    f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
                )
                return

            states = scan_with_new_weights(states, weights.X)

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
                    syn_states,
                    rec_states,
                    silent_violations,
                    in_synapses.height,
                )
                model.addConstr(
                    syn_lin_map @ weights + rec_lin_offset <= zmax
                )  # silent area: z <= zmax

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
                    syn_states,
                    rec_states,
                    active_violations,
                    in_synapses.height,
                    deriv=1,
                )
                model.addConstr(
                    syn_lin_map @ weights + rec_lin_offset >= dzmin
                )  # active area: dz >= dzmin

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

        # logger.warning(f"Optimization stopped after {n_iter} iterations")
        # return
    return pl.concat(synapses_lst)

    # def continuous_offline_optimization(
    #     spikes,
    #     synapses,
    #     states,
    #     wmin=float("-inf"),
    #     wmax=float("inf"),
    #     zmax=0.0,
    #     dzmin=1.0,
    #     n_iter=1000,
    #     feas_tol=1e-5,
    # ):
    #     # spikes is a dataframe with columns: index, period, neuron, time, time_prev
    #     # synapses is a dataframe with columns: in_index, source, target, delay, weight_0, weight_1
    #     # states is a dataframe with columns: index, neuron, active, start, length, in_index, weight_0, weight_1, coef_0, coef_1

    #     if zmax > FIRING_THRESHOLD:
    #         raise ValueError("zmax must be less than or equal to the firing threshold.")

    #     if dzmin < 0:
    #         raise ValueError("dzmin must be non-negative.")

    #     if wmin >= wmax:
    #         raise ValueError("wmin must be less than wmax.")

    #     ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
    #     model = gp.Model("model")
    #     model.setParam("OutputFlag", 0)  # Disable output

    #     # Setup variables to be optimized = the synaptic weights
    #     weights = model.addMVar(shape=synapses.height, lb=wmin, ub=wmax)

    #     # Objective function
    #     model.setObjective(weights @ weights, sense=gp.GRB.MINIMIZE)  # to be adapted

    #     # Firing constraints
    #     syn_lin_map, rec_lin_offset = compute_linear_map(spikes, states, synapses.height)
    #     model.addConstr(syn_lin_map @ weights + rec_lin_offset == FIRING_THRESHOLD)

    #     # for i in trange(n_iter):
    #     for i in range(n_iter):
    #         # 1. Optimize weights
    #         model.optimize()

    #         if model.status != gp.GRB.OPTIMAL:
    #             logger.warning(f"Optimization failed: {GUROBI_STATUS[model.status]}")
    #             return synapses.drop("in_index")

    #         synapses = update_synapses_from_weights(synapses, weights.X)
    #         states = update_states_from_weights(states, weights.X)

    #         # 2. Refine constraints
    #         states_silent = states.filter(~pl.col("active"))
    #         max_violations_silent = compute_maxima(
    #             states_silent, feas_tol + zmax, k=1, by="f_index"
    #         )

    #         if max_violations_silent.height > 0:
    #             syn_lin_map, rec_lin_offset = compute_linear_map(
    #                 max_violations_silent.select("f_index", "time"),
    #                 states,
    #                 synapses.height,
    #             )
    #             model.addConstr(
    #                 syn_lin_map @ weights + rec_lin_offset <= zmax
    #             )  # silent area: z <= zmax

    #         states_active = states.filter(pl.col("active")).with_columns(
    #             (pl.col("coef_0") - pl.col("coef_1")).alias("coef_0")
    #         )
    #         max_violations_active = compute_maxima(
    #             states_active, feas_tol - dzmin, k=1, by="f_index"
    #         )

    #         if max_violations_active.height > 0:
    #             syn_lin_map, rec_lin_offset = compute_linear_map(
    #                 max_violations_active.select("f_index", "time"),
    #                 states,
    #                 synapses.height,
    #                 deriv=1,
    #             )
    #             model.addConstr(
    #                 syn_lin_map @ weights + rec_lin_offset >= dzmin
    #             )  # active area: dz >= dzmin

    #         # Prune neuron states with no more violations (their synaptic weights are optimal)
    #         max_violations = max_violations_active.vstack(max_violations_silent)
    #         states = states.join(max_violations, on="neuron", how="semi")

    #         logger.debug(
    #             f"Iteration {i}. Optimization: the objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints. Constraint refinement: {max_violations.height} linear constraints added over {max_violations.n_unique('neuron')} different neurons."
    #         )

    #         if max_violations.height == 0:
    #             logger.info(f"Optimization successful (in {i} iterations)")
    #             return synapses.drop("in_index")

    #     logger.warning(f"Optimization stopped after {n_iter} iterations")
    #     return synapses.drop("in_index")

    # def compute_linear_map(times, states, n_synapses, deriv=0):
    #     """Times and states should be indexed by f_index. Returns A and b in A w <= b. Assume recovery and synaptic transmission are zero and first-order fading kernels, respectively."""
    #     times = times.with_row_index("time_index")
    #     rec_lin_offset = (
    #         (
    #             times.join(
    #                 states.filter(pl.col("in_index").is_null()),
    #                 on="f_index",
    #                 how="left",
    #             )
    #             .filter((pl.col("start") <= pl.col("time")))
    #             .select(
    #                 pl.col("time_index"),
    #                 (pl.col("time") - pl.col("start")).alias("delta"),
    #                 pl.col("weight_0"),
    #                 pl.col("weight_1"),
    #             )
    #         )
    #         .group_by("time_index")
    #         .agg(
    #             (
    #                 (-1) ** (deriv % 2)
    #                 * (
    #                     pl.col("weight_0")
    #                     - deriv * pl.col("weight_1")
    #                     + pl.col("weight_1") * pl.col("delta")
    #                 )
    #                 * (-pl.col("delta")).exp()
    #             )
    #             .sum()
    #             .alias("coef")
    #         )
    #     )

    #     syn_lin_map = (
    #         (
    #             times.join(
    #                 states.filter(pl.col("in_index").is_not_null()),
    #                 on="f_index",
    #                 how="left",
    #             )
    #             .filter((pl.col("start") <= pl.col("time")))
    #             .select(
    #                 pl.col("time_index"),
    #                 pl.col("in_index"),
    #                 (pl.col("time") - pl.col("start")).alias("delta"),
    #             )
    #         )
    #         .group_by("time_index", "in_index")
    #         .agg(
    #             (
    #                 (-1) ** (deriv % 2)
    #                 * (
    #                     pl.col("weight_0")
    #                     - deriv * pl.col("weight_1")
    #                     + pl.col("weight_1") * pl.col("delta")
    #                 )
    #                 * (-pl.col("delta")).exp()
    #             )
    #             .sum()
    #             .alias("coef")
    #         )
    #     )

    #     return (
    #         sparse.csr_array(
    #             (
    #                 (syn_lin_map.get_column("coef").to_numpy()),
    #                 (
    #                     syn_lin_map.get_column("time_index").to_numpy(),
    #                     syn_lin_map.get_column("in_index").to_numpy(),
    #                 ),
    #             ),
    #             shape=(times.height, n_synapses),
    #         ),
    #         rec_lin_offset.sort("time_index").get_column("coef").to_numpy(),
    #     )

    # states = states.with_columns(
    #     rp.scan_coef_1(pl.col("length").shift(), pl.col("weight_1"))
    #     .over("f_index")
    #     .alias("coef_1")
    # )
    # states = states.with_columns(
    #     rp.scan_coef_0(
    #         pl.col("length").shift(),
    #         pl.col("coef_1").shift(),
    #         pl.col("weight_0"),
    #     )
    #     .over("f_index")
    #     .alias("coef_0")
    # )

    # states = update_coef(states, over="f_index")
    # return states


def scan_with_new_weights(states, weights):
    """WARNING: states must be sorted by start in the grouping provided by over."""
    return (
        states.update(
            pl.DataFrame({"weight": weights}).with_row_index("in_index"),
            on="in_index",
        )
        .with_columns(
            rp.scan_coef_1(
                pl.col("length").shift(), pl.col("in_coef_1") * pl.col("weight")
            )
            .over("f_index", order_by="start")
            .alias("coef_1")
        )
        .with_columns(
            rp.scan_coef_0(
                pl.col("length").shift(),
                pl.col("coef_1").shift(),
                pl.col("in_coef_0") * pl.col("weight"),
            )
            .over("f_index", order_by="start")
            .alias("coef_0")
        )
    )
