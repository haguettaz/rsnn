from collections import defaultdict

import gurobipy as gp
import numpy as np
import polars as pl
import scipy.sparse as ss

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


# def create_recovery_states(spikes):
#     return spikes.select(
#         pl.col("index"),
#         pl.col("neuron"),
#         pl.col("f_index"),
#         pl.lit(None, pl.Boolean).alias("active"),
#         pl.col("time").alias("start"),
#         pl.lit(None, pl.UInt32).alias("in_index"),
#         pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight_0"),
#         pl.lit(0.0, pl.Float64).alias("weight_1"),
#     )

# return spikes.with_columns(
#     pl.lit(None, pl.Boolean).alias("active"),
#     pl.lit(None, pl.UInt32).alias("in_index"),
#     pl.col("time").alias("start"),
#     pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight_0"),
#     pl.lit(0.0, pl.Float64).alias("weight_1"),
#     pl.lit(REFRACTORY_RESET, pl.Float64).alias("coef_0"),
#     pl.lit(0.0, pl.Float64).alias("coef_1"),
# ).select(
#     "index",
#     "neuron",
#     "f_index",
#     "active",
#     "start",
#     "in_index",
#     "weight_0",
#     "weight_1",
#     "coef_0",
#     "coef_1",
# )


# def create_synaptic_states(spikes, synapses, origins):
#     return (
#         synapses.join(spikes, left_on="source", right_on="neuron")
#         .join(origins, left_on=["index", "target"], right_on=["index", "neuron"])
#         .select(
#             pl.col("index"),
#             pl.col("target").alias("neuron"),
#             pl.lit(None, pl.UInt32).alias("f_index"),
#             pl.lit(None, pl.Boolean).alias("active"),
#             modulo_with_offset(
#                 pl.col("time") + pl.col("delay"),
#                 pl.col("period"),
#                 pl.col("time_origin"),
#             ).alias("start"),
#             pl.col("in_index"),
#             pl.col("weight_0"),
#             pl.col("weight_1"),
#         )
#     )


# def create_offline_states(spikes, synapses, eps):
#     """Returns states with transformed spikes and synapses."""
#     if eps < 0:
#         raise ValueError("epsilon must be positive.")

#     spikes = extend_with_time_prev(spikes, over=["index", "neuron"])  # result is sorted
#     spikes = spikes.with_row_index("f_index")

#     ## Refractoriness
#     rec_states = create_recovery_states(
#         spikes.drop("time").rename({"time_prev": "time"})
#     )

#     ## Synaptic transmission
#     # 0. Compute origins for each neuron
#     origins = spikes.group_by("index", "neuron").agg(
#         pl.min("time_prev").alias("time_origin")
#     )

#     # 1. Extract synapses to spiking neurons (the other synapses can be ignored, i.e., have weights = 0.0)
#     synapses = synapses.join(spikes, left_on="target", right_on="neuron", how="semi")
#     synapses = synapses.with_row_index("in_index")

#     # 2. Create synaptic states
#     syn_states = create_synaptic_states(spikes, synapses, origins)

#     ## Virtual states for specific time marks
#     v_states = (
#         spikes.with_columns(
#             (pl.col("time") + [-eps, 0.0]).alias("start"),
#         )
#         .explode("start")
#         .select(
#             pl.col("index"),
#             pl.col("neuron"),
#             pl.col("f_index"),
#             pl.lit(True, pl.Boolean).alias("active"),
#             pl.col("start").clip(pl.col("time_prev")),
#             pl.lit(None, pl.UInt32).alias("in_index"),
#             pl.lit(0.0, pl.Float64).alias("weight_0"),
#             pl.lit(0.0, pl.Float64).alias("weight_1"),
#         )
#     )

#     # Merge all states
#     states = pl.concat([rec_states, syn_states, v_states])

#     # Sort and update interval information
#     # Note: the first state in each group (indexed by f_index) is a refractory state
#     states = states.sort(["start", "f_index"])
#     states = states.with_columns(
#         pl.col("f_index").forward_fill().over(["index", "neuron"])
#     )
#     states = states.with_columns(
#         pl.col("active").forward_fill().fill_null(False).over("f_index")
#     )
#     states = states.with_columns(
#         pl.col("start").diff().shift(-1, fill_value=0.0).over("f_index").alias("length")
#     )

#     return spikes, synapses, states


# def init_offline_optimization(spikes, synapses):
#     """Returns states with transformed spikes and synapses."""
#     synapses = synapses.with_row_index("in_index")

#     out_spikes = (
#         spikes.join(synapses, left_on="neuron", right_on="target", how="semi")
#         .sort("time")
#         .with_row_index("f_index")
#     )

#     out_spikes = out_spikes.with_columns(
#         modulo_with_offset(
#             pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
#             pl.col("period"),
#             pl.col("time") - pl.col("period"),
#         )
#         .over(["index", "neuron"])
#         .alias("time_prev")
#     )

#     ## Refractoriness
#     rec_states = out_spikes.select(
#         pl.col("index"),
#         pl.col("neuron"),
#         pl.col("f_index"),
#         pl.col("time_prev").alias("start"),
#         pl.lit(None, pl.UInt32).alias("in_index"),
#         pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight_0"),
#         pl.lit(0.0, pl.Float64).alias("weight_1"),
#     )

#     ## Synaptic transmission
#     origins = out_spikes.group_by(["index", "neuron"]).agg(
#         pl.min("time_prev").alias("time_origin")
#     )
#     syn_states = (
#         synapses.join(spikes, left_on="source", right_on="neuron")
#         .join(origins, left_on=["index", "target"], right_on=["index", "neuron"])
#         .select(
#             pl.col("index"),
#             pl.col("target").alias("neuron"),
#             pl.lit(None, pl.UInt32).alias("f_index"),
#             modulo_with_offset(
#                 pl.col("time") + pl.col("delay"),
#                 pl.col("period"),
#                 pl.col("time_origin"),
#             ).alias("start"),
#             pl.col("in_index"),
#             pl.lit(0.0, pl.Float64).alias("weight_0"),
#             pl.lit(1.0, pl.Float64).alias("weight_1"),
#         )
#     )

#     ## Virtual states at firing times
#     v_states = (
#         out_spikes.join(synapses, left_on="neuron", right_on="target")
#         .select(
#             pl.col("index"),
#             pl.col("neuron"),
#             pl.col("f_index"),
#             pl.col("time").alias("start"),
#             pl.col("in_index"),
#             pl.lit(0.0, pl.Float64).alias("weight_0"),
#             pl.lit(0.0, pl.Float64).alias("weight_1"),
#         )
#         .extend(
#             out_spikes.select(
#                 pl.col("index"),
#                 pl.col("neuron"),
#                 pl.col("f_index"),
#                 pl.col("time").alias("start"),
#                 pl.lit(None, pl.UInt32).alias("in_index"),
#                 pl.lit(0.0, pl.Float64).alias("weight_0"),
#                 pl.lit(0.0, pl.Float64).alias("weight_1"),
#             )
#         )
#     )

#     # Merge all states
#     states = pl.concat([rec_states, syn_states, v_states])

#     # Sort and update interval information
#     # Note: the first state in each group (indexed by f_index) is a refractory state
#     states = states.sort(["start", "f_index"])
#     states = states.with_columns(
#         pl.col("f_index").forward_fill().over("index", order_by="start")
#     )
#     states = states.with_columns(
#         pl.col("start")
#         .diff()
#         .shift(-1, fill_value=0.0)
#         .over(["f_index", "in_index"], order_by="start")
#         .alias("length")
#     )

#     return out_spikes, synapses, states


def compute_energy_metrics(
    syn_states,
    n_synapses,
    rec_states,
    out_spikes,
    syn_loc,
):
    """_summary_

    syn_states contains: index, f_index, in_index, time, in_coef_0, in_coef_1
    rec_states contains: index, f_index, time, in_coef_0, in_coef_1
    f_states contains: index, f_index, time
    periods contains: index, period

    syn_local: True or False (default True). If True, use synapse-local (=diagonal) energy metric. Otherwise, use neuron-local (=full) energy metric.
    reset: True or False (default True). If True, reset at each spike.
    recovery: True or False (default False). If True, use the refractory reset (=weighted mean) in the energy metric.


    Args:
        out_spikes (_type_): _description_
        synapses (_type_): _description_. must have in_index column
        in_states (_type_): _description_
        which (_type_): _description_.

    Returns:
        _type_: _description_
    """
    # over = ["index"]
    # if syn_local:
    #     over.append("in_index")
    # if reset:
    #     over.append("f_index")

    if n_synapses is None:
        n_synapses = syn_states.select(pl.col("in_index").max()).item() + 1

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
                    * pl.col("weight_right")
                )
                .sum()
                .alias("coef")
            )
        )
        weighted_mean = dataframe_to_1d_array(
            rec_to_syn_metric, "in_index", "coef", n_synapses
        )

    return weighted_mean, precision

    # if which == "activity-agnostic":
    #     weighted_mean = np.zeros(synapses.height)
    #     precision = ss.eye_array(synapses.height)

    # elif which == "synapse-local":
    #     # Note that we do not need to consider the delays here since the metric is synapse-local
    #     syn_states = spikes.join(synapses, left_on="neuron", right_on="source").select(
    #         pl.col("index"),
    #         pl.col("in_index"),
    #         pl.col("time"),
    #     )
    #     syn_to_syn_metric = (
    #         syn_states.join(syn_states, on=["index", "in_index"])
    #         .join(periods, on="index")
    #         .with_columns(
    #             pl.min_horizontal(
    #                 (pl.col("time_right") - pl.col("time")).mod(pl.col("period")),
    #                 ((pl.col("time") - pl.col("time_right")).mod(pl.col("period"))),
    #             ).alias("delta"),
    #             pl.lit(None, pl.Float64).alias("duration"),
    #         )
    #         .with_columns(
    #             rp.energy_syn_to_syn_metric(
    #                 pl.col("delta"),
    #                 pl.col("duration"),
    #             ).alias("coef")
    #         )
    #         .group_by("in_index")
    #         .agg(pl.sum("coef"))
    #     )

    #     weighted_mean = np.zeros(synapses.height)
    #     precision = dataframe_to_2d_array(
    #         syn_to_syn_metric,
    #         "in_index",
    #         "in_index",
    #         "coef",
    #         (synapses.height, synapses.height),
    #     )

    # elif which == "neuron-local-no-recovery":
    #     logger.warning("neuron-local-no-recovery optimization is extremely slow...")

    #     syn_states = spikes.join(synapses, left_on="neuron", right_on="source").select(
    #         pl.col("index"),
    #         pl.col("in_index"),
    #         pl.col("time") + pl.col("delay"),
    #     )

    #     syn_to_syn_metric = (
    #         syn_states.join(syn_states, on="index")
    #         .join(periods, on="index")
    #         .with_columns(
    #             pl.min_horizontal(
    #                 (pl.col("time_right") - pl.col("time")).mod(pl.col("period")),
    #                 ((pl.col("time") - pl.col("time_right")).mod(pl.col("period"))),
    #             ).alias(
    #                 "delta"
    #             ),  #
    #             pl.lit(None, pl.Float64).alias("duration"),
    #         )
    #         .with_columns(
    #             rp.energy_syn_to_syn_metric(
    #                 pl.col("delta"),
    #                 pl.col("duration"),
    #             ).alias("coef")
    #         )
    #         .group_by("in_index", "in_index_right")
    #         .agg(pl.sum("coef"))
    #     )

    #     weighted_mean = np.zeros(synapses.height)
    #     precision = dataframe_to_2d_array(
    #         syn_to_syn_metric,
    #         "in_index",
    #         "in_index_right",
    #         "coef",
    #         (synapses.height, synapses.height),
    #     )

    # else:
    #     raise NotImplementedError("neuron-local rules are not implemented yet.")

    # rec_states = in_states.filter(pl.col("in_index").is_null()).select(
    #     "f_index", "start", "weight"
    # )
    # syn_states = in_states.filter(pl.col("in_index").is_not_null()).select(
    #     "f_index", "start", "in_index"
    # )

    # rec_to_syn_metric = (
    #     rec_states.join(syn_states, on="f_index", suffix="_other")
    #     .join(out_spikes.select("f_index", "time"), on="f_index")
    #     .with_columns(
    #         rp.energy_rec_to_syn_metric("start", "start_other", "time").alias("coef")
    #     )
    #     .group_by("in_index")
    #     .agg((pl.col("coef") * pl.col("weight")).sum())
    # )

    # syn_to_syn_metric = (
    #     syn_states.join(syn_states, on="f_index", suffix="_other")
    #     .join(out_spikes.select("f_index", "time"), on="f_index")
    #     .with_columns(
    #         rp.energy_syn_to_syn_metric("start", "start_other", "time").alias("coef")
    #     )
    #     .group_by("in_index", "in_index_other")
    #     .agg(pl.sum("coef"))
    # )

    # weighted_mean = dataframe_to_1d_array(
    #     rec_to_syn_metric, "in_index", "coef", synapses.height
    # )
    # precision = dataframe_to_2d_array(
    #     syn_to_syn_metric,
    #     "in_index",
    #     "in_index_other",
    #     "coef",
    #     (synapses.height, synapses.height),
    # )

    # return weighted_mean, precision


# def init_energy_based_optimization(spikes, synapses):
#     return init_offline_optimization(spikes, synapses)

# # active states
# v_states = (
#     states.select("f_index", "in_index")
#     # .filter(pl.col("in_index").is_not_null() | (pl.col("weight").abs() > 0.0))
#     .unique().join(
#         out_spikes.select(
#             pl.col("index"),
#             pl.col("neuron"),
#             pl.col("f_index"),
#             (pl.col("time") - eps).clip(pl.col("time_prev")).alias("start"),
#             pl.lit(True, pl.Boolean).alias("active"),
#         ),
#         on="f_index",
#     )
#     # .with_columns(
#     #     (pl.col("")).over(["f_index", "in_index"]).alias("coef_0"),
#     #     pl.lit(0.0, pl.Float64).alias("coef_1"),
#     # )
# )

# states = (
#     states.match_to_schema(v_states.schema, missing_columns="insert")
#     .extend(v_states)
#     .with_columns(
#         pl.col("active")
#         .forward_fill()
#         .fill_null(False)
#         .over("f_index", order_by="start")
#     )
# ).sort("start")

# states = scan_states(states, over=["f_index", "in_index"])

# synapses = synapses.with_row_index("in_index")

# out_spikes = spikes.sort("time").with_row_index("f_index")

# out_spikes = out_spikes.with_columns(
#     modulo_with_offset(
#         pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
#         pl.col("period"),
#         pl.col("time") - pl.col("period"),
#     )
#     .over(["index", "neuron"])
#     .alias("time_prev")
# )

# ## Refractoriness
# rec_states = out_spikes.select(
#     pl.col("index"),
#     pl.col("neuron"),
#     pl.col("f_index"),
#     pl.lit(None, pl.Boolean).alias("active"),
#     pl.col("time_prev").alias("start"),
#     pl.lit(None, pl.UInt32).alias("in_index"),
#     pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight"),
#     pl.lit(1.0, pl.Float64).alias("coef_0"),
#     pl.lit(0.0, pl.Float64).alias("coef_1"),
# )

# ## Synaptic transmission
# origins = out_spikes.group_by(["index", "neuron"]).agg(
#     pl.min("time_prev").alias("time_origin")
# )
# syn_states = (
#     synapses.join(spikes, left_on="source", right_on="neuron")
#     .join(origins, left_on=["index", "target"], right_on=["index", "neuron"])
#     .select(
#         pl.col("index"),
#         pl.col("target").alias("neuron"),
#         pl.lit(None, pl.UInt32).alias("f_index"),
#         pl.lit(None, pl.Boolean).alias("active"),
#         modulo_with_offset(
#             pl.col("time") + pl.col("delay"),
#             pl.col("period"),
#             pl.col("time_origin"),
#         ).alias("start"),
#         pl.col("in_index"),
#         pl.lit(None, pl.Float64).alias("weight"),
#         pl.lit(0.0, pl.Float64).alias("coef_0"),
#         pl.lit(1.0, pl.Float64).alias("coef_1"),
#     )
# )

# ## Virtual states for specific time marks
# v_states = (
#     out_spikes.with_columns(
#         (pl.col("time") + [-eps, 0.0]).alias("start"),
#     )
#     .explode("start")
#     .select(
#         pl.col("index"),
#         pl.col("neuron"),
#         pl.col("f_index"),
#         pl.lit(True, pl.Boolean).alias("active"),
#         pl.col("start").clip(pl.col("time_prev")),
#         pl.lit(None, pl.UInt32).alias("in_index"),
#         pl.lit(0.0, pl.Float64).alias("weight_0"),
#         pl.lit(0.0, pl.Float64).alias("weight_1"),
#     )
# )

# # Merge all states
# states = pl.concat([rec_states, syn_states, v_states])

# # Sort and update interval information
# # Note: the first state in each group (indexed by f_index) is a refractory state
# states = states.sort(["start", "f_index"])
# states = states.with_columns(pl.col("f_index").forward_fill().over(["index"]))
# states = states.with_columns(
#     pl.col("active").forward_fill().fill_null(False).over("f_index")
# )
# states = states.with_columns(
#     pl.col("start").diff().shift(-1, fill_value=0.0).over("f_index").alias("length")
# )

# return (
#     out_spikes,
#     synapses,
#     states.filter(pl.col("active")).sort("start"),
#     states.filter(~pl.col("active")).sort("start"),
# )


# def compute_syn_energy(states):
#     return (
#         states.select(
#             pl.col("in_index"),
#             (
#                 (
#                     2 * pl.col("in_coef_0") ** 2
#                     + 2 * pl.col("in_coef_0") * pl.col("in_coef_1")
#                     + pl.col("in_coef_1") ** 2
#                 )
#                 - (
#                     2 * pl.col("in_coef_0") ** 2
#                     + pl.col("in_coef_1")
#                     * (2 * pl.col("in_coef_0") + pl.col("in_coef_1"))
#                     * (1 + 2 * pl.col("length"))
#                     + 2 * pl.col("in_coef_1") ** 2 * pl.col("length") ** 2
#                 )
#                 * (-2 * pl.col("length")).exp()
#             ).alias("energy"),
#         )
#         .group_by("in_index")
#         .agg(pl.sum("energy").alias("syn_energy"))
#     )


# def computes_syn_energy_metric_matrix(states, synapses):
#     syn_energy = compute_syn_energy(states)
#     return sparse.csr_array(
#         (
#             (syn_energy.get_column("syn_energy").to_numpy()),
#             (
#                 syn_energy.get_column("in_index").to_numpy(),
#                 syn_energy.get_column("in_index").to_numpy(),
#             ),
#         ),
#         shape=(synapses.height, synapses.height),
#     )


# def update_weights(dataframe, weights):
#     # update synapses with given weights
#     return dataframe.update(
#         pl.DataFrame({"weight": weights}).with_row_index("in_index"),
#         on="in_index",
#     )


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
    """
    which is one of
    - activity-agnostic: identity precision, zero (weighted) mean
    - synapse-local: diagonal precision, zero (weighted) mean
    - neuron-local-no-recovery: full precision, no reset
    - neuron-local: full precision, with reset

    syn_local: True or False (default True). If True, use synapse-local (=diagonal) energy metric. Otherwise, use neuron-local (=full) energy metric.
    reset: True or False (default True). If True, reset at each spike.
    recovery: True or False (default False). If True, use the refractory reset (=weighted mean) in the energy metric.
    """
    if wmin >= wmax:
        raise ValueError("wmin must be less than wmax.")

    synapses_lst = []

    for (neuron,), in_synapses in synapses.partition_by("target", as_dict=True).items():
        # Prepare synapses and output spikes for the current neuron
        in_synapses = init_synapses(in_synapses, spikes)
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
            syn_states,
            in_synapses.height,
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
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time"),
            in_synapses.height,
        )
        model.addConstr(syn_lin_map @ weights + rec_lin_offset == FIRING_THRESHOLD)

        syn_lin_map, rec_lin_offset = compute_linear_map(
            syn_states,
            rec_states,
            out_spikes.select("f_index", "time"),
            in_synapses.height,
            deriv=1,
        )
        model.addConstr(syn_lin_map @ weights + rec_lin_offset >= dzmin)
        logger.debug(f"Neuron {neuron}. Firing constraints added.")

        model.optimize()
        if model.status != gp.GRB.OPTIMAL:
            logger.warning(
                f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
            )
            return

        logger.info(f"Neuron {neuron}. Optimization completed!")
        synapses_lst.append(
            in_synapses.update(
                pl.DataFrame({"weight": weights.X}).with_row_index("in_index"),
                on="in_index",
            ).drop("in_index")
        )

    return pl.concat(synapses_lst)

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

    # def continuous_optimization(
    #     spikes,
    #     synapses,
    #     # states,
    #     wmin=float("-inf"),
    #     wmax=float("inf"),
    #     eps=0.2,
    #     zmax=0.0,
    #     dzmin=1.0,
    #     n_iter=1000,
    #     feas_tol=1e-5,
    # ):
    #     # spikes is a dataframe with columns: index, period, neuron, time, time_prev
    #     # synapses is a dataframe with columns: in_index, source, target, delay, weight_0, weight_1
    #     # states is a dataframe with columns: index, neuron, active, start, length, in_index, weight_0, weight_1, coef_0, coef_1

    #     if eps < 0:
    #         raise ValueError("epsilon must be positive.")

    #     if zmax > FIRING_THRESHOLD:
    #         raise ValueError("zmax must be less than or equal to the firing threshold.")

    #     if dzmin < 0:
    #         raise ValueError("dzmin must be non-negative.")

    #     if wmin >= wmax:
    #         raise ValueError("wmin must be less than wmax.")

    #     synapses_lst = []

    #     for (neuron,), in_synapses in synapses.partition_by("target", as_dict=True).items():
    #         out_spikes, in_synapses, active_states, silent_states = (
    #             init_continuous_optimization(spikes, in_synapses, eps)
    #         )

    #         ## Initialize Gurobi model with variables, objective, and initial constraints (at firing times)
    #         model = gp.Model("model")
    #         model.setParam("OutputFlag", 0)  # Disable output

    #         # Setup variables to be optimized = the synaptic weights
    #         weights = model.addMVar(shape=in_synapses.height, lb=wmin, ub=wmax)

    #         # Objective function
    #         model.setObjective(
    #             weights @ weights,
    #             sense=gp.GRB.MINIMIZE,
    #         )

    #         # Firing constraints
    #         logger.warning(f"active_states: {active_states}")
    #         syn_lin_map, rec_lin_offset = compute_linear_map(
    #             active_states, out_spikes.select("f_index", "neuron", "time"), in_synapses
    #         )
    #         model.addConstr(syn_lin_map @ weights + rec_lin_offset == FIRING_THRESHOLD)

    #         for i in range(n_iter):
    #             # 1. Optimize weights
    #             model.optimize()

    #             if model.status != gp.GRB.OPTIMAL:
    #                 logger.warning(
    #                     f"Neuron {neuron}. Optimization failed: {GUROBI_STATUS[model.status]}"
    #                 )
    #                 return

    #             in_synapses = update_weights(in_synapses, weights.X)
    #             active_states = update_weights(active_states, weights.X)
    #             silent_states = update_weights(silent_states, weights.X)

    #             # 2. Refine constraints
    #             silent_violations = (
    #                 silent_states.group_by("f_index", "neuron")
    #                 .agg(
    #                     rp.max_violation(
    #                         pl.col("start"),
    #                         pl.col("length"),
    #                         pl.col("length").shift(),
    #                         pl.col("weight"),
    #                         pl.col("coef_0"),
    #                         pl.col("coef_1"),
    #                         vmin=feas_tol + zmax,
    #                     ).alias("time")
    #                 )
    #                 .filter(pl.col("time").is_not_nan())
    #             )

    #             if silent_violations.height > 0:
    #                 syn_lin_map, rec_lin_offset = compute_linear_map(
    #                     silent_states,
    #                     silent_violations,
    #                     in_synapses,
    #                 )
    #                 model.addConstr(
    #                     syn_lin_map @ weights + rec_lin_offset <= zmax
    #                 )  # silent area: z <= zmax

    #             active_violations = (
    #                 active_states.group_by("f_index", "neuron")
    #                 .agg(
    #                     rp.max_violation(
    #                         pl.col("start"),
    #                         pl.col("length"),
    #                         pl.col("length").shift(),
    #                         pl.col("weight"),
    #                         (pl.col("coef_0") - pl.col("coef_1")),
    #                         pl.col("coef_1"),
    #                         vmin=feas_tol - dzmin,
    #                     ).alias("time")
    #                 )
    #                 .filter(pl.col("time").is_not_nan())
    #             )
    #             if active_violations.height > 0:
    #                 syn_lin_map, rec_lin_offset = compute_linear_map(
    #                     active_states,
    #                     active_violations,
    #                     in_synapses,
    #                     deriv=1,
    #                 )
    #                 model.addConstr(
    #                     syn_lin_map @ weights + rec_lin_offset >= dzmin
    #                 )  # active area: dz >= dzmin

    #             n_violations = active_violations.height + silent_violations.height

    #             logger.debug(
    #                 f"Neuron {neuron}: iteration {i}. The objective is {model.getAttr('ObjVal')} for {model.getAttr('NumConstrs')} linear constraints. Still {n_violations} linear potential constraints to add."
    #             )

    #             if n_violations == 0:
    #                 logger.info(
    #                     f"Neuron {neuron}. Optimization successful (in {i} iterations)"
    #                 )
    #                 synapses_lst.append(in_synapses.drop("in_index"))
    #                 break

    #         # logger.warning(f"Optimization stopped after {n_iter} iterations")
    #         # return
    #     return pl.concat(synapses_lst)

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

    # def compute_linear_map(states, times, synapses, deriv=0):
    #     times = times.with_row_index("t_index")

    #     syn_coef = (
    #         times.join(
    #             synapses.select("in_index", "target"),
    #             left_on="neuron",
    #             right_on="target",
    #         )
    #         .sort("time")
    #         .join_asof(
    #             states.filter(pl.col("in_index").is_not_null())
    #             .select("f_index", "start", "in_index", "cum_coef_0", "cum_coef_1")
    #             .sort("start"),
    #             left_on="time",
    #             right_on="start",
    #             by=["f_index", "in_index"],
    #             check_sortedness=False,
    #         )
    #         .drop_nulls("start")
    #         .select(
    #             pl.col("in_index"),
    #             pl.col("t_index"),
    #             (
    #                 (-1) ** (deriv % 2)
    #                 * (
    #                     pl.col("cum_coef_0")
    #                     - deriv * pl.col("cum_coef_1")
    #                     + pl.col("cum_coef_1") * (pl.col("time") - pl.col("start"))
    #                 )
    #                 * (pl.col("start") - pl.col("time")).exp()
    #             ).alias("coef"),
    #         )
    #     )

    #     rec_offset = (
    #         times.sort("time")
    #         .join_asof(
    #             states.filter(pl.col("in_index").is_null()).select(
    #                 "f_index", "start", "in_index", "weight", "cum_coef_0", "cum_coef_1"
    #             ),
    #             left_on="time",
    #             right_on="start",
    #             by="f_index",
    #             check_sortedness=False,
    #         )
    #         .drop_nulls("start")
    #         .select(
    #             pl.col("t_index"),
    #             (
    #                 (-1) ** (deriv % 2)
    #                 * pl.col("weight")
    #                 * (
    #                     pl.col("cum_coef_0")
    #                     - deriv * pl.col("cum_coef_1")
    #                     + pl.col("cum_coef_1") * (pl.col("time") - pl.col("start"))
    #                 )
    #                 * (pl.col("start") - pl.col("time")).exp()
    #             ).alias("coef"),
    #         )
    #     )

    #     return (
    #         sparse.csr_array(
    #             (
    #                 (syn_coef.get_column("coef").to_numpy()),
    #                 (
    #                     syn_coef.get_column("t_index").to_numpy(),
    #                     syn_coef.get_column("in_index").to_numpy(),
    #                 ),
    #             ),
    #             shape=(times.height, synapses.height),
    #         ),
    #         rec_offset.sort("t_index").get_column("coef").to_numpy(),
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
    return states


# def scan_coef(states, over):
#     return (
#         states.with_columns(
#             pl.col("start")
#             .diff()
#             .shift(-1)
#             .over(over, order_by="start")
#             .alias("length")
#         )
#         .drop_nulls("length")
#         .with_columns(
#             rp.scan_coef_1(pl.col("length").shift(), pl.col("coef_1"))
#             .over(over, order_by="start")
#             .alias("cum_coef_1")
#         )
#         .with_columns(
#             rp.scan_coef_0(
#                 pl.col("length").shift(),
#                 pl.col("cum_coef_1").shift(),
#                 pl.col("coef_0"),
#             )
#             .over(over, order_by="start")
#             .alias("cum_coef_0")
#         )
#     )
