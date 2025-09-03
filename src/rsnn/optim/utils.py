import polars as pl
import scipy.sparse as sparse

import rsnn_plugin as rp
from rsnn import REFRACTORY_RESET
from rsnn.log import setup_logging

# from ..states import compute_max_violations

logger = setup_logging(__name__, console_level="DEBUG", file_level="DEBUG")


def init_spikes(spikes):
    """Initialize spikes with: previous firing time (with periodic extension) and (unique) firing time index. The resulting spikes are sorted by 1) index and 2) time."""
    if "index" in spikes.columns:
        return spikes.sort("index", "time").with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
            modulo_with_offset(
                pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
                pl.col("period"),
                pl.col("time") - pl.col("period"),
            )
            .over(["index", "neuron"])
            .alias("time_prev"),
        )

    return spikes.sort("time").with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
        modulo_with_offset(
            pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
            pl.col("period"),
            pl.col("time") - pl.col("period"),
        )
        .over("neuron")
        .alias("time_prev"),
    )


def init_offline_optimization(spikes, synapses):
    synapses = synapses.join(
        spikes, left_on="source", right_on="neuron", how="semi"
    )  # prune synapses that do not transmit any spike
    synapses = synapses.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("in_index")
    )

    # Init output spikes of the neurons whose synapses are optimized
    out_spikes = init_spikes(
        spikes.join(synapses, left_on="neuron", right_on="target", how="semi")
    )

    if "index" in out_spikes.columns:
        # Refractoriness
        rec_states = out_spikes.select(
            pl.col("index"),
            pl.col("neuron"),
            pl.col("f_index"),
            pl.col("time_prev").alias("start"),
            pl.lit(None, pl.UInt32).alias("in_index"),
            pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight"),
            pl.lit(1.0, pl.Float64).alias("in_coef_0"),
            pl.lit(0.0, pl.Float64).alias("in_coef_1"),
        )

        # Synaptic transmission
        origins = out_spikes.group_by(["index", "neuron"]).agg(
            pl.first("time_prev").alias("time_origin")
        )
        syn_states = (
            synapses.join(spikes, left_on="source", right_on="neuron")
            .join(origins, left_on=["index", "target"], right_on=["index", "neuron"])
            .select(
                pl.col("index"),
                pl.col("target").alias("neuron"),
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

        # Merge input states (refractoriness + synapses) and complete f_index information
        in_states = (
            syn_states.extend(rec_states)
            .sort("start")
            .with_columns(pl.col("f_index").forward_fill().over(["index", "neuron"]))
            .drop("index")
        )

        return out_spikes, synapses, in_states

    # Refractoriness
    rec_states = out_spikes.select(
        pl.col("neuron"),
        pl.col("f_index"),
        pl.col("time_prev").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight"),
        pl.lit(1.0, pl.Float64).alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    )

    # Synaptic transmission
    origins = out_spikes.group_by("neuron").agg(
        pl.first("time_prev").alias("time_origin")
    )
    syn_states = (
        synapses.join(spikes, left_on="source", right_on="neuron")
        .join(origins, left_on="target", right_on="neuron")
        .select(
            pl.col("target").alias("neuron"),
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

    # Merge input states (refractoriness + synapses)
    in_states = syn_states.extend(rec_states)

    # Sort and update interval information
    # Note: every group of states with the same f_index starts with a recovery state
    in_states = in_states.sort("start").with_columns(
        pl.col("f_index").forward_fill().over("neuron")
    )

    return out_spikes, synapses, in_states


# def scan_coef(states, over):
#     """WARNING: states must be sorted by start in the grouping provided by over."""
#     return states.with_columns(
#         rp.scan_coef_1(pl.col("length").shift(), pl.col("coef_1"))
#         .over(over)
#         .alias("scan_coef_1")
#     ).with_columns(
#         rp.scan_coef_0(
#             pl.col("length").shift(),
#             pl.col("scan_coef_1").shift(),
#             pl.col("coef_0"),
#         )
#         .over(over)
#         .alias("scan_coef_0")
#     )


def compute_linear_map(states, times, synapses, deriv=0):
    times = times.with_row_index("t_index")

    syn_coef = (
        times.join(states.filter(pl.col("in_index").is_not_null()), on="f_index")
        .filter(pl.col("time") >= pl.col("start"))
        .group_by("t_index", "in_index")
        .agg(
            (
                (
                    (-1) ** (deriv % 2)
                    * (
                        pl.col("in_coef_0")
                        - deriv * pl.col("in_coef_1")
                        + pl.col("in_coef_1") * (pl.col("time") - pl.col("start"))
                    )
                    * (pl.col("start") - pl.col("time")).exp()
                ).sum()
            ).alias("coef"),
        )
    )

    rec_offset = (
        times.join(states.filter(pl.col("in_index").is_null()), on="f_index")
        .filter(pl.col("time") >= pl.col("start"))
        .group_by("t_index")
        .agg(
            (
                (
                    (-1) ** (deriv % 2)
                    * pl.col("weight")
                    * (
                        pl.col("in_coef_0")
                        - deriv * pl.col("in_coef_1")
                        + pl.col("in_coef_1") * (pl.col("time") - pl.col("start"))
                    )
                    * (pl.col("start") - pl.col("time")).exp()
                ).sum()
            ).alias("coef"),
        )
    )

    return (
        sparse.csr_array(
            (
                (syn_coef.get_column("coef").to_numpy()),
                (
                    syn_coef.get_column("t_index").to_numpy(),
                    syn_coef.get_column("in_index").to_numpy(),
                ),
            ),
            shape=(times.height, synapses.height),
        ),
        rec_offset.sort("t_index").get_column("coef").to_numpy(),
    )


# def update_weights(dataframe, weights):
#     # update synapses with given weights
#     return dataframe.update(
#         pl.DataFrame({"weight": weights}).with_row_index("in_index"),
#         on="in_index",
#     )


def modulo_with_offset(x, period, offset):
    return x - period * (x - offset).floordiv(period)
