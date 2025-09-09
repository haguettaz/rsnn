import numpy as np
import polars as pl
import scipy.sparse as ss

import rsnn_plugin as rp
from rsnn import REFRACTORY_RESET
from rsnn.log import setup_logging

# from ..states import compute_max_violations

logger = setup_logging(__name__, console_level="DEBUG", file_level="DEBUG")


# def add_spikes(spikes):
#     """Initialize spikes with: previous firing time (with periodic extension) and (unique) firing time index. The resulting spikes are sorted by 1) index and 2) time."""
#     if "index" in spikes.columns:
#         return spikes.sort("index", "time").with_columns(
#             pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
#             modulo_with_offset(
#                 pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
#                 pl.col("period"),
#                 pl.col("time") - pl.col("period"),
#             )
#             .over(["index", "neuron"])
#             .alias("time_prev"),
#         )

#     return spikes.sort("time").with_columns(
#         pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
#         modulo_with_offset(
#             pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
#             pl.col("period"),
#             pl.col("time") - pl.col("period"),
#         )
#         .over("neuron")
#         .alias("time_prev"),
#     )


# def init_spikes(spikes, periods):
#     """Add memory index, previous firing time (with periodic extension) and (unique) firing time index to spikes. The resulting spikes are sorted by 1) index and 2) time."""
#     if ("index" in spikes.columns) != ("index" in periods.columns):
#         raise ValueError(
#             "spikes and periods must both contain or both not contain the index column."
#         )

#     if "index" not in spikes.columns:
#         spikes = spikes.with_columns(pl.lit(0, pl.UInt32).alias("index"))
#         periods = periods.with_columns(pl.lit(0, pl.UInt32).alias("index"))

#     spikes = (
#         spikes.sort("index", "time")
#         .join(periods, on="index")
#         .with_columns(
#             pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
#             modulo_with_offset(
#                 pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
#                 pl.col("period"),
#                 pl.col("time") - pl.col("period"),
#             )
#             .over(["index", "neuron"])
#             .alias("time_prev"),
#         )
#     )

#     return spikes, periods


def init_synapses(synapses, spikes):
    """Prune synapses to keep only those that have a source neuron that spikes at least once. Add unique in_index to each synapse.

    Args:
        synapses (_type_): _description_
        spikes (_type_): _description_

    Returns:
        _type_: _description_
    """
    return synapses.join(
        spikes, left_on="source", right_on="neuron", how="semi"
    ).with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("in_index"))


def init_out_spikes(spikes):
    return spikes.sort("time").select(
        pl.col("index"),
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
        pl.col("neuron"),
        pl.col("time"),
        modulo_with_offset(
            pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
            pl.col("period"),
            pl.col("time") - pl.col("period"),
        )
        .over(["index", "neuron"])
        .alias("time_prev"),
    )


def compute_states(synapses, out_spikes, src_spikes):
    # Origins per (index, neuron)
    origins = out_spikes.group_by(["index", "neuron"]).agg(
        pl.min("time_prev").alias("time_origin")
    )

    # Refractoriness
    rec_states = out_spikes.select(
        pl.col("index"),
        pl.col("neuron"),
        pl.col("f_index"),
        pl.col("time_prev").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
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
            pl.lit(0.0, pl.Float64).alias("in_coef_0"),
            pl.col("weight").alias("in_coef_1"),
        )
    )

    in_states = syn_states.extend(rec_states).select(
        pl.col("index"),
        pl.col("neuron"),
        pl.col("f_index").forward_fill().over(["index", "neuron"], order_by="start"),
        pl.col("start"),
        pl.col("in_index"),
        pl.col("in_coef_0"),
        pl.col("in_coef_1"),
    )

    f_states = out_spikes.select(
        pl.col("index"),
        pl.col("neuron"),
        pl.col("f_index"),
        pl.col("time").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
    )

    states = in_states.extend(f_states).sort("index", "neuron", "start")
    return scan_states(states)


def scan_states(states):
    return (
        states.with_columns(
            pl.col("start")
            .diff()
            .shift(-1, fill_value=0.0)
            .over("f_index", order_by="start")
            .alias("length")
        )
        .with_columns(
            rp.scan_coef_1(pl.col("length").shift(), pl.col("in_coef_1"))
            .over("f_index", order_by="start")
            .alias("coef_1")
        )
        .with_columns(
            rp.scan_coef_0(
                pl.col("length").shift(),
                pl.col("coef_1").shift(),
                pl.col("in_coef_0"),
            )
            .over("f_index", order_by="start")
            .alias("coef_0")
        )
    )


# def compute_in_states(synapses, out_spikes, source_spikes, periods):
#     """Returns:
#     - out_spikes: spikes of the post-synaptic neurons (with index, f_index, time_prev)
#     - synapses: synapses with unique in_index
#     - in_states: input states with (f_index, start, in_index, weight, in_coef_0, in_coef_1). Completely determine the fading traces at any time.
#     """
#     # Refractoriness
#     rec_states = out_spikes.select(
#         pl.col("index"),
#         pl.col("f_index"),
#         pl.col("time_prev").alias("start"),
#         pl.lit(None, pl.UInt32).alias("in_index"),
#         pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight"),
#         pl.lit(1.0, pl.Float64).alias("in_coef_0"),
#         pl.lit(0.0, pl.Float64).alias("in_coef_1"),
#     )

#     # Synaptic transmission
#     origins = out_spikes.group_by("index").agg(pl.min("time_prev").alias("time_origin"))
#     syn_states = (
#         synapses.join(source_spikes, left_on="source", right_on="neuron")
#         .join(origins, on="index")
#         .join(periods, on="index")
#         .select(
#             pl.col("index"),
#             pl.lit(None, pl.UInt32).alias("f_index"),
#             modulo_with_offset(
#                 pl.col("time") + pl.col("delay"),
#                 pl.col("period"),
#                 pl.col("time_origin"),
#             ).alias("start"),
#             pl.col("in_index"),
#             pl.col("weight"),
#             pl.lit(0.0, pl.Float64).alias("in_coef_0"),
#             pl.lit(1.0, pl.Float64).alias("in_coef_1"),
#         )
#     )

#     # Merge input states (refractoriness + synapses) and complete f_index information
#     in_states = syn_states.extend(rec_states).select(
#         pl.col("f_index").forward_fill().over("index", order_by="start"),
#         pl.col("start"),
#         pl.col("in_index"),
#         pl.col("weight"),
#         pl.col("in_coef_0"),
#         pl.col("in_coef_1"),
#     )

#     return in_states


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


def compute_linear_map(syn_states, rec_states, times, n_synapses, deriv=0):
    times = times.with_row_index("t_index")

    syn_coef = (
        times.join(syn_states, on="f_index")
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
        times.join(rec_states, on="f_index")
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
        ss.csr_array(
            (
                (syn_coef.get_column("coef").to_numpy()),
                (
                    syn_coef.get_column("t_index").to_numpy(),
                    syn_coef.get_column("in_index").to_numpy(),
                ),
            ),
            shape=(times.height, n_synapses),
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
    # return x - period * (x - offset).floordiv(period)
    # return x - offset - period * (x - offset).floordiv(period) + offset
    return (x - offset).mod(period) + offset


def dataframe_to_1d_array(dataframe, index, value, shape):
    arr = np.zeros(shape)
    arr[dataframe.get_column(index).to_numpy()] = dataframe.get_column(value).to_numpy()
    return arr


def dataframe_to_sym_2d_array(dataframe, index_row, index_col, value, shape):
    array = ss.csr_array(
        (
            (dataframe.get_column(value).to_numpy()),
            (
                dataframe.get_column(index_row).to_numpy(),
                dataframe.get_column(index_col).to_numpy(),
            ),
        ),
        shape=shape,
    )
    return array + array.T - ss.diags(array.diagonal())
