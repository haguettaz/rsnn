import numpy as np
import polars as pl
import scipy.sparse as ss
from numba import njit

from .constants import REFRACTORY_RESET
from .log import setup_logging
from .spikes import extend_with_time_prev
from .utils import modulo_with_offset

# Set up logging
logger = setup_logging(__name__, console_level="DEBUG", file_level="INFO")


@njit
def compute_phi_matrix(A, n_spikes):
    Phi = np.identity(n_spikes, dtype=np.float64)
    for n in range(n_spikes):
        Phi[n] = A[n] @ Phi
    Phi -= 1 / n_spikes
    return Phi


def compute_phi_eigenvals(spikes, synapses, k=3):
    # Spikes is a dataframe with columns: index, period, neuron, origin, time, prev_time
    # Synapses is a dataframe with columns: source, target, delay, weight
    # States is a dataframe with columns: f_index_source, f_index_target, f_time_in_target (=f_time_out_source + delay), f_time_out_target, weight_0, weight_1 (index, period, f_time_out_source, and delay are optional)
    phis = {}
    for (i,), spikes_i in spikes.partition_by(
        "index", include_key=False, as_dict=True
    ).items():
        ## Extend spikes with additional information
        spikes_i = extend_with_time_prev(spikes_i, over="neuron")  # result is sorted
        spikes_i = spikes_i.with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index")
        )

        ## Refractoriness
        rec_states = spikes_i.with_columns(
            pl.col("neuron").alias("source"),
            pl.col("neuron").alias("target"),
            pl.col("f_index")
            .gather((pl.int_range(pl.len()) - 1) % pl.len())
            .over("neuron")
            .alias("f_index_source"),
            pl.col("f_index").alias("f_index_target"),
            pl.col("time_prev").alias("f_time_in_target"),
            pl.col("time").alias("f_time_out_target"),
            pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight_0"),
            pl.lit(0.0, pl.Float64).alias("weight_1"),
        ).select(
            "source",
            "target",
            "f_index_source",
            "f_index_target",
            "f_time_in_target",
            "f_time_out_target",
            "weight_0",
            "weight_1",
        )

        ## Synaptic transmission
        # 0. Compute origins for each neuron
        origins_i = spikes_i.group_by("neuron").agg(
            pl.min("time_prev").alias("time_origin")
        )

        # 1. Extract synapses to spiking neurons (the other synapses can be ignored, i.e., have weights = 0.0)
        synapses_i = synapses.join(
            spikes_i, left_on="target", right_on="neuron", how="semi"
        )

        # 2. Create synaptic states
        syn_states = synapses_i.join(
            spikes_i, left_on="source", right_on="neuron"
        ).join(origins_i, left_on="target", right_on="neuron")
        syn_states = syn_states.with_columns(
            pl.col("f_index").alias("f_index_source"),
            pl.lit(None, pl.UInt32).alias("f_index_target"),
            modulo_with_offset(
                pl.col("time") + pl.col("delay"),
                pl.col("period"),
                pl.col("time_origin"),
            ).alias("f_time_in_target"),
            pl.lit(None, pl.Float64).alias("f_time_out_target"),
        )
        syn_states = syn_states.select(
            "source",
            "target",
            "f_index_source",
            "f_index_target",
            "f_time_in_target",
            "f_time_out_target",
            "weight_0",
            "weight_1",
        )

        # Merge and sort (grouped by neuron)
        states = pl.concat([syn_states, rec_states])
        states = states.sort("f_time_in_target")
        states = states.with_columns(
            pl.col("f_index_target").forward_fill().over("target"),
            pl.col("f_time_out_target").forward_fill().over("target"),
        )
        states = states.with_columns(
            (pl.col("f_time_out_target") - pl.col("f_time_in_target")).alias("delta")
        )

        # Aggregate states for spike to spike contributions
        agg_states = states.group_by(["f_index_source", "f_index_target"]).agg(
            (
                (
                    pl.col("weight_1")
                    - pl.col("weight_0")
                    - pl.col("weight_1") * pl.col("delta")
                )
                * (-pl.col("delta")).exp()
            )
            .sum()
            .alias("coef")
        )

        # Build linear propagation operator
        A = ss.csr_array(
            (
                agg_states["coef"].to_numpy(),
                (
                    agg_states["f_index_target"].to_numpy(),
                    agg_states["f_index_source"].to_numpy(),
                ),
            ),
        ).todense()
        A /= np.sum(A, axis=1, keepdims=True)

        Phi = compute_phi_matrix(A, spikes_i.height)

        phis[i] = np.abs(ss.linalg.eigs(Phi, k=k, return_eigenvectors=False))

    return phis
