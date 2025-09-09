import numpy as np
import polars as pl
import scipy.sparse as ss

from rsnn import REFRACTORY_RESET
from rsnn.log import setup_logging
from rsnn.optim.utils import modulo_with_offset

# Set up logging
logger = setup_logging(__name__, console_level="DEBUG", file_level="INFO")


def compute_Phi(synapses, spikes):
    # Init spikes with f_index and time_prev
    spikes = spikes.sort("time").with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
        modulo_with_offset(
            pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
            pl.col("period"),
            pl.col("time") - pl.col("period"),
        )
        .over("neuron")
        .alias("time_prev"),
    )

    ## Refractoriness
    rec_states = spikes.select(
        pl.col("neuron").alias("source"),
        pl.col("neuron").alias("target"),
        pl.col("f_index")
        .gather((pl.int_range(pl.len()) - 1) % pl.len())
        .over("neuron")
        .alias("f_index_source"),
        pl.col("f_index").alias("f_index_target"),
        pl.col("time_prev").alias("f_time_in_target"),
        pl.col("time").alias("f_time_out_target"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    )

    ## Synaptic transmission
    # Compute origins for each neuron
    origins = spikes.group_by("neuron").agg(pl.first("time_prev").alias("time_origin"))

    # Create synaptic states
    syn_states = (
        synapses.join(spikes, left_on="target", right_on="neuron", how="semi")
        .join(spikes, left_on="source", right_on="neuron")
        .join(origins, left_on="target", right_on="neuron")
    ).select(
        pl.col("source"),
        pl.col("target"),
        pl.col("f_index").alias("f_index_source"),
        pl.lit(None, pl.UInt32).alias("f_index_target"),
        modulo_with_offset(
            pl.col("time") + pl.col("delay"),
            pl.col("period"),
            pl.col("time_origin"),
        ).alias("f_time_in_target"),
        pl.lit(None, pl.Float64).alias("f_time_out_target"),
        pl.lit(0.0, pl.Float64).alias("in_coef_0"),
        pl.col("weight").alias("in_coef_1"),
    )

    # Merge and sort (grouped by neuron)
    states = syn_states.extend(rec_states).sort("f_time_in_target")
    states = states.with_columns(
        pl.col("f_index_target").forward_fill().over("target"),
        # pl.col("f_time_out_target").forward_fill().over("target"),
        (pl.col("f_time_out_target").forward_fill() - pl.col("f_time_in_target"))
        .over("target")
        .alias("time_source_to_target"),
    )

    # Aggregate states for spike to spike contributions
    agg_states = (
        states.group_by(["f_index_source", "f_index_target"])
        .agg(
            (
                (
                    pl.col("in_coef_1")
                    - pl.col("in_coef_0")
                    - pl.col("in_coef_1") * pl.col("time_source_to_target")
                )
                * (-pl.col("time_source_to_target")).exp()
            )
            .sum()
            .alias("coef")
        )
        .sort("f_index_target")
    )

    Phi = np.identity(spikes.height, dtype=np.float64)
    a = np.empty(spikes.height, dtype=np.float64)
    for n, agg_states_ in enumerate(agg_states.partition_by("f_index_target")):
        a.fill(0.0)
        a[agg_states_["f_index_source"]] = agg_states_["coef"]
        a /= np.sum(a)
        Phi[n] = a @ Phi

    return Phi


def compute_lm_jitters_eigenvalues(synapses, spikes, k=3):
    # Spikes is a dataframe with columns: index, period, neuron, time
    # Synapses is a dataframe with columns: source, target, delay, weight
    # States is a dataframe with columns: f_index_source, f_index_target, f_time_in_target (=f_time_out_source + delay), f_time_out_target, weight_0, weight_1 (index, period, f_time_out_source, and delay are optional)

    ## Extend spikes with additional information

    eigenvalues = {}
    for (i,), spikes_i in spikes.partition_by("index", as_dict=True).items():
        Phi = compute_Phi(synapses, spikes_i)
        eigenvalues[i] = ss.linalg.eigs(
            Phi - 1 / Phi.shape[0], k=k, return_eigenvectors=False
        )

    return eigenvalues


def compute_jitters_eigenvalues(synapses, spikes):
    # Spikes is a dataframe with columns: index, period, neuron, time
    # Synapses is a dataframe with columns: source, target, delay, weight
    # States is a dataframe with columns: f_index_source, f_index_target, f_time_in_target (=f_time_out_source + delay), f_time_out_target, weight_0, weight_1 (index, period, f_time_out_source, and delay are optional)

    eigenvalues = {}
    for (i,), spikes_i in spikes.partition_by("index", as_dict=True).items():
        Phi = compute_Phi(synapses, spikes_i)
        eigenvalues[i] = np.linalg.eigvals(Phi - 1 / Phi.shape[0])

    return eigenvalues
