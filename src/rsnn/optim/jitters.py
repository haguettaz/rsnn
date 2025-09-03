import numpy as np
import polars as pl
import scipy.sparse as ss
from numba import njit

from rsnn import REFRACTORY_RESET
from rsnn.log import setup_logging
from rsnn.optim.utils import init_spikes, modulo_with_offset

# Set up logging
logger = setup_logging(__name__, console_level="DEBUG", file_level="INFO")


@njit
def a_to_Phi(a, n_spikes):
    Phi = np.identity(n_spikes, dtype=np.float64)
    for n in range(n_spikes):
        Phi[n] = a[n] @ Phi
    return Phi


def compute_Phi(spikes, synapses):
    spikes = init_spikes(spikes)

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
    origins = spikes.group_by("neuron").agg(pl.min("time_prev").alias("time_origin"))

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
    agg_states = states.group_by(["f_index_source", "f_index_target"]).agg(
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

    # Build linear propagation operator
    a = ss.csr_array(
        (
            agg_states["coef"].to_numpy(),
            (
                agg_states["f_index_target"].to_numpy(),
                agg_states["f_index_source"].to_numpy(),
            ),
        ),
    ).todense()
    a /= np.sum(a, axis=1, keepdims=True)

    return a_to_Phi(a, spikes.height)


def compute_lm_jitters_eigenvalues(spikes, synapses, k=3):
    # Spikes is a dataframe with columns: index, period, neuron, time
    # Synapses is a dataframe with columns: source, target, delay, weight
    # States is a dataframe with columns: f_index_source, f_index_target, f_time_in_target (=f_time_out_source + delay), f_time_out_target, weight_0, weight_1 (index, period, f_time_out_source, and delay are optional)

    ## Extend spikes with additional information
    eigenvalues = {}
    for (id,), spikes_ in spikes.partition_by("index", as_dict=True).items():
        Phi = compute_Phi(spikes_, synapses)
        eigenvalues[id] = ss.linalg.eigs(
            Phi - 1 / Phi.shape[0], k=k, return_eigenvectors=False
        )

    return eigenvalues


def compute_jitters_eigenvalues(spikes, synapses):
    # Spikes is a dataframe with columns: index, period, neuron, time
    # Synapses is a dataframe with columns: source, target, delay, weight
    # States is a dataframe with columns: f_index_source, f_index_target, f_time_in_target (=f_time_out_source + delay), f_time_out_target, weight_0, weight_1 (index, period, f_time_out_source, and delay are optional)

    ## Extend spikes with additional information
    eigenvalues = {}
    for (id,), spikes_ in spikes.partition_by("index", as_dict=True).items():
        Phi = compute_Phi(spikes_, synapses)
        eigenvalues[id] = np.linalg.eigvals(Phi - 1 / Phi.shape[0])

    return eigenvalues


# def compute_non_global_phi_eigenvals(Phi, k=3):
#     return ss.linalg.eigs(Phi - 1 / Phi.shape[0], k=k, return_eigenvectors=False)


# def compute_global_drift(Phi):
#     return ss.linalg.eigs(Phi, k=1, sigma=1.0)
