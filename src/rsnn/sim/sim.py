from typing import Callable, Optional, Tuple

import numpy as np
import polars as pl
from scipy.sparse import csr_array
from scipy.sparse.csgraph import floyd_warshall

import rsnn_plugin as rp
from rsnn import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.log import setup_logging

# from .channels import *
# from .states import *

logger = setup_logging(__name__, console_level="INFO", file_level="DEBUG")


def filter_new_spikes(new_spikes, min_delays):
    """Filter new spikes based on minimum propagation delays. All causally independent spikes are kept."""
    max_times = (
        min_delays.join(new_spikes, left_on="source", right_on="neuron", how="inner")
        .with_columns(
            (pl.col("time") + pl.col("delay")).alias("max_time"),
        )
        .group_by("target")
        .agg(pl.min("max_time"))
    )
    new_spikes = new_spikes.join(
        max_times, left_on="neuron", right_on="target", how="left"
    )
    new_spikes = new_spikes.remove(pl.col("time") > pl.col("max_time"))
    return new_spikes.select("neuron", "time")


# def extract_next_spikes(states):
#     """Extract the first next spike per group, e.g., neuron."""
#     new_spikes = (
#         states.filter(pl.col("f_time").is_not_nan())
#         .group_by("neuron")
#         .agg(pl.min("f_time").alias("time"))
#     )
#     return new_spikes


# def cleanse_states_recovery(states, neurons):
#     """Remove all states strictily before the last spike (reset)."""
#     # Remove all states STRICTLY before the last_f_time (otherwise, problem with recovery mechanism)
#     return (
#         states.join(neurons.select("neuron", "last_f_time"), on="neuron", how="left")
#         .filter(pl.col("start") >= pl.col("last_f_time"))
#         .drop("last_f_time")
#     )


# def cleanse_states_causal(states, tmin):
#     states = states.filter(pl.col("start") >= tmin)
#     return states


# def create_recovery_states(spikes):
#     return spikes.with_columns(
#         pl.col("time").alias("start"),
#         pl.lit(REFRACTORY_RESET, pl.Float64).alias("in_coef_0"),
#         pl.lit(0.0, pl.Float64).alias("in_coef_1"),
#     )


# def create_synaptic_states(spikes, synapses):
#     return synapses.join(
#         spikes, left_on="source", right_on="neuron", how="inner"
#     ).select(
#         (pl.col("time") + pl.col("delay")).alias("start"),
#         pl.col("target").alias("neuron"),
#         pl.lit(0.0, pl.Float64).alias("in_coef_0"),
#         pl.col("weight").alias("in_coef_1"),
#     )


def filter_spikes(spikes, min_delays):
    """Filter new spikes based on minimum propagation delays. All causally independent spikes are kept."""
    max_times = (
        min_delays.join(spikes, left_on="source", right_on="neuron", how="inner")
        .with_columns(
            (pl.col("time") + pl.col("delay")).alias("max_time"),
        )
        .group_by("target")
        .agg(pl.min("max_time"))
    )
    return (
        spikes.join(max_times, left_on="neuron", right_on="target", how="left")
        .filter(pl.col("time") <= pl.col("max_time"))
        .select("neuron", "time")
    )


def filter_states(states, min_delays, spikes):
    max_times = (
        min_delays.join(spikes, left_on="source", right_on="neuron", how="inner")
        .with_columns(
            (pl.col("time") + pl.col("delay")).alias("max_time"),
        )
        .group_by("target")
        .agg(pl.min("max_time"))
    )
    return (
        states.join(max_times, left_on="neuron", right_on="target", how="left")
        .filter(pl.col("start") <= pl.col("max_time"))
        .drop("max_time")
    )


# def update_new_spikes(states, new_spikes, min_delays):
#     states = compute_f_time(states)
#     new_spikes = filter_new_spikes(
#         new_spikes.extend(
#             states.group_by("neuron").agg(pl.col("f_time").min().alias("time"))
#         ),
#         min_delays,
#     )
#     return new_spikes


# def create_states(
#     neuron,
#     start,
#     length=None,
#     weight_0=0.0,
#     weight_1=0.0,
#     coef_0=None,
#     coef_1=None,
#     f_thresh=None,
#     f_time=None,
# ):
#     data = {
#         "neuron": neuron,
#         "start": start,
#         "length": length,
#         "weight_0": weight_0,
#         "weight_1": weight_1,
#         "coef_0": coef_0,
#         "coef_1": coef_1,
#         "f_thresh": f_thresh,
#         "f_time": f_time,
#     }
#     schema = {
#         "neuron": pl.UInt32,
#         "start": pl.Float64,
#         "length": pl.Float64,
#         "weight_0": pl.Float64,
#         "weight_1": pl.Float64,
#         "coef_0": pl.Float64,
#         "coef_1": pl.Float64,
#         "f_thresh": pl.Float64,
#         "f_time": pl.Float64,
#     }
#     return pl.DataFrame(data, schema)


def init_states(spikes, synapses):
    """Warning: spikes must be sorted by time over the neurons"""
    # Compute last spikes per neuron
    last_spikes = spikes.group_by("neuron").agg(pl.last("time"))

    # Synaptic states
    syn_states = synapses.join(spikes, left_on="source", right_on="neuron").select(
        pl.col("target").alias("neuron"),
        (pl.col("time") + pl.col("delay")).alias("start"),
        pl.lit(0.0, pl.Float64).alias("in_coef_0"),
        pl.col("weight").alias("in_coef_1"),
    )

    # Refractory states
    rec_states = last_spikes.select(
        pl.col("neuron"),
        pl.col("time").alias("start"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    )

    return (
        syn_states.extend(rec_states)
        .join(last_spikes, on="neuron", how="left")
        .remove(pl.col("start") < pl.col("time"))
        .drop("time")
    )


def init_min_delays(synapses, n_neurons):
    """Compute the fastest synapses between nodes in a directed graph."""

    # Initialize fast synapses by grouping and aggregating the minimum delay
    min_delays = synapses.group_by(["source", "target"]).agg(
        pl.min("delay").alias("delay")
    )

    # Convert to a sparse matrix for Floyd-Warshall algorithm
    row_ind = min_delays.get_column("source").to_numpy()
    col_ind = min_delays.get_column("target").to_numpy()
    data = min_delays.get_column("delay").to_numpy()
    graph = csr_array((data, (row_ind, col_ind)), shape=(n_neurons, n_neurons))
    graph = floyd_warshall(graph, directed=True, overwrite=True)

    # Convert back to DataFrame
    min_delays = pl.DataFrame(
        {
            "source": np.repeat(np.arange(n_neurons), n_neurons),
            "target": np.tile(np.arange(n_neurons), n_neurons),
            "delay": graph.flatten(),
        }
    )
    min_delays = min_delays.filter(pl.col("delay").is_finite())

    return min_delays


def run(neurons, spikes, synapses, start, end, std_threshold=0.0, rng=None):
    """
    Run the simulation from tmin to tmax.

    neurons has columns: neuron, f_thresh
    spikes has columns: neuron, time
    synapses has columns: source, target, delay, weight

    states has columns: neuron, start, in_coef_0, in_coef_1
    """
    if rng is None:
        rng = np.random.default_rng()

    spikes = spikes.sort("time")
    states = init_states(spikes, synapses)
    logger.info("States initialized.")

    min_delays = init_min_delays(synapses, neurons.select(pl.max("neuron")).item() + 1)
    logger.info("Minimum delays initialized.")

    # Main simulation loop
    logger.info(f"Simulation from {start} to {end} in progress...")
    time = start
    while time < end:
        # Sort states according to their starting time
        states = states.sort("start")

        # New spikes
        new_spikes = (
            states.join(neurons, on="neuron")
            .group_by("neuron")
            .agg(
                time=rp.first_ftime(
                    pl.col("start"),
                    pl.col("start").diff().shift(-1),  # length
                    pl.col("start").diff(),  # prev_length
                    pl.col("in_coef_0"),
                    pl.col("in_coef_1"),
                    pl.col("f_thresh"),
                )
            )
            .drop_nulls()
        )
        new_spikes = filter_new_spikes(new_spikes, min_delays)

        # Simulation time
        time = new_spikes.select("time").min().item() or end
        logger.debug(f"Simulation time: {time}")

        # Append new spikes
        spikes = spikes.vstack(new_spikes)

        # Firing threshold
        neurons = neurons.update(
            new_spikes.select(
                pl.col("neuron"),
                pl.lit(
                    rng.normal(FIRING_THRESHOLD, std_threshold, size=new_spikes.height)
                ).alias("f_thresh"),
            ),
            on="neuron",
        )

        # Recovery states
        rec_states = new_spikes.select(
            pl.col("neuron"),
            pl.col("time").alias("start"),
            pl.lit(REFRACTORY_RESET, pl.Float64).alias("in_coef_0"),
            pl.lit(0.0, pl.Float64).alias("in_coef_1"),
        )

        # Synaptic states
        syn_states = synapses.join(
            new_spikes, left_on="source", right_on="neuron"
        ).select(
            pl.col("target").alias("neuron"),
            (pl.col("time") + pl.col("delay")).alias("start"),
            pl.lit(0.0, pl.Float64).alias("in_coef_0"),
            pl.col("weight").alias("in_coef_1"),
        )

        # Merge and cleanse states
        states = (
            states.extend(rec_states)
            .extend(syn_states)
            .join(new_spikes, on="neuron", how="left")
            .remove(pl.col("start") < pl.col("time"))
            .drop("time")
        )
        # states = states.extend(
        #     rec_states.select(
        #         "neuron", "start", "weight_0", "weight_1"
        #     ).match_to_schema(states.schema, missing_columns="insert"),
        # ).extend(
        #     syn_states.select(
        #         "neuron", "start", "weight_0", "weight_1"
        #     ).match_to_schema(states.schema, missing_columns="insert"),
        # )

        # states = cleanse_states_recovery(states, neurons)  # reset

    return neurons, spikes, states
