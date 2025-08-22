from typing import Callable, Optional, Tuple

import numpy as np
import polars as pl

from .channels import *
from .constants import *
from .log import *
from .states import *

logger = setup_logging(__name__, console_level="DEBUG", file_level="INFO")

STATE_CHUNK_SIZE = 20_000


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


def extract_next_spikes(states):
    """Extract the first next spike per group, e.g., neuron."""
    new_spikes = (
        states.filter(pl.col("f_time").is_not_nan())
        .group_by("neuron")
        .agg(pl.min("f_time").alias("time"))
    )
    return new_spikes


def cleanse_states_recovery(states, neurons):
    """Remove all states before the last spike (reset)."""
    states = states.join(
        neurons.select("neuron", "last_f_time"), on="neuron", how="left"
    )
    states = states.remove(pl.col("start") <= pl.col("last_f_time")).drop("last_f_time")
    return states


def cleanse_states_causal(states, tmin):
    states = states.filter(pl.col("start") >= tmin)
    return states


def create_recovery_states(spikes):
    states = spikes.with_columns(
        pl.col("time").alias("start"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("w0"),
        pl.lit(0.0, pl.Float64).alias("w1"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("c0"),
        pl.lit(0.0, pl.Float64).alias("c1"),
    )
    return states.select("neuron", "start", "w0", "w1", "c0", "c1")


def create_synaptic_states(spikes, synapses):
    states = synapses.join(spikes, left_on="source", right_on="neuron", how="inner")
    states = states.with_columns(
        (pl.col("time") + pl.col("delay")).alias("start"),
        pl.col("target").alias("neuron"),
        pl.col("w0").alias("c0"),
        pl.col("w1").alias("c1"),
    )
    return states.select("neuron", "start", "w0", "w1", "c0", "c1")


def update_f_thresh(states, neurons):
    """Update the firing thresholds in states."""
    # Update the firing thresholds for the spiking neurons
    return states.update(neurons, on="neuron")


def compute_f_time(states):
    """Update rising crossing time in states."""
    f_times = compute_rising_crossing_times(
        states.get_column("f_thresh").to_numpy(),
        states.get_column("start").to_numpy(),
        states.get_column("length").to_numpy(),
        states.get_column("c0").to_numpy(),
        states.get_column("c1").to_numpy(),
    )
    return states.with_columns(f_time=f_times)


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
    spikes = spikes.join(max_times, left_on="neuron", right_on="target", how="left")
    spikes = spikes.remove(pl.col("time") > pl.col("max_time"))
    return spikes.select("neuron", "time")


def filter_states(states, min_delays, spikes):
    max_times = (
        min_delays.join(spikes, left_on="source", right_on="neuron", how="inner")
        .with_columns(
            (pl.col("time") + pl.col("delay")).alias("max_time"),
        )
        .group_by("target")
        .agg(pl.min("max_time"))
    )
    states = states.join(max_times, left_on="neuron", right_on="target", how="left")
    states = states.remove(pl.col("start") > pl.col("max_time"))
    return states.drop("max_time")


def update_new_spikes(states, new_spikes, min_delays):
    states = compute_f_time(states)
    new_spikes = filter_new_spikes(
        new_spikes.extend(
            states.group_by("neuron").agg(pl.col("f_time").min().alias("time"))
        ),
        min_delays,
    )
    return new_spikes


# def create_states(
#     neuron,
#     start,
#     length=None,
#     w0=0.0,
#     w1=0.0,
#     c0=None,
#     c1=None,
#     f_thresh=None,
#     f_time=None,
# ):
#     data = {
#         "neuron": neuron,
#         "start": start,
#         "length": length,
#         "w0": w0,
#         "w1": w1,
#         "c0": c0,
#         "c1": c1,
#         "f_thresh": f_thresh,
#         "f_time": f_time,
#     }
#     schema = {
#         "neuron": pl.UInt32,
#         "start": pl.Float64,
#         "length": pl.Float64,
#         "w0": pl.Float64,
#         "w1": pl.Float64,
#         "c0": pl.Float64,
#         "c1": pl.Float64,
#         "f_thresh": pl.Float64,
#         "f_time": pl.Float64,
#     }
#     return pl.DataFrame(data, schema)


def create_initial_states(neurons, spikes, synapses):

    # neurons is a dataframe with columns: index, f_thresh, last_f_time (potentially null)
    # spikes is a dataframe with columns: neuron, time
    # synapses is a dataframe with columns: source, target, delay, w0, w1

    # 1. Synaptic states
    syn_states = create_synaptic_states(spikes, synapses)
    # syn_states = synapses.join(spikes, left_on="source", right_on="neuron")
    # syn_states = syn_states.with_columns(
    #     pl.col("target").alias("neuron"),
    #     (pl.col("time") + pl.col("delay")).alias("start"),
    # )
    # syn_states = syn_states.select(
    #     pl.col("neuron"),
    #     pl.col("start"),
    #     pl.col("w0"),
    #     pl.col("w1"),
    # )
    syn_states = cleanse_states_recovery(syn_states, neurons)

    # 2. Refractory states
    last_spikes = neurons.select("neuron", "last_f_time").rename(
        {"last_f_time": "time"}
    )
    rec_states = create_recovery_states(last_spikes)

    # 3. Merge all states
    states = pl.concat([syn_states, rec_states])

    # 4. Update states information
    states = states.sort("start")
    states = states.with_columns(pl.lit(None, pl.Float64).alias("f_thresh"))
    states = update_f_thresh(states, neurons)
    states = update_length(states, over="neuron", fill_value=float("inf"))
    states = update_coef(states, over="neuron")

    return states


def simulate(
    neurons, spikes, synapses, min_delays, states, start, end, f_thresh_noise=None
):
    """
    Run the simulation from tmin to tmax.

    neurons has columns: index, f_thresh, last_f_time
    spikes has columns: neuron, time
    states has columns: neuron, start, w0, w1, length, c0, c1, f_thresh, f_time
    synapses has columns: source, target, delay, w0, w1

    """
    if f_thresh_noise is None:
        f_thresh_noise = lambda _: FIRING_THRESHOLD

    # Main simulation loop
    time = start
    while time < end:
        # # Get the current spikes
        # states = update_f_time(
        #     states
        # )  # Slowest operation | Problem: Many computed f_time are useless, we only need the smallest f_time (per neuron). Solution: Compute per chunk instead??? Possibility to combine with filtering through min_delays?

        # # Accept as many spikes as possible, based on the minimum propagation delay
        # new_spikes = extract_next_spikes(states)
        # new_spikes = filter_new_spikes(new_spikes, min_delays)

        new_spikes = pl.DataFrame(schema={"neuron": pl.UInt32, "time": pl.Float64})
        for states_chunk in states.iter_slices(STATE_CHUNK_SIZE):
            # 1. Filter chunk with new_spikes and min_delays
            states_chunk = filter_states(states_chunk, min_delays, new_spikes)
            if states_chunk.height == 0:
                break

            # 2. Update new_spikes
            new_spikes = update_new_spikes(states_chunk, new_spikes, min_delays)

        # Update the simulation time
        time = new_spikes["time"].min() or end
        logger.debug(f"Simulation time: {time}")

        # Add new spikes and update firing threshold of spiking neurons
        spikes = spikes.vstack(new_spikes)
        neurons = neurons.update(
            new_spikes.rename({"time": "last_f_time"}).with_columns(
                f_thresh=f_thresh_noise(new_spikes.height),
            ),
            on="neuron",
        )

        # Synaptic states from new spikes
        syn_states = create_synaptic_states(new_spikes, synapses)
        states = states.extend(
            syn_states.match_to_schema(states.schema, missing_columns="insert")
        )

        # Recovery states from new spikes
        states = cleanse_states_recovery(states, neurons)  # reset
        rec_states = create_recovery_states(new_spikes)
        states = states.extend(
            rec_states.match_to_schema(states.schema, missing_columns="insert")
        )

        # Sort states and sort coefficients
        states = states.sort("neuron", "start")
        states = update_f_thresh(states, neurons)
        states = update_length(states, over="neuron", fill_value=float("inf"))
        states = update_coef(states, over="neuron")  # second slowest operation

        # Causal cleansing: filter out states that are useless for simulation after current time
        states = cleanse_states_causal(states, time)

    return spikes, states
