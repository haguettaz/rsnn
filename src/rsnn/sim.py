from typing import Callable, Optional, Tuple

import numpy as np
import polars as pl

import rsnn_plugin as rp

from .channels import *
from .constants import *
from .log import *
from .states import *

logger = setup_logging(__name__, console_level="DEBUG", file_level="INFO")


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


def cleanse_states_recovery(states, neurons):
    """Remove all states before the last spike (reset)."""
    states = states.join(
        neurons.select("neuron", "last_f_time"), on="neuron", how="left"
    )
    # Remove all states STRICTLY before the last_f_time (otherwise, problem with recovery mechanism)
    states = states.remove(pl.col("start") < pl.col("last_f_time")).drop("last_f_time")
    return states


# def cleanse_states_causal(states, tmin):
#     states = states.filter(pl.col("start") >= tmin)
#     return states


def create_recovery_states(spikes):
    states = spikes.with_columns(
        pl.col("time").alias("start"),
        pl.lit(REFRACTORY_RESET, pl.Float64).alias("weight_0"),
        pl.lit(0.0, pl.Float64).alias("weight_1"),
        # pl.lit(REFRACTORY_RESET, pl.Float64).alias("coef_0"),
        # pl.lit(0.0, pl.Float64).alias("coef_1"),
    )
    return states.select(
        "neuron", "start", "weight_0", "weight_1"
    )  # , "coef_0", "coef_1")


def create_synaptic_states(spikes, synapses):
    states = synapses.join(spikes, left_on="source", right_on="neuron", how="inner")
    states = states.with_columns(
        (pl.col("time") + pl.col("delay")).alias("start"),
        pl.col("target").alias("neuron"),
        # pl.col("weight_0").alias("coef_0"),
        # pl.col("weight_1").alias("coef_1"),
    )
    return states.select(
        "neuron", "start", "weight_0", "weight_1"
    )  # , "coef_0", "coef_1")


# def update_f_thresh(states, neurons):
#     """Update the firing thresholds in states."""
#     # Update the firing thresholds for the spiking neurons
#     return states.update(neurons, on="neuron")


# def compute_f_time(states):
#     """Update rising crossing time in states."""
#     f_times = compute_rising_crossing_times(
#         states.get_column("f_thresh").to_numpy(),
#         states.get_column("start").to_numpy(),
#         states.get_column("length").to_numpy(),
#         states.get_column("coef_0").to_numpy(),
#         states.get_column("coef_1").to_numpy(),
#     )
#     return states.with_columns(f_time=f_times)


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


def create_initial_states(neurons, spikes, synapses):

    # neurons is a dataframe with columns: index, f_thresh, last_f_time (potentially null)
    # spikes is a dataframe with columns: neuron, time
    # synapses is a dataframe with columns: source, target, delay, weight_0, weight_1

    # 1. Synaptic states
    syn_states = create_synaptic_states(spikes, synapses)

    syn_states = create_synaptic_states(spikes, synapses)
    syn_states = syn_states.join(neurons.select("neuron", "last_f_time"), on="neuron")
    # syn_states = syn_states.with_columns(
    #     start=modulo_with_offset(
    #         pl.col("start"), pl.col("period"), pl.col("last_f_time")
    #     )
    # )
    # syn_states = syn_states.filter(pl.col("start") > pl.col("last_f_time"))

    # syn_states = synapses.join(spikes, left_on="source", right_on="neuron")
    # syn_states = syn_states.with_columns(
    #     pl.col("target").alias("neuron"),
    #     (pl.col("time") + pl.col("delay")).alias("start"),
    # )
    # syn_states = syn_states.select(
    #     pl.col("neuron"),
    #     pl.col("start"),
    #     pl.col("weight_0"),
    #     pl.col("weight_1"),
    # )
    syn_states = cleanse_states_recovery(syn_states, neurons)

    # 2. Refractory states
    last_spikes = neurons.select("neuron", "last_f_time").rename(
        {"last_f_time": "time"}
    )
    rec_states = create_recovery_states(last_spikes)

    # 3. Merge all states
    states = pl.concat(
        [
            syn_states.select(
                "neuron", "start", "weight_0", "weight_1"  # , "coef_0", "coef_1"
            ),
            rec_states.select(
                "neuron", "start", "weight_0", "weight_1"  # , "coef_0", "coef_1"
            ),
        ]
    )

    # 4. Add firing threshold column
    states = states.join(neurons.select("neuron", "f_thresh"), on="neuron")

    # 5. Update states information
    states = states.sort("start")
    # states = states.with_columns(
    #     length=pl.col("start").diff().shift(-1, fill_value=float("inf")).over("neuron")
    # )
    # states = states.with_columns(
    #     coef_1=rp.scan_coef_1(pl.col("start").diff(), pl.col("weight_1")).over("neuron")
    # )
    # states = states.with_columns(
    #     coef_0=rp.scan_coef_0(
    #         pl.col("start").diff(),  # prev delta
    #         pl.col("coef_1").shift(),  # prev c1
    #         pl.col("weight_0"),
    #     ).over("neuron")
    # )
    return states


def simulate(
    neurons, spikes, synapses, min_delays, states, start, end, f_thresh_noise=None
):
    """
    Run the simulation from tmin to tmax.

    neurons has columns: index, f_thresh, last_f_time
    spikes has columns: neuron, time
    states has columns: neuron, start, weight_0, weight_1, length, coef_0, coef_1, f_thresh, f_time
    synapses has columns: source, target, delay, weight_0, weight_1

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

        # new_spikes = pl.DataFrame(schema={"neuron": pl.UInt32, "time": pl.Float64})
        # for states_chunk in states.iter_slices(STATE_CHUNK_SIZE):
        #     # 1. Filter chunk with new_spikes and min_delays
        #     states_chunk = filter_states(states_chunk, min_delays, new_spikes)
        #     if states_chunk.height == 0:
        #         break

        #     # 2. Update new_spikes
        #     new_spikes = update_new_spikes(states_chunk, new_spikes, min_delays)

        #
        # Sort states and update firing thresholds
        states = states.sort("start")
        states = states.update(neurons.select("neuron", "f_thresh"), on="neuron")

        # Compute new spikes based on current states
        new_spikes = (
            states.group_by("neuron")
            .agg(
                time=rp.first_ftime(
                    pl.col("start"),
                    pl.col("start").diff().shift(-1),  # length
                    pl.col("start").diff(),  # prev length
                    pl.col("weight_0"),
                    pl.col("weight_1"),
                    pl.col("f_thresh"),
                )
            )
            .drop_nulls()
        )
        new_spikes = filter_new_spikes(new_spikes, min_delays)

        # Update the simulation time
        time = new_spikes.select("time").min().item() or end
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
        # logger.debug(f"New spikes:\n{new_spikes}")
        syn_states = create_synaptic_states(new_spikes, synapses)
        states = states.extend(
            syn_states.select(
                "neuron", "start", "weight_0", "weight_1"  # , "coef_0", "coef_1"
            ).match_to_schema(states.schema, missing_columns="insert")
        )

        # Recovery states from new spikes
        states = cleanse_states_recovery(states, neurons)  # reset
        rec_states = create_recovery_states(new_spikes)
        states = states.extend(
            rec_states.select(
                "neuron", "start", "weight_0", "weight_1"  # , "coef_0", "coef_1"
            ).match_to_schema(states.schema, missing_columns="insert")
        )
        states = states.with_columns(pl.col("f_thresh").forward_fill().over("neuron"))

        # # Sort states and sort coefficients
        # states = states.sort("neuron", "start")
        # states = update_f_thresh(states, neurons)
        # states = update_length(states, over="neuron", fill_value=float("inf"))
        # states = update_coef(states, over="neuron")  # second slowest operation

        # # Causal cleansing: filter out states that are useless for simulation after current time
        # states = cleanse_states_causal(states, time)

    return spikes, states
