import numpy as np
import polars as pl

import rsnn_plugin as rp
from rsnn import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.log import setup_logging

# from rsnn.sim.update import update_weights
from rsnn.sim.utils import init_min_delays, init_states, init_syn_states


def filter_new_spikes(
    new_spikes: pl.DataFrame, min_delays: pl.DataFrame
) -> pl.DataFrame:
    """Filter new spikes based on minimum propagation delays.

    Removes spikes that cannot causally affect the network due to propagation
    delays. Keeps only causally independent spikes that can occur without
    violating temporal causality constraints.

    Args:
        new_spikes (pl.DataFrame): New spike events with columns 'neuron', 'time'.
        min_delays (pl.DataFrame): Minimum propagation delays with columns
            'source', 'target', 'delay'.

    Returns:
        pl.DataFrame: Filtered spikes that satisfy causal constraints.

    Notes:
        All causally independent spikes are kept. A spike is filtered if
        it would violate causality based on minimum propagation delays.
    """
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


def filter_spikes(spikes: pl.DataFrame, min_delays: pl.DataFrame) -> pl.DataFrame:
    """Filter spikes based on minimum propagation delays.

    Removes spikes that violate causal constraints based on minimum propagation
    delays between neurons. Ensures temporal consistency in the spike train.

    Args:
        spikes (pl.DataFrame): Spike events with columns 'neuron', 'time'.
        min_delays (pl.DataFrame): Minimum propagation delays with columns
            'source', 'target', 'delay'.

    Returns:
        pl.DataFrame: Filtered spikes satisfying causal temporal constraints.

    Notes:
        All causally independent spikes are preserved. Only spikes that would
        create temporal inconsistencies are filtered out.
    """
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


def filter_states(
    states: pl.DataFrame, min_delays: pl.DataFrame, spikes: pl.DataFrame
) -> pl.DataFrame:
    """Filter neuronal states based on causal propagation constraints.

    Removes states that start after the maximum causally consistent time
    for each neuron. Ensures state evolution respects temporal causality
    based on minimum propagation delays.

    Args:
        states (pl.DataFrame): Neuronal states with columns 'neuron', 'start'.
        min_delays (pl.DataFrame): Minimum propagation delays with columns
            'source', 'target', 'delay'.
        spikes (pl.DataFrame): Spike events with columns 'neuron', 'time'.

    Returns:
        pl.DataFrame: Filtered states satisfying causal temporal constraints.

    Notes:
        States are filtered if their start time exceeds the maximum time
        at which events can causally affect the target neuron.
    """
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


# def run_wiener_threshold(
#     neurons,
#     spikes,
#     synapses,
#     start,
#     end,
#     refractory_reset=REFRACTORY_RESET,
#     noise_std=0.0,
#     states=None,
#     min_delays=None,
#     rng=None,
#     logger=None,
# ):
#     """Run discrete event simulation of spiking neural network.

#     Executes a discrete event simulation of the spiking neural network from
#     start to end time. Uses event-driven simulation with causal filtering
#     to maintain temporal consistency and efficiency.

#     Args:
#         neurons (pl.DataFrame): Neuron properties with columns 'neuron', 'f_thresh'.
#         spikes (pl.DataFrame): Initial spike events with columns 'neuron', 'time'.
#         synapses (pl.DataFrame): Synaptic connections with columns 'source',
#             'target', 'delay', 'weight'.
#         start (float): Simulation start time.
#         end (float): Simulation end time.
#         std_thresh (float, optional): Standard deviation for threshold
#             noise. Defaults to 0.0.
#         rng (np.random.Generator, optional): Random number generator for
#             threshold noise. Defaults to None.

#     Returns:
#         tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
#             - neurons: Updated neuron states with final thresholds
#             - spikes: Complete spike train including generated spikes
#             - states: Final neuronal states at simulation end

#     Notes:
#         Uses discrete event simulation with:
#         - Causal filtering based on minimum propagation delays
#         - Dynamic threshold updates with optional noise
#         - Event-driven state updates for efficiency
#         - Refractory and synaptic state management
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     if logger is None:
#         logger = setup_logging(
#             __name__,
#             console_level="INFO",
#             file_level="INFO",
#             file_path="run-wiener-threshold.log",
#         )

#     spikes = spikes.sort("time")

#     if "index" not in synapses.columns:
#         synapses = synapses.with_row_index()

#     if states is None:
#         states = init_states(spikes, synapses)
#         logger.info("Neuronal states initialized.")

#     if min_delays is None:
#         min_delays = init_min_delays(synapses)
#         logger.info("Minimum delays initialized.")

#     # Main simulation loop
#     logger.info(f"Simulation from {start} to {end} in progress...")
#     time = start
#     while time < end:
#         # Sort states according to their starting time
#         states = states.sort("start")

#         # New spikes
#         new_spikes = (
#             states.join(neurons, on="neuron")
#             .group_by("neuron")
#             .agg(
#                 time=rp.first_ftime(
#                     pl.col("start"),
#                     pl.col("start").diff().shift(-1),  # length
#                     pl.col("start").diff(),  # prev_length
#                     pl.col("in_coef_0"),
#                     pl.col("in_coef_1"),
#                     pl.col("f_thresh"),
#                 )
#             )
#             .drop_nulls()
#         )
#         new_spikes = filter_new_spikes(new_spikes, min_delays)

#         # Simulation time
#         time = new_spikes.select("time").min().item() or end
#         logger.debug(f"Simulation time: {time}")

#         # Append new spikes
#         spikes = spikes.vstack(new_spikes)

#         # Update firing threshold with Wiener noise
#         tmp = rng.normal(0, 1.0, size=new_spikes.height)
#         neurons = neurons.update(
#             new_spikes.join(neurons, on="neuron", how="left")
#             .with_columns(
#                 pl.lit(tmp).alias("noise"),
#                 (pl.col("time") - pl.col("prev_time")).alias("delta"),
#             )
#             .with_columns(
#                 (
#                     pl.col("f_thresh")
#                     + noise_std * pl.col("noise") * pl.col("delta").sqrt()
#                 ).alias("f_thresh"),
#             )
#             .select(
#                 pl.col("neuron"), pl.col("f_thresh"), pl.col("time").alias("prev_time")
#             ),
#             on="neuron",
#         )

#         # Recovery states
#         rec_states = new_spikes.select(
#             pl.col("neuron"),
#             pl.col("time").alias("start"),
#             pl.lit(refractory_reset, pl.Float64).alias("in_coef_0"),
#             pl.lit(0.0, pl.Float64).alias("in_coef_1"),
#         )

#         # Synaptic states
#         syn_states = synapses.join(
#             new_spikes, left_on="source", right_on="neuron"
#         ).select(
#             pl.col("target").alias("neuron"),
#             (pl.col("time") + pl.col("delay")).alias("start"),
#             pl.lit(0.0, pl.Float64).alias("in_coef_0"),
#             pl.col("weight").alias("in_coef_1"),
#         )

#         # Merge and cleanse states
#         states = (
#             states.extend(rec_states)
#             .extend(syn_states)
#             .join(new_spikes, on="neuron", how="left")
#             .remove(pl.col("start") < pl.col("time"))
#             .drop("time")
#         )

#     logger.info("Simulation completed!")
#     return neurons, spikes, states


def run(
    neurons,
    spikes,
    synapses,
    start,
    end,
    states=None,
    min_delays=None,
    rng=None,
    logger=None,
):
    """Run discrete event simulation of spiking neural network.

    Executes a discrete event simulation of the spiking neural network from
    start to end time. Uses event-driven simulation with causal filtering
    to maintain temporal consistency and efficiency.

    Args:
        neurons (pl.DataFrame): Neuron properties with columns 'neuron', 'threshold'.
        spikes (pl.DataFrame): Initial spike events with columns 'neuron', 'time'.
        synapses (pl.DataFrame): Synaptic connections with columns 'source', 'target', 'delay', 'weight'.
        start (float): Simulation start time.
        end (float): Simulation end time.
        rng (np.random.Generator, optional): Random number generator for
            threshold noise. Defaults to None.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
            - neurons: Updated neuron states with final thresholds
            - spikes: Complete spike train including generated spikes
            - states: Final neuronal states at simulation end

    Notes:
        Uses discrete event simulation with:
        - Causal filtering based on minimum propagation delays
        - Dynamic threshold updates with optional noise
        - Event-driven state updates for efficiency
        - Refractory and synaptic state management
    """
    if rng is None:
        rng = np.random.default_rng()

    if logger is None:
        logger = setup_logging(
            __name__,
            console_level="INFO",
            file_level="INFO",
            file_path="run-white-threshold.log",
        )

    spikes = spikes.sort("time")

    # if "index" not in synapses.columns:
    #     synapses = synapses.with_row_index()

    if states is None:
        states = init_states(
            spikes,
            synapses.join(neurons, left_on="target", right_on="neuron", how="semi"),
        )
        logger.info("Neuronal states initialized.")

    if min_delays is None:
        min_delays = init_min_delays(
            synapses.join(
                neurons, left_on="source", right_on="neuron", how="semi"
            ).join(neurons, left_on="target", right_on="neuron", how="semi")
        )
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
                    pl.col("threshold"),
                )
            )
            .drop_nulls()
        )
        new_spikes = filter_new_spikes(new_spikes, min_delays)
        f_neurons = new_spikes.join(neurons, on="neuron")

        # Simulation time
        time = new_spikes.select("time").min().item() or end
        logger.debug(f"Simulation time: {time}")

        # Append new spikes
        spikes = spikes.vstack(new_spikes)

        # Recovery states
        rec_states = f_neurons.select(
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

    logger.info("Simulation completed!")
    return neurons, spikes, states


def run_white_threshold(
    neurons,
    spikes,
    synapses,
    start,
    end,
    std_thresh=1e-3,
    states=None,
    min_delays=None,
    rng=None,
    logger=None,
):
    """Run discrete event simulation of spiking neural network.

    Executes a discrete event simulation of the spiking neural network from
    start to end time. Uses event-driven simulation with causal filtering
    to maintain temporal consistency and efficiency.

    Args:
        neurons (pl.DataFrame): Neuron properties with columns 'neuron', 'threshold'.
        spikes (pl.DataFrame): Initial spike events with columns 'neuron', 'time'.
        synapses (pl.DataFrame): Synaptic connections with columns 'source',
            'target', 'delay', 'weight'.
        start (float): Simulation start time.
        end (float): Simulation end time.
        rng (np.random.Generator, optional): Random number generator for
            threshold noise. Defaults to None.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
            - neurons: Updated neuron states with final thresholds
            - spikes: Complete spike train including generated spikes
            - states: Final neuronal states at simulation end

    Notes:
        Uses discrete event simulation with:
        - Causal filtering based on minimum propagation delays
        - Dynamic threshold updates with optional noise
        - Event-driven state updates for efficiency
        - Refractory and synaptic state management
    """
    if rng is None:
        rng = np.random.default_rng()

    if logger is None:
        logger = setup_logging(
            __name__,
            console_level="INFO",
            file_level="INFO",
            file_path="run-white-threshold.log",
        )

    spikes = spikes.sort("time")

    # if "index" not in synapses.columns:
    #     synapses = synapses.with_row_index()

    if states is None:
        states = init_states(
            spikes,
            synapses.join(neurons, left_on="target", right_on="neuron", how="semi"),
        )
        logger.info("Neuronal states initialized.")

    if min_delays is None:
        # use only autonomous neurons for the matrix of propagation delays
        min_delays = init_min_delays(
            synapses.join(
                neurons, left_on="source", right_on="neuron", how="semi"
            ).join(neurons, left_on="target", right_on="neuron", how="semi")
        )
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
                    pl.col("threshold"),
                )
            )
            .drop_nulls()
        )
        new_spikes = filter_new_spikes(new_spikes, min_delays)
        f_neurons = new_spikes.join(neurons, on="neuron")

        # Simulation time
        time = new_spikes.select("time").min().item() or end
        logger.debug(f"Simulation time: {time}")

        # Append new spikes
        spikes = spikes.vstack(new_spikes)

        # Update firing threshold with white noise
        neurons = neurons.update(
            f_neurons.with_columns(
                pl.lit(
                    rng.normal(FIRING_THRESHOLD, std_thresh, size=f_neurons.height)
                ).alias("threshold")
            ).select(
                pl.col("neuron"),
                pl.col("threshold"),
            ),
            on="neuron",
        )

        # Recovery states
        rec_states = f_neurons.select(
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

    logger.info("Simulation completed!")
    return neurons, spikes, states


# def run(
#     neurons,
#     spikes,
#     synapses,
#     start,
#     end,
#     states=None,
#     min_delays=None,
#     rng=None,
#     logger=None,
# ):
#     """Run discrete event simulation of spiking neural network without noise.

#     Executes a discrete event simulation of the spiking neural network from
#     start to end time. Uses event-driven simulation with causal filtering
#     to maintain temporal consistency and efficiency.

#     Args:
#         neurons (pl.DataFrame): Neuron properties with columns 'neuron', 'threshold'.
#         spikes (pl.DataFrame): Initial spike events with columns 'neuron', 'time'.
#         synapses (pl.DataFrame): Synaptic connections with columns 'source',
#             'target', 'delay', 'weight'.
#         start (float): Simulation start time.
#         end (float): Simulation end time.
#         rng (np.random.Generator, optional): Random number generator for
#             threshold noise. Defaults to None.

#     Returns:
#         tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
#             - neurons: Updated neuron states with final thresholds
#             - spikes: Complete spike train including generated spikes
#             - states: Final neuronal states at simulation end

#     Notes:
#         Uses discrete event simulation with:
#         - Causal filtering based on minimum propagation delays
#         - Dynamic threshold updates with optional noise
#         - Event-driven state updates for efficiency
#         - Refractory and synaptic state management
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     if logger is None:
#         logger = setup_logging(
#             __name__,
#             console_level="INFO",
#             file_level="INFO",
#             file_path="run-white-threshold.log",
#         )

#     if spikes is None:
#         spikes = pl.DataFrame(
#             schema={"neuron": pl.UInt32, "time": pl.Float64}
#         )  # Empty spikes
#     spikes = spikes.sort("time")

#     if synapses is None:
#         synapses = pl.DataFrame(
#             schema={
#                 "source": pl.UInt32,
#                 "target": pl.UInt32,
#                 "delay": pl.Float64,
#                 "weight": pl.Float64,
#             }
#         )  # Empty synapses

#     if states is None:
#         states = init_states(spikes, synapses)
#         logger.info("Neuronal states initialized.")

#     if min_delays is None:
#         min_delays = init_min_delays(synapses)
#         logger.info("Minimum delays initialized.")

#     # Main simulation loop
#     logger.info(f"Simulation from {start} to {end} in progress...")
#     time = start
#     while time < end:
#         # Sort states according to their starting time
#         states = states.sort("start")

#         # New spikes
#         new_spikes = (
#             states.join(neurons, on="neuron")
#             .group_by("neuron")
#             .agg(
#                 time=rp.first_ftime(
#                     pl.col("start"),
#                     pl.col("start").diff().shift(-1),  # length
#                     pl.col("start").diff(),  # prev_length
#                     pl.col("in_coef_0"),
#                     pl.col("in_coef_1"),
#                     pl.col("threshold"),
#                 )
#             )
#             .drop_nulls()
#         )
#         new_spikes = filter_new_spikes(new_spikes, min_delays)
#         f_neurons = new_spikes.join(neurons, on="neuron")

#         # Simulation time
#         time = new_spikes.select("time").min().item() or end
#         logger.debug(f"Simulation time: {time}")

#         # Append new spikes
#         spikes = spikes.vstack(new_spikes)

#         # Recovery states
#         rec_states = f_neurons.select(
#             pl.col("neuron"),
#             pl.col("time").alias("start"),
#             pl.col("reset").alias("in_coef_0"),
#             pl.lit(0.0, pl.Float64).alias("in_coef_1"),
#         )

#         # Synaptic states
#         syn_states = synapses.join(
#             new_spikes, left_on="source", right_on="neuron"
#         ).select(
#             pl.col("target").alias("neuron"),
#             (pl.col("time") + pl.col("delay")).alias("start"),
#             pl.lit(0.0, pl.Float64).alias("in_coef_0"),
#             pl.col("weight").alias("in_coef_1"),
#         )

#         # Merge and cleanse states
#         states = (
#             states.extend(rec_states)
#             .extend(syn_states)
#             .join(new_spikes, on="neuron", how="left")
#             .remove(pl.col("start") < pl.col("time"))
#             .drop("time")
#         )

#     logger.info("Simulation completed!")
#     return neurons, spikes, states


# def run_adaptive_weights(
#     neurons,
#     spikes,
#     synapses,
#     start,
#     end,
#     alpha=1e-3,
#     l2_reg=0.0,
#     full=False,
#     first_order=True,
#     last_only=True,
#     states=None,
#     syn_states=None,
#     min_delays=None,
#     rng=None,
#     logger=None,
# ):
#     """Run discrete event simulation of spiking neural network.

#     Executes a discrete event simulation of the spiking neural network from
#     start to end time. Uses event-driven simulation with causal filtering
#     to maintain temporal consistency and efficiency.

#     Args:
#         neurons (pl.DataFrame): Neuron properties with columns 'neuron', 'f_thresh'.
#         spikes (pl.DataFrame): Initial spike events with columns 'neuron', 'time'.
#         synapses (pl.DataFrame): Synaptic connections with columns 'source',
#             'target', 'delay', 'weight'.
#         start (float): Simulation start time.
#         end (float): Simulation end time.
#         std_thresh (float, optional): Standard deviation for threshold
#             noise. Defaults to 0.0.
#         rng (np.random.Generator, optional): Random number generator for
#             threshold noise. Defaults to None.

#     Returns:
#         tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing:
#             - neurons: Updated neuron states with final thresholds
#             - spikes: Complete spike train including generated spikes
#             - states: Final neuronal states at simulation end

#     Notes:
#         Uses discrete event simulation with:
#         - Causal filtering based on minimum propagation delays
#         - Dynamic threshold updates with optional noise
#         - Event-driven state updates for efficiency
#         - Refractory and synaptic state management
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     if logger is None:
#         logger = setup_logging(
#             __name__,
#             console_level="INFO",
#             file_level="INFO",
#             file_path="run-white-threshold.log",
#         )

#     spikes = spikes.sort("time")

#     if "index" not in synapses.columns:
#         synapses = synapses.with_row_index()

#     if syn_states is None:
#         syn_states = init_syn_states(spikes, synapses)
#         logger.info("Synaptic states initialized.")

#     if states is None:
#         states = init_states(spikes, synapses)
#         logger.info("Neuronal states initialized.")

#     if min_delays is None:
#         min_delays = init_min_delays(synapses)
#         logger.info("Minimum delays initialized.")

#     # Main simulation loop
#     logger.info(f"Simulation from {start} to {end} in progress...")
#     time = start
#     while time < end:
#         # Sort states according to their starting time
#         syn_states = syn_states.sort("start")
#         states = states.sort("start")

#         # New spikes
#         new_spikes = (
#             states.join(neurons, on="neuron")
#             .group_by("neuron")
#             .agg(
#                 time=rp.first_ftime(
#                     pl.col("start"),
#                     pl.col("start").diff().shift(-1),  # length
#                     pl.col("start").diff(),  # prev_length
#                     pl.col("in_coef_0"),
#                     pl.col("in_coef_1"),
#                     pl.col("f_thresh"),
#                 )
#             )
#             .drop_nulls()
#         )
#         new_spikes = filter_new_spikes(new_spikes, min_delays)

#         # Simulation time
#         time = new_spikes.select("time").min().item() or end
#         logger.debug(f"Simulation time: {time}")

#         # Append new spikes
#         spikes = spikes.vstack(new_spikes)

#         # Update synaptic weights of spiking neurons
#         synapses = update_weights(
#             new_spikes,
#             synapses,
#             syn_states,
#             alpha,
#             l2_reg,
#             full,
#             first_order,
#             last_only,
#         )
#         logger.debug(
#             f"New synaptic weights: {synapses.join(new_spikes, left_on='target', right_on='neuron')}"
#         )

#         # New synaptic states
#         new_syn_states = synapses.join(
#             new_spikes, left_on="source", right_on="neuron"
#         ).select(
#             pl.col("target").alias("neuron"),
#             pl.col("in_index"),
#             (pl.col("time") + pl.col("delay")).alias("start"),
#             pl.col("weight"),
#             # pl.lit(0.0, pl.Float64).alias("in_coef_0"),
#         )

#         # Merge and cleanse synaptic states
#         syn_states = (
#             syn_states.extend(
#                 new_syn_states.select(
#                     pl.col("neuron"),
#                     pl.col("in_index"),
#                     pl.col("start"),
#                     # pl.col("weight"),
#                 )
#             )
#             .join(new_spikes, on="neuron", how="left")
#             .remove(pl.col("start") < pl.col("time"))
#             .drop("time")
#         )

#         # Merge and cleanse states
#         states = (
#             states.extend(
#                 new_spikes.select(
#                     pl.col("neuron"),
#                     pl.col("time").alias("start"),
#                     pl.lit(REFRACTORY_RESET, pl.Float64).alias("in_coef_0"),
#                     pl.lit(0.0, pl.Float64).alias("in_coef_1"),
#                 )  # recovery states
#             )
#             .extend(
#                 new_syn_states.select(
#                     pl.col("neuron"),
#                     pl.col("start"),
#                     pl.lit(0.0, pl.Float64).alias("in_coef_0"),
#                     pl.col("weight").alias("in_coef_1"),
#                 )  # synaptic states
#             )
#             .join(new_spikes, on="neuron", how="left")
#             .remove(pl.col("start") < pl.col("time"))
#             .drop("time")
#         )

#     logger.info("Simulation completed!")
#     return neurons, spikes, synapses, states, syn_states
