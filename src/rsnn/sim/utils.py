import numpy as np
import polars as pl
from scipy.sparse import csr_array
from scipy.sparse.csgraph import floyd_warshall

from rsnn import REFRACTORY_RESET

# from rsnn.utils import scan_states

# def compute_states(
#     neurons: pl.DataFrame, out_spikes: pl.DataFrame, in_spikes: pl.DataFrame
# ) -> pl.DataFrame:
#     """Compute all neuronal states for analysis.

#     Calculates the complete set of neuronal states including synaptic transmission events, refractory periods, and firing events. Combines all state types and performs temporal scanning to compute membrane potential coefficients.

#     Args:
#         neurons (pl.DataFrame): Neuron parameters with columns 'neuron', 'reset'.
#         out_spikes (pl.DataFrame): Output spike data with columns 'neuron', 'f_index', 'time', 'time_prev'. Must be sorted by time within each neuron group.
#         in_spikes (pl.DataFrame): Input spike data with columns 'neuron', 'time', 'weight'. Must be sorted by time within each neuron group.

#     Returns:
#         pl.DataFrame: Complete state representation with temporal dynamics with columns including 'neuron', 'f_index', 'start', 'length', 'coef_0', and 'coef_1'.

#     Notes:
#         The states are sorted by time with each firing index group and integrates multiple state types:
#         - Refractory states from previous spikes
#         - Synaptic transmission states from incoming connections
#         - Firing states at exact spike times
#     """

#     # Refractoriness
#     rec_states = out_spikes.join(neurons, on="neuron").select(
#         pl.col("neuron"),
#         pl.col("f_index"),
#         pl.col("time_prev").alias("start"),
#         pl.col("reset").alias("in_coef_0"),
#         pl.lit(0.0, pl.Float64).alias("in_coef_1"),
#     )

#     # Synaptic transmission
#     syn_states = in_spikes.select(
#         pl.col("neuron"),
#         pl.lit(None, pl.UInt32).alias("f_index"),
#         pl.col("time").alias("start"),
#         pl.lit(0.0, pl.Float64).alias("in_coef_0"),
#         pl.col("weight").alias("in_coef_1"),
#     )

#     in_states = (
#         syn_states.extend(rec_states)
#         .sort("neuron", "start")
#         .select(
#             pl.col("neuron"),
#             pl.col("f_index").forward_fill().over("neuron"),
#             pl.col("start"),
#             pl.col("in_coef_0"),
#             pl.col("in_coef_1"),
#         )
#         .drop_nulls("f_index")
#     )

#     f_states = out_spikes.select(
#         pl.col("neuron"),
#         pl.col("f_index"),
#         pl.col("time").alias("start"),
#         pl.lit(None, pl.Float64).alias("in_coef_0"),
#         pl.lit(None, pl.Float64).alias("in_coef_1"),
#     )

#     states = in_states.extend(f_states).sort("f_index", "start")

#     return scan_states(states)


def init_syn_states(spikes: pl.DataFrame, synapses: pl.DataFrame) -> pl.DataFrame:
    """Initialize synaptic states from spikes and synaptic connections.

    Creates initial synaptic states representation for the simulation from existing spikes. Filters out states that start before the last spike of each neuron.

    Args:
        spikes (pl.DataFrame): Spike events with columns 'neuron', 'time'. Must be sorted by time within each neuron group.
        synapses (pl.DataFrame): Synaptic connections with columns 'source', 'target', 'delay', 'weight', 'index'.

    Returns:
        pl.DataFrame: Initial synaptic states with columns 'neuron', 'index', 'start'.

    Notes:
        Combines synaptic states (from incoming connections) and refractory
        states (from last spikes) to create the initial condition for simulation.
    """
    # Compute last spikes per neuron
    last_spikes = spikes.group_by("neuron").agg(pl.max("time"))

    # Synaptic states
    syn_states = synapses.join(spikes, left_on="source", right_on="neuron").select(
        pl.col("target").alias("neuron"),
        pl.col("in_index"),
        (pl.col("time") + pl.col("delay")).alias("start"),
    )

    return (
        syn_states.join(last_spikes, on="neuron", how="left")
        .remove(pl.col("start") < pl.col("time"))
        .drop("time")
    )


def init_states(spikes: pl.DataFrame, synapses: pl.DataFrame) -> pl.DataFrame:
    """Initialize neuronal states from spikes and synaptic connections.

    Creates initial state representation for the simulation by computing synaptic and refractory states from existing spikes. Filters out states that start before the last spike of each neuron.

    Args:
        neurons (pl.DataFrame): Neuron parameters with columns 'neuron'.
        spikes (pl.DataFrame): Spike events with columns 'neuron', 'time'. Must be sorted by time within each neuron group.
        synapses (pl.DataFrame): Synaptic connections with columns 'source', 'target', 'delay', 'weight'.

    Returns:
        pl.DataFrame: Initial states with columns 'neuron', 'start',
            'in_coef_0', 'in_coef_1'.

    Warning:
        Spikes must be sorted by time over the neurons for correct operation.

    Notes:
        Combines synaptic states (from incoming connections) and refractory
        states (from last spikes) to create the initial condition for simulation.
    """
    # Compute last spikes per neuron
    last_spikes = spikes.group_by("neuron").agg(pl.max("time"))

    # Refractory states
    rec_states = last_spikes.select(
        pl.col("neuron"),
        pl.col("time").alias("start"),
        pl.lit(REFRACTORY_RESET).alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    )

    # Synaptic states
    syn_states = (
        synapses.join(spikes, left_on="source", right_on="neuron")
        .select(
            pl.col("target").alias("neuron"),
            (pl.col("time") + pl.col("delay")).alias("start"),
            pl.lit(0.0, pl.Float64).alias("in_coef_0"),
            pl.col("weight").alias("in_coef_1"),
        )
        .join(last_spikes, on="neuron", how="left")
        .remove(pl.col("start") < pl.col("time"))
        .drop("time")
    )

    states = rec_states.extend(syn_states).sort("start", "neuron", maintain_order=True)

    return states


def init_min_delays(synapses):
    """Compute shortest path delays between all pairs of neurons.

    Calculates the minimum propagation delays between all neuron pairs
    using the Floyd-Warshall algorithm on the synaptic connectivity graph.
    Used for enforcing causal constraints during simulation.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns 'source',
            'target', 'delay'.

    Returns:
        pl.DataFrame: Minimum delays between all connected neuron pairs
            with columns 'source', 'target', 'delay'.

    Notes:
        Uses sparse matrix representation and Floyd-Warshall algorithm for
        efficient all-pairs shortest path computation. Only finite delays
        (reachable pairs) are returned in the result.
    """
    # Initialize fast synapses by grouping and aggregating the minimum delay
    min_delays = synapses.group_by(["source", "target"]).agg(
        pl.min("delay").alias("delay")
    )
    n_neurons = synapses.select(pl.max_horizontal("source", "target")).max().item() + 1

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
