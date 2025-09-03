# import numpy as np
# import polars as pl
# from numpy.typing import NDArray
# from scipy.sparse import csr_array
# from scipy.sparse.csgraph import floyd_warshall

# # from .utils import modulo_with_offset

# # def new_synapses(
# #     sources: NDArray[np.intp],
# #     targets: NDArray[np.intp],
# #     delays: NDArray[np.float64] | float = 0.0,
# #     weight: NDArray[np.float64] | float = 0.0,
# #     # in_coef_0: NDArray[np.float64] | float = 0.0,
# #     # in_coef_1: NDArray[np.float64] | float = 0.0,
# # ):
# #     """Note: synapses are first order connections, i.e., alpha kernels"""

# #     return pl.DataFrame(
# #         data={
# #             "source": sources,
# #             "target": targets,
# #             "delay": delays,
# #             "weight": weight,
# #             # "in_coef_0": in_coef_0,
# #             # "in_coef_1": in_coef_1,
# #         },
# #         schema={
# #             "source": pl.UInt32,
# #             "target": pl.UInt32,
# #             "delay": pl.Float64,
# #             "weight": pl.Float64,
# #             # "in_coef_0": pl.Float64,
# #             # "in_coef_1": pl.Float64,
# #         },
# #     )


# # def create_channels(n_neurons, synapses):
# #     synapses = synapses.select(
# #         "source", "target", "delay", "weight", "coef_0", "coef_1"
# #     )
# #     recovery = new_channels(
# #         np.arange(n_neurons),
# #         np.arange(n_neurons),
# #         weight=REFRACTORY_RESET,
# #         coef_0=1.0,
# #     )
# #     return synapses.extend(recovery)


# def compute_min_delays(n_neurons, synapses):
#     """Compute the fastest synapses between nodes in a directed graph."""

#     # Initialize fast synapses by grouping and aggregating the minimum delay
#     min_delays = synapses.group_by(["source", "target"]).agg(
#         pl.min("delay").alias("delay")
#     )

#     # Convert to a sparse matrix for Floyd-Warshall algorithm
#     row_ind = min_delays.get_column("source").to_numpy()
#     col_ind = min_delays.get_column("target").to_numpy()
#     data = min_delays.get_column("delay").to_numpy()
#     graph = csr_array((data, (row_ind, col_ind)), shape=(n_neurons, n_neurons))
#     graph = floyd_warshall(graph, directed=True, overwrite=True)

#     # Convert back to DataFrame
#     min_delays = pl.DataFrame(
#         {
#             "source": np.repeat(np.arange(n_neurons), n_neurons),
#             "target": np.tile(np.arange(n_neurons), n_neurons),
#             "delay": graph.flatten(),
#         }
#     )
#     min_delays = min_delays.filter(pl.col("delay").is_finite())

#     return min_delays


# def out_to_in_spikes(out_spikes, synapses):
#     """For output spike trains with cols: index, neuron, time (and some other optional information)"""
#     in_spikes = synapses.join(out_spikes, left_on="source", right_on="neuron")
#     in_spikes = in_spikes.rename({"target": "neuron"})
#     in_spikes = in_spikes.with_columns(pl.col("time") + pl.col("delay"))
#     return in_spikes


# def extend_over_period(in_spikes, meta):
#     """For input spike trains with periodic (period and origin) information in meta"""
#     in_spikes = in_spikes.join(meta, on=["index", "neuron"])
#     in_spikes = in_spikes.with_columns(
#         modulo_with_offset(pl.col("time"), pl.col("period"), pl.col("origin"))
#     )
#     return in_spikes


# def transfer_through_channels(spikes, channels, time_str, sort=False):
#     states = channels.join(spikes, left_on="source", right_on="neuron", how="inner")
#     states = states.with_columns(
#         (pl.col("time") + pl.col("delay")).alias(time_str),
#         # pl.lit(None, pl.Float64).alias("length"),
#         # pl.lit(None, pl.Float64).alias("coef_0"),
#         # pl.lit(None, pl.Float64).alias("coef_1"),
#     )
#     # states = states.drop("target", "delay", "time")
#     if sort:
#         return states.sort(time_str)
#     return states
