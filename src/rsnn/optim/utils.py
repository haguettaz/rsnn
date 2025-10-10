from typing import Callable

import gurobipy as gp
import numpy as np
import polars as pl
import scipy.sparse as ss
from numpy.typing import NDArray

import rsnn_plugin as rp
from rsnn import REFRACTORY_RESET
from rsnn.log import setup_logging
from rsnn.utils import modulo_with_offset

# from ..states import compute_max_violations

logger = setup_logging(__name__, console_level="DEBUG", file_level="DEBUG")


def init_synapses(synapses: pl.DataFrame, spikes: pl.DataFrame) -> pl.DataFrame:
    """Initialize and filter synapses to include only active connections.

    Prunes synapses to keep only those that have a source neuron that spikes at least once, then adds a unique in_index to each remaining synapse for optimization purposes.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns including 'source'.
        spikes (pl.DataFrame): Spike train data with columns including 'neuron'.

    Returns:
        pl.DataFrame: Filtered synapses with added 'in_index' column containing unique identifiers for each synapse.

    Notes:
        Only synapses whose source neurons appear in the spike data are retained.
        The in_index provides a compact indexing scheme for optimization variables.
        An existing in_index column will be overwritten.
    """
    return synapses.join(
        spikes, left_on="source", right_on="neuron", how="semi"
    ).with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("in_index"))


def init_out_spikes(spikes: pl.DataFrame) -> pl.DataFrame:
    """Initialize output spikes with indices and temporal information.

    Processes (periodic) spike data to add firing indices (f_index) and previous spike times (time_prev) needed for neuronal state computations.

    Args:
        spikes (pl.DataFrame): Spike data with columns 'index', 'period', 'neuron', 'time'.

    Returns:
        pl.DataFrame: Processed spikes with additional columns 'f_index' and 'time_prev' for tracking firing events and refractory periods.

    Notes:
        The firing indices provide sequential numbering of spikes that is useful for grouping states.
    """
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


def compute_states(
    synapses: pl.DataFrame, out_spikes: pl.DataFrame, src_spikes: pl.DataFrame
) -> pl.DataFrame:
    """Compute all neuronal states for analysis.

    Calculates the complete set of neuronal states including synaptic transmission events, refractory periods, and firing events. Combines all state types and performs temporal scanning to compute membrane potential coefficients.

    Args:
        synapses (pl.DataFrame): Synaptic connections with columns including 'source', 'target', 'delay', 'weight'.
        out_spikes (pl.DataFrame): Output spike data with columns 'index', 'period', 'neuron', 'f_index', 'time', 'time_prev'. Must be sorted by time within each (index, neuron) group.
        src_spikes (pl.DataFrame): Source spike data with columns 'index', 'period', 'neuron', 'time'.

    Returns:
        pl.DataFrame: Complete state representation with temporal dynamics with columns including 'index', 'neuron', 'f_index', 'start', 'length', 'coef_0', and 'coef_1'.

    Notes:
        The states are sorted by time with each firing index group and integrates multiple state types:
        - Refractory states from previous spikes
        - Synaptic transmission states from incoming connections
        - Firing states at exact spike times
    """
    # Origins per (index, neuron)
    origins = out_spikes.group_by(["index", "neuron"]).agg(
        pl.first("time_prev").alias("time_origin")
    )

    # Refractoriness
    rec_states = out_spikes.select(
        pl.col("index"),
        pl.col("neuron"),
        pl.col("f_index"),
        pl.col("time_prev").alias("start"),
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
            pl.lit(0.0, pl.Float64).alias("in_coef_0"),
            pl.col("weight").alias("in_coef_1"),
        )
    )

    in_states = (
        syn_states.extend(rec_states)
        .sort("index", "neuron", "start")
        .select(
            pl.col("index"),
            pl.col("neuron"),
            pl.col("f_index").forward_fill().over(["index", "neuron"]),
            pl.col("start"),
            pl.col("in_coef_0"),
            pl.col("in_coef_1"),
        )
    )

    f_states = out_spikes.select(
        pl.col("index"),
        pl.col("neuron"),
        pl.col("f_index"),
        pl.col("time").alias("start"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
    )

    states = in_states.extend(f_states).sort("f_index", "start")

    return scan_states(states)


def scan_states(states: pl.DataFrame) -> pl.DataFrame:
    """Scan states to compute membrane potential coefficients with temporal dynamics.

    Performs temporal scanning of neuronal states to compute the membrane potential coefficients (coef_0, coef_1) that describe the linear evolution of the potential between discrete events with exponential decay.

    Args:
        states (pl.DataFrame): Neuronal states with columns 'f_index', 'start', 'in_coef_0', 'in_coef_1' and temporal ordering. Must be sorted by starting time within each firing index group.

    Returns:
        pl.DataFrame: States with added columns:
            - length: Duration of each state interval
            - coef_0: Constant term of membrane potential
            - coef_1: Linear term of membrane potential

    Notes:
        Uses exponential decay scanning to accumulate the effects of synaptic inputs over time. The coefficients describe how the membrane potential evolves as z(start + dt) = (coef_0 + coef_1 * dt) * exp(- dt) for 0 <= dt < length.
    """
    return (
        states.with_columns(
            pl.col("start")
            .diff()
            .shift(-1, fill_value=0.0)
            .over("f_index")
            .alias("length")
        )
        .with_columns(
            rp.scan_coef_1(pl.col("length").shift(), pl.col("in_coef_1"))
            .over("f_index")
            .alias("coef_1")
        )
        .with_columns(
            rp.scan_coef_0(
                pl.col("length").shift(),
                pl.col("coef_1").shift(),
                pl.col("in_coef_0"),
            )
            .over("f_index")
            .alias("coef_0")
        )
    )


def compute_linear_map(
    n_synapses: int,
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    times: pl.DataFrame,
    deriv: int = 0,
) -> Callable[[gp.MVar], gp.LinExpr]:
    """Compute linear mapping for membrane potential constraints.

    Calculates the linear relationship between synaptic weights and membrane potential (or its derivatives) at specified times, e.g., to enforce firing constraints and sufficient rise conditions in optimization.

    Args:
        n_synapses (int): Total number of synapses for sparse matrix dimensions.
        syn_states (pl.DataFrame): Synaptic states with columns 'f_index', 'start', 'in_index'.
        rec_states (pl.DataFrame): Refractory states with columns 'f_index', 'start'.
        times (pl.DataFrame): Evaluation times with columns 'f_index', 'time'.
        deriv (int, optional): Derivative order. Defaults to 0.

    Returns:
        tuple[scipy.sparse.csr_array, np.ndarray]: A tuple containing:
            - Sparse matrix mapping synaptic weights to membrane potential
            - Offset vector from refractory contributions
    """
    syn_coef = (
        times.join(syn_states, on="f_index")
        .filter(pl.col("time") >= pl.col("start"))
        .group_by("t_index", "in_index")
        .agg(
            (
                (
                    (-1) ** (deriv % 2)
                    * (-deriv + (pl.col("time") - pl.col("start")))
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
                ((pl.col("start") - pl.col("time")).exp()).sum()
                * (-1) ** (deriv % 2)
                * REFRACTORY_RESET
            ).alias("coef"),
        )
    )

    A = ss.csr_array(
        (
            (syn_coef.get_column("coef").to_numpy()),
            (
                syn_coef.get_column("t_index").to_numpy(),
                syn_coef.get_column("in_index").to_numpy(),
            ),
        ),
        shape=(times.height, n_synapses),
    )

    b = rec_offset.sort("t_index").get_column("coef").to_numpy()

    return lambda x_: A @ x_ + b  # type: ignore


def find_max_violations(states: pl.DataFrame, tol: float = 1e-6) -> pl.DataFrame:
    return (
        states.with_columns(
            rp.critical_dtime(
                pl.col("length"),
                pl.col("coef_0"),
                pl.col("coef_1"),
                pl.col("rhs_coef_1"),
            ).alias("dtc")
        )
        .with_columns(
            (
                (pl.col("coef_0") + pl.col("coef_1") * pl.col("dtc"))
                * (-pl.col("dtc")).exp()
                - (pl.col("rhs_coef_0") + pl.col("rhs_coef_1") * pl.col("dtc"))
            ).alias("vdtc"),
            (pl.col("coef_0") - pl.col("rhs_coef_0")).alias("vstart"),
            (
                (
                    pl.col("coef_0")
                    + pl.col("coef_1") * (pl.col("length") - 1e-9).clip(0.0)
                )
                * (-(pl.col("length") - 1e-9).clip(0.0)).exp()
                - (
                    pl.col("rhs_coef_0")
                    + pl.col("rhs_coef_1") * (pl.col("length") - 1e-9).clip(0.0)
                )
            ).alias("vend"),
        )
        .with_columns(
            pl.when(pl.col("dtc").is_null())
            .then(
                pl.when(pl.col("vstart") >= pl.col("vend"))
                .then(pl.lit(0.0))
                .otherwise("length")
            )
            .otherwise(pl.col("dtc"))
            .alias("dtime")
        )
        .with_columns(
            (pl.col("start") + pl.col("dtime")).alias("tmax"),
            (
                (pl.col("coef_0") + pl.col("coef_1") * pl.col("dtime"))
                * (-pl.col("dtime")).exp()
                # - (pl.col("rhs_coef_0") + pl.col("rhs_coef_1") * pl.col("dtime"))
            ).alias("zmax"),
            (pl.col("rhs_coef_0") + pl.col("rhs_coef_1") * pl.col("dtime")).alias(
                "bound"
            ),
        )
        .filter(pl.col("zmax") > pl.col("bound") + tol)
        .group_by("f_index")
        .agg(
            pl.col("tmax", "zmax", "bound").top_k_by(
                pl.col("zmax") - pl.col("bound"), k=1
            )
        )
        .explode("tmax", "zmax", "bound")
    )


# return (
#     states.with_columns(
#         rp.critical_dtime(
#             pl.col("length"),
#             pl.col("coef_0"),
#             pl.col("coef_1"),
#             pl.col("rhs_coef_1"),
#         ).alias("dtc")
#     )
#     .with_columns(
#         (
#             (pl.col("coef_0") + pl.col("coef_1") * pl.col("dtc"))
#             * (-pl.col("dtc")).exp()
#             - (pl.col("rhs_coef_0") + pl.col("rhs_coef_1") * pl.col("dtc"))
#         ).alias("vdtc"),
#         (pl.col("coef_0") - pl.col("rhs_coef_0")).alias("vstart"),
#         (
#             (pl.col("coef_0") + pl.col("coef_1") * pl.col("length"))
#             * (-pl.col("length")).exp()
#             - (pl.col("rhs_coef_0") + pl.col("rhs_coef_1") * pl.col("length"))
#         ).alias("vend"),
#     )
#     .with_columns(
#         pl.when(pl.col("dtc").is_null())
#         .then(
#             pl.when(pl.col("vstart") >= pl.col("vend"))
#             .then(pl.lit(0.0))
#             .otherwise("length")
#         )
#         .otherwise(pl.col("dtc"))
#         .alias("dtime")
#     )
#     .with_columns(
#         (pl.col("start") + pl.col("dtime")).alias("tmax"),
#         (
#             (pl.col("coef_0") + pl.col("coef_1") * pl.col("dtime"))
#             * (-pl.col("dtime")).exp()
#             - (pl.col("rhs_coef_0") + pl.col("rhs_coef_1") * pl.col("dtime"))
#         ).alias("vmax"),
#         (pl.col("rhs_coef_0") + pl.col("dtime") * pl.col("rhs_coef_1")).alias(
#             "bound"
#         ),
#     )
#     .filter(pl.col("vmax") > 0.0)
#     .group_by("f_index")
#     .agg(pl.col("tmax", "vmax", "bound").top_k_by("vmax", k=1))
#     .explode("tmax", "vmax", "bound")
# )


def dataframe_to_1d_array(
    dataframe: pl.DataFrame, index: str, value: str, shape: int
) -> NDArray[np.float64]:
    """Convert DataFrame columns to a 1D numpy array with specified indexing.

    Creates a zero-initialized array and populates it using DataFrame values
    at positions specified by the index column. Useful for converting sparse
    DataFrame representations to dense arrays.

    Args:
        dataframe (pl.DataFrame): Input DataFrame with index and value columns.
        index (str): Column name containing array indices.
        value (str): Column name containing values to insert.
        shape (int): Length of the output array.

    Returns:
        np.ndarray: 1D array with values inserted at specified indices,
            zeros elsewhere.

    Notes:
        Assumes index values are valid array indices within [0, shape).
        Multiple values for the same index will overwrite previous values.
    """
    arr = np.zeros(shape)
    arr[dataframe.get_column(index).to_numpy()] = dataframe.get_column(value).to_numpy()
    return arr


def dataframe_to_2d_array(
    dataframe: pl.DataFrame,
    index_row: str,
    index_col: str,
    value: str,
    shape: tuple[int, int],
) -> ss.csr_array[np.float64]:
    """Convert DataFrame to a symmetric 2D sparse array.

    Creates a symmetric sparse matrix from DataFrame data by constructing
    the upper/lower triangle and then symmetrizing. Useful for creating
    precision matrices and other symmetric structures from sparse data.

    Args:
        dataframe (pl.DataFrame): Input DataFrame with row index, column index,
            and value columns.
        index_row (str): Column name containing row indices.
        index_col (str): Column name containing column indices.
        value (str): Column name containing matrix values.
        shape (tuple[int, int]): Shape of the output matrix (rows, cols).

    Returns:
        scipy.sparse.csr_array: Symmetric sparse matrix where
            result[i,j] = result[j,i] for all valid i,j.

    Notes:
        The symmetrization is performed as: A + A.T - diag(A)
        This avoids double-counting diagonal elements while ensuring symmetry.
    """
    return ss.csr_array(
        (
            (dataframe.get_column(value).to_numpy()),
            (
                dataframe.get_column(index_row).to_numpy(),
                dataframe.get_column(index_col).to_numpy(),
            ),
        ),
        shape=shape,
    )


def dataframe_to_sym_2d_array(
    dataframe: pl.DataFrame,
    index_row: str,
    index_col: str,
    value: str,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Convert DataFrame to a symmetric 2D sparse array.

    Creates a symmetric sparse matrix from DataFrame data by constructing
    the upper/lower triangle and then symmetrizing. Useful for creating
    precision matrices and other symmetric structures from sparse data.

    Args:
        dataframe (pl.DataFrame): Input DataFrame with row index, column index,
            and value columns.
        index_row (str): Column name containing row indices.
        index_col (str): Column name containing column indices.
        value (str): Column name containing matrix values.
        shape (tuple[int, int]): Shape of the output matrix (rows, cols).

    Returns:
        scipy.sparse.csr_array: Symmetric sparse matrix where
            result[i,j] = result[j,i] for all valid i,j.

    Notes:
        The symmetrization is performed as: A + A.T - diag(A)
        This avoids double-counting diagonal elements while ensuring symmetry.
    """
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
