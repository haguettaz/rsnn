from typing import Callable, List, Optional, Tuple, Union

import gurobipy as gp
import numpy as np
import polars as pl
import scipy.sparse as ss
from numpy.typing import NDArray

import rsnn_plugin as rp
from rsnn import REFRACTORY_RESET
from rsnn.utils import modulo_with_offset, scan_states


def scan_with_weights(states: pl.DataFrame, weights: np.ndarray) -> pl.DataFrame:
    """Update state coefficients with new synaptic weights using cumulative scanning.

    Recalculates the membrane potential coefficients (coef_0, coef_1) for all states after updating synaptic weights. Uses temporal scanning to accumulate the effects of synaptic inputs over time with exponential decay.

    Args:
        states (pl.DataFrame): Neuronal states with columns 'f_index', 'start', 'length', 'in_index', 'in_coef_0', 'in_coef_1', 'weight'. Must be sorted by start time within each f_index group
        weights (np.ndarray): New synaptic weight values to apply.

    Returns:
        pl.DataFrame: Updated states with recalculated 'weight', 'coef_0', and 'coef_1' columns.

    Notes:
        - Uses exponential decay scanning for temporal dynamics
        - Updates both constant (coef_0) and linear (coef_1) membrane potential terms
    """
    return (
        states.update(
            pl.DataFrame({"weight": weights}).with_row_index("in_index"),
            on="in_index",
        )
        .with_columns(
            rp.scan_coef_1(
                pl.col("length").shift(), pl.col("in_coef_1") * pl.col("weight")
            )
            .over("f_index")
            .alias("coef_1")
        )
        .with_columns(
            rp.scan_coef_0(
                pl.col("length").shift(),
                pl.col("coef_1").shift(),
                pl.col("in_coef_0") * pl.col("weight"),
            )
            .over("f_index")
            .alias("coef_0")
        )
    )


def init_spikes(
    out_spikes: pl.DataFrame,
    syn_spikes: pl.DataFrame,
    period: Optional[float] = None,
    f_index_min: int = 0,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Results are sorted by time.

    Args:
        out_spikes (pl.DataFrame): sorted by time.
        syn_spikes (pl.DataFrame): _description_
        period (float): _description_

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: _description_
    """

    if period is None:
        tmin = out_spikes.select(pl.min("time")).item()
        out_spikes = out_spikes.with_columns(
            pl.int_range(f_index_min, f_index_min + pl.len(), dtype=pl.UInt32).alias(
                "f_index"
            ),
            pl.col("time") - tmin,
        )  # Note: all output spikes are within [0, +inf)
        in_spikes = (
            out_spikes.select(
                pl.col("f_index"),
                pl.lit(None, pl.UInt32).alias("in_index"),
                pl.col("time").shift().fill_null(float("-inf")),
            )
            .extend(
                syn_spikes.select(
                    pl.lit(None, pl.UInt32).alias("f_index"),
                    pl.col("in_index"),
                    pl.col("time") - tmin,
                )
            )
            .sort("time", maintain_order=True)
            .with_columns(pl.col("f_index").forward_fill())
        )  # Note: all input spikes are within [-inf, +inf)

    else:
        tmax = out_spikes.select(pl.max("time")).item()
        out_spikes = out_spikes.with_columns(
            pl.int_range(f_index_min, f_index_min + pl.len(), dtype=pl.UInt32).alias(
                "f_index"
            ),
            pl.col("time") - tmax + period,
        )  # Note: all output spikes are within (0, period]
        in_spikes = (
            out_spikes.select(
                pl.col("f_index"),
                pl.lit(None, pl.UInt32).alias("in_index"),
                pl.col("time").shift().fill_null(0),
            )
            .extend(
                syn_spikes.select(
                    pl.lit(None, pl.UInt32).alias("f_index"),
                    pl.col("in_index"),
                    (pl.col("time") - tmax).mod(period),
                )
            )
            .sort("time", maintain_order=True)
            .with_columns(pl.col("f_index").forward_fill())
        )  # Note: all input spikes are within [0, period)

        # in_spikes = (
        #     in_spikes.with_columns(
        #         ((pl.col("time") - tmax).mod(period)).alias("time"),
        #         pl.lit(True, pl.Boolean).alias("in"),
        #     )  # Note: all input spikes are within [0, period)
        #     .extend(
        #         out_spikes.with_columns(
        #             pl.lit(False, pl.Boolean).alias("in"),
        #         )
        #     )
        #     .sort("time", maintain_order=True)
        #     .with_columns(pl.col("f_index").backward_fill().over("index"))
        #     .drop_nulls("f_index")
        #     .filter(pl.col("in"))
        # )

    return out_spikes, in_spikes


def init_spikes_multi(
    out_spikes: List[pl.DataFrame],
    in_spikes: List[pl.DataFrame],
    period: Union[float, List[float]],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Results are sorted by time.

    Args:
        out_spikes (pl.DataFrame): sorted by time.
        in_spikes (pl.DataFrame): _description_.
        periods (pl.DataFrame): _description_

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: _description_
    """
    out_schema = {"f_index": pl.UInt32, "time": pl.Float64}
    in_schema = {"f_index": pl.UInt32, "time": pl.Float64, "in_index": pl.UInt32}

    ext_out_spikes = pl.DataFrame(schema=out_schema)
    ext_in_spikes = pl.DataFrame(schema=in_schema)

    if len(out_spikes) != len(in_spikes):
        raise ValueError("Length mismatch between out_spikes and in_spikes.")

    if not isinstance(period, (float, list)):
        raise ValueError("Incorrect type for period argument.")

    if isinstance(period, list):
        if len(period) != len(out_spikes):
            raise ValueError("Length mismatch between periods and spikes.")

        if any(not isinstance(p, float) for p in period):
            raise ValueError("All periods must be floats when providing a list.")

        last_f_index = 0
        for out_spikes_i, in_spikes_i, period_i in zip(out_spikes, in_spikes, period):
            out_spikes_i, in_spikes_i = init_spikes(
                out_spikes_i, in_spikes_i, period_i, last_f_index
            )
            last_f_index += out_spikes_i.height
            ext_out_spikes.extend(out_spikes_i)
            ext_in_spikes.extend(in_spikes_i)

    else:
        last_f_index = 0
        for out_spikes_i, in_spikes_i in zip(out_spikes, in_spikes):
            out_spikes_i, in_spikes_i = init_spikes(
                out_spikes_i, in_spikes_i, period, last_f_index
            )
            last_f_index += out_spikes_i.height
            ext_out_spikes.extend(out_spikes_i)
            ext_in_spikes.extend(in_spikes_i)

    return ext_out_spikes.sort("f_index", "time"), ext_in_spikes.sort("f_index", "time")


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


# def init_out_spikes(spikes: pl.DataFrame) -> pl.DataFrame:
#     """Initialize output spikes with indices and temporal information.

#     Processes (periodic) spike data to add firing indices (f_index) and previous spike times (time_prev) needed for neuronal state computations.

#     Args:
#         spikes (pl.DataFrame): Spike data with columns 'index', 'period', 'neuron', 'time'.

#     Returns:
#         pl.DataFrame: Processed spikes with additional columns 'f_index' and 'time_prev' for tracking firing events and refractory periods.

#     Notes:
#         The firing indices provide sequential numbering of spikes that is useful for grouping states.
#     """
#     return spikes.sort("time").select(
#         pl.col("index"),
#         pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
#         pl.col("neuron"),
#         pl.col("time"),
#         modulo_with_offset(
#             pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
#             pl.col("period"),
#             pl.col("time") - pl.col("period"),
#         )
#         .over(["index", "neuron"])
#         .alias("time_prev"),
#     )


def compute_states(
    out_spikes: pl.DataFrame,
    in_spikes: pl.DataFrame,
    reset: float,
    periods: pl.DataFrame,
) -> pl.DataFrame:
    """Compute all neuronal states for analysis.

    Calculates the complete set of neuronal states including synaptic transmission events, refractory periods, and firing events.
    Combines all state types and performs temporal scanning to compute membrane potential coefficients.

    Args:
        out_spikes (pl.DataFrame): Output spike data with columns 'index', 'f_index', 'time'.
        in_spikes (pl.DataFrame): Input spike data with columns 'index', 'in_index', 'time', 'weight'.
        threshold (float): Neuronal firing threshold.
        reset (float): Neuronal reset value after firing.
        periods (pl.DataFrame): Period information with columns 'index', 'period'.

    Returns:
        pl.DataFrame: Complete state representation with temporal dynamics with columns including 'index', 'neuron', 'f_index', 'start', 'length', 'coef_0', and 'coef_1'.

    Notes:
        The states are sorted by time with each firing index group and integrates multiple state types:
        - Refractory states from previous spikes
        - Synaptic transmission states from incoming connections
        - Firing states at exact spike times
    """
    lasts = (
        out_spikes.group_by("index")
        .agg(pl.max("time").alias("tmax"))
        .join(periods, on="index")
    )
    out_spikes = out_spikes.join(lasts, on="index").with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("f_index"),
        pl.col("time") - pl.col("tmax") + pl.col("period"),
    )  # Note: last output spike time is at period
    in_spikes = in_spikes.join(lasts, on="index").with_columns(
        (pl.col("time") - pl.col("tmax")).mod(pl.col("period"))
    )

    # Firing states
    f_states = out_spikes.select(
        pl.col("f_index"),
        pl.col("time").alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(None, pl.Float64).alias("weight"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
    )

    # Input states
    rec_states = out_spikes.select(
        pl.col("index"),
        pl.col("f_index"),
        pl.col("time").shift().fill_null(0).alias("start"),
        pl.lit(None, pl.UInt32).alias("in_index"),
        pl.lit(reset, pl.Float64).alias("weight"),
        pl.lit(1.0, pl.Float64).alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    )
    syn_states = in_spikes.select(
        pl.col("index"),
        pl.lit(None, pl.UInt32).alias("f_index"),
        pl.col("time").alias("start"),
        pl.col("in_index"),
        pl.lit(None, pl.Float64).alias("weight"),
        pl.lit(0.0, pl.Float64).alias("in_coef_0"),
        pl.lit(1.0, pl.Float64).alias("in_coef_1"),
    )
    in_states = (
        syn_states.extend(rec_states)
        .sort("start")
        .select(
            pl.col("f_index").forward_fill().over("index"),
            pl.col("start"),
            pl.col("in_index"),
            pl.col("weight"),
            pl.col("in_coef_0"),
            pl.col("in_coef_1"),
        )
    )

    # Combine all states
    states = (
        in_states.extend(f_states)
        .sort("f_index", "start")
        .with_columns(
            pl.col("start")
            .diff()
            .shift(-1, fill_value=0.0)
            .over("f_index")
            .alias("length"),
        )
    )

    return scan_states(states)

    # # Origins per (index, neuron)
    # origins = out_spikes.group_by(["index", "neuron"]).agg(
    #     pl.first("time_prev").alias("time_origin")
    # )

    # # Refractoriness
    # rec_states = out_spikes.select(
    #     pl.col("index"),
    #     pl.col("neuron"),
    #     pl.col("f_index"),
    #     pl.col("time_prev").alias("start"),
    #     pl.lit(REFRACTORY_RESET, pl.Float64).alias("in_coef_0"),
    #     pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    # )

    # # Synaptic transmission
    # syn_states = (
    #     origins.join(synapses, left_on="neuron", right_on="target")
    #     .join(src_spikes, left_on=["index", "source"], right_on=["index", "neuron"])
    #     .select(
    #         pl.col("index"),
    #         pl.col("neuron"),
    #         pl.lit(None, pl.UInt32).alias("f_index"),
    #         modulo_with_offset(
    #             pl.col("time") + pl.col("delay"),
    #             pl.col("period"),
    #             pl.col("time_origin"),
    #         ).alias("start"),
    #         pl.lit(0.0, pl.Float64).alias("in_coef_0"),
    #         pl.col("weight").alias("in_coef_1"),
    #     )
    # )

    # in_states = (
    #     syn_states.extend(rec_states)
    #     .sort("index", "neuron", "start")
    #     .select(
    #         pl.col("index"),
    #         pl.col("neuron"),
    #         pl.col("f_index").forward_fill().over(["index", "neuron"]),
    #         pl.col("start"),
    #         pl.col("in_coef_0"),
    #         pl.col("in_coef_1"),
    #     )
    # )

    # f_states = out_spikes.select(
    #     pl.col("index"),
    #     pl.col("neuron"),
    #     pl.col("f_index"),
    #     pl.col("time").alias("start"),
    #     pl.lit(None, pl.Float64).alias("in_coef_0"),
    #     pl.lit(None, pl.Float64).alias("in_coef_1"),
    # )

    # states = in_states.extend(f_states).sort("f_index", "start")

    # return scan_states(states)


def compute_nd_linear_map(
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
        syn_states (pl.DataFrame): Synaptic states with columns 'f_index', 'start', and 'in_index'.
        rec_states (pl.DataFrame): Refractory states with columns 'f_index', 'start', and 'weight'.
        times (pl.DataFrame): Evaluation times with columns 'f_index', 'time', and 't_index'.
        deriv (int, optional): Derivative order. Defaults to 0.

    Returns:
        tuple[scipy.sparse.csr_array, np.ndarray]: A tuple containing:
            - Sparse matrix mapping synaptic weights to membrane potential
            - Offset vector from refractory contributions
    """
    n = times.height

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
                * pl.col("weight")
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
        shape=(n, n_synapses),
    )

    b = rec_offset.sort("t_index").get_column("coef").to_numpy()

    return lambda x_: A @ x_ + b  # type: ignore


def compute_1d_linear_map(
    n_synapses: int,
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    times: dict,
    deriv: int = 0,
) -> Callable[[gp.MVar], gp.LinExpr]:
    """Compute linear mapping for membrane potential constraints.

    Calculates the linear relationship between synaptic weights and membrane potential (or its derivatives) at specified times, e.g., to enforce firing constraints and sufficient rise conditions in optimization.

    Args:
        n_synapses (int): Total number of synapses for sparse matrix dimensions.
        syn_states (pl.DataFrame): Synaptic states with columns 'f_index', 'start', and 'in_index'.
        rec_states (pl.DataFrame): Refractory states with columns 'f_index', 'start', and 'weight'.
        times (pl.DataFrame): Evaluation times with columns 'f_index', 'time'.
        deriv (int, optional): Derivative order. Defaults to 0.

    Returns:
        tuple[scipy.sparse.csr_array, np.ndarray]: A tuple containing:
            - Sparse matrix mapping synaptic weights to membrane potential
            - Offset vector from refractory contributions
    """
    syn_coef = (
        syn_states.filter(
            pl.col("f_index") == times["f_index"], pl.col("start") <= times["time"]
        )
        .group_by("in_index")
        .agg(
            (
                (
                    (-1) ** (deriv % 2)
                    * (-deriv + (times["time"] - pl.col("start")))
                    * (pl.col("start") - times["time"]).exp()
                ).sum()
            ).alias("coef"),
        )
    )

    b = (
        rec_states.filter(
            pl.col("f_index") == times["f_index"], pl.col("start") <= times["time"]
        )
        .select(
            (
                ((pl.col("start") - times["time"]).exp()).sum()
                * (-1) ** (deriv % 2)
                * pl.col("weight")
            )
        )
        .item()
    )

    a = np.zeros(n_synapses)
    a[syn_coef["in_index"].to_numpy()] = syn_coef.get_column("coef").to_numpy()
    # b = rec_offset["coef"].item()

    return lambda x_: a @ x_ + b  # type: ignore


def find_max_violations(states: pl.DataFrame, tol: float = 1e-5) -> pl.DataFrame:
    """Find maximum violations of the neuron potential bounds  per interspike interval (grouped by 'f_index').

    Args:
        states (pl.DataFrame): Collection of states with columns 'f_index', 'start', 'length', 'coef_0', 'coef_1', 'rhs_coef_0', 'rhs_coef_1'.
        tol (float, optional): Tolerance for considering a violation significant. Defaults to 1e-6.

    Returns:
        pl.DataFrame: DataFrame containing the maximum violations with columns 'f_index', 'tmax', 'zmax', 'bound'.
    """
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


def get_nrg_matrix(
    n_synapses: int,
    syn_states: pl.DataFrame,
    diag: bool = False,
    # first_order: bool = False,
    # full: bool = True,
    # last_only: bool = False,
) -> ss.csr_array[np.float64]:
    """Compute energy-based metrics for synaptic weights optimization.

    Calculates the weighted mean (linear term) and precision matrix (quadratic term) for the energy-based objective function.
    The metric can be computed either synapse-locally (diagonal) or neuron-locally (full matrix).

    Args:
        n_synapses (int): Number of synapses to optimize.
        syn_states (pl.DataFrame): Synaptic states containing 'f_index', 'in_index', 'start', and 'end'.
        l2_reg (float, optional): L2 regularization coefficient. Defaults to 1.0.

    Returns:
        Callable[[gp.MVar], gp.MQuadExpr]: A quadratic function of the synaptic weights.
    """

    if diag:
        syn_energies = (
            syn_states.join(syn_states, on=["f_index", "end", "in_index"])
            .group_by("in_index")
            .agg(
                rp.inner_syn_2nd(pl.col("start"), pl.col("start_right"), pl.col("end"))
                .sum()
                .alias("energy")
            )
        )
        return dataframe_to_2d_array(
            syn_energies,
            "in_index",
            "in_index",
            "energy",
            (n_synapses, n_synapses),
        )

    syn_energies = (
        syn_states.join(syn_states, on=["f_index", "end"])
        .group_by(["in_index", "in_index_right"])
        .agg(
            rp.inner_syn_2nd(pl.col("start"), pl.col("start_right"), pl.col("end"))
            .sum()
            .alias("energy")
        )
    )
    return dataframe_to_2d_array(
        syn_energies,
        "in_index",
        "in_index_right",
        "energy",
        (n_synapses, n_synapses),
    )

    # else:
    #     if first_order:
    #         syn_energies = get_diag_nrg_1st(syn_states, last_only=last_only)
    #     else:
    #         syn_energies = get_diag_nrg_2nd(syn_states, last_only=last_only)

    #     Q = dataframe_to_2d_array(
    #         syn_energies,
    #         "in_index",
    #         "in_index",
    #         "energy",
    #         (n_synapses, n_synapses),
    #     )
