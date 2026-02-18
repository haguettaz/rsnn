import polars as pl

import rsnn_plugin as rp
from rsnn import REFRACTORY_RESET


def modulo_with_offset(x, period, offset):
    """Compute modulo operation with custom offset for periodic boundary conditions.

    Performs modular arithmetic with a specified offset, useful for handling
    periodic spike trains and temporal boundary conditions in neural simulations.

    Args:
        x (pl.Expr): Input values to apply modulo operation.
        period (pl.Expr): Period for the modulo operation.
        offset (pl.Expr): Offset value to define the modulo base.

    Returns:
        pl.Expr: Result of (x - offset) mod period + offset.

    Notes:
        Equivalent to: x - period * floor((x - offset) / period)
        Useful for wrapping spike times within periodic boundaries.
    """
    return (x - offset).mod(period) + offset


def compute_states(
    syn_states: pl.DataFrame,
    rec_states: pl.DataFrame,
    out_spikes: pl.DataFrame,
    # reset: float,
    scan: bool = False,
) -> pl.DataFrame:
    """Compute neuronal states from input and output spikes for analysis.

    Calculates the complete set of neuronal states including synaptic transmission events, refractory periods, and firing events.
    Combines all state types without temporal scanning.

    Args:
        out_spikes (pl.DataFrame): Output spike data with columns 'f_index', 'time'.
        in_spikes (pl.DataFrame): Input spike data with columns 'f_index', 'time', 'weight'. Includes synaptic input and refractory events.
        reset (float): Neuronal reset value after firing.

    Returns:
        pl.DataFrame: All states with columns 'f_index', 'start', 'weight', 'in_coef_0', 'in_coef_1', 'active', 'length', and sorted by start time over each firing index group.
    """
    # Firing states
    f_states = out_spikes.select(
        pl.col("f_index"),
        pl.col("time").alias("start"),
        pl.lit(None, pl.Float64).alias("in_coef_0"),
        pl.lit(None, pl.Float64).alias("in_coef_1"),
    )

    # Input states: recovery and synaptic
    rec_states = rec_states.select(
        pl.col("f_index"),
        pl.col("start"),
        pl.col("weight").alias("in_coef_0"),
        pl.lit(0.0, pl.Float64).alias("in_coef_1"),
    )
    syn_states = syn_states.select(
        pl.col("f_index"),
        pl.col("start"),
        pl.lit(0.0, pl.Float64).alias("in_coef_0"),
        pl.col("weight").alias("in_coef_1"),
    )
    states = pl.concat([rec_states, syn_states, f_states]).sort(
        "f_index", "start", maintain_order=True
    )

    # in_states = in_spikes.select(
    #     pl.col("f_index"),
    #     pl.col("time").alias("start"),
    #     pl.when(pl.col("weight").is_not_null())
    #     .then(0.0)
    #     .otherwise(reset)
    #     .alias("in_coef_0"),
    #     pl.when(pl.col("weight").is_not_null())
    #     .then(pl.col("weight"))
    #     .otherwise(0.0)
    #     .alias("in_coef_1"),
    # )

    return scan_states(states) if scan else states


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
    states = (
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
    return states
