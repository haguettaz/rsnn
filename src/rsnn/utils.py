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
