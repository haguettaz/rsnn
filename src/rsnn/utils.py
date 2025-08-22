def modulo_with_offset(x, period, offset):
    return x - period * (x - offset).floordiv(period)
