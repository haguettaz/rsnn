import polars as pl

from .utils import modulo_with_offset


def extend_with_time_prev(spikes, over):
    """Sort if necessary"""
    # return spikes.sort("time").with_columns(
    #     pl.when(pl.col("period").is_not_null() & pl.col("period").is_finite())
    #     .then(
    #         modulo_with_offset(
    #             pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
    #             pl.col("period"),
    #             pl.col("time") - pl.col("period"),
    #         )
    #     )
    #     .otherwise(pl.col("time").shift())
    #     .over(["index", "neuron"])
    #     .alias("time_prev")
    # )
    return spikes.sort("time").with_columns(
        modulo_with_offset(
            pl.col("time").gather((pl.int_range(pl.len()) - 1) % pl.len()),
            pl.col("period"),
            pl.col("time") - pl.col("period"),
        )
        .over(over)
        .alias("time_prev")
    )


# def extend_with_time_origin(spikes):
#     origins = spikes.group_by(["index", "neuron"]).agg(
#         pl.min("time_prev").alias("time_origin")
#     )
#     return spikes.join(origins, on=["index", "neuron"])


# def extend_with_indices(spikes):
#     return
