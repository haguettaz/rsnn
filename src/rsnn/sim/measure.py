from typing import List, Optional, Tuple

import numpy as np
import polars as pl

import rsnn_plugin as rp
from rsnn import REFRACTORY_PERIOD
from rsnn.log import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")


def compute_similarities(
    spikes, spikes_ref, shifts, n_neurons, eps=REFRACTORY_PERIOD / 2
):
    # Adjust spikes
    spikes = spikes.join(
        pl.DataFrame({"neuron": np.arange(n_neurons, dtype=np.uint32)}),
        on="neuron",
        how="right",
    )
    spikes = spikes.join(shifts, how="cross")
    spikes = spikes.select(
        pl.col("neuron"),
        pl.col("delta"),
        pl.col("time") + pl.col("delta"),
    )
    spikes = spikes.sort("time")

    # Number of spikes per neuron and number of active neurons
    n_spikes_per_neuron = spikes.drop_nulls().group_by("delta", "neuron").agg(pl.len())
    n_silent_neurons = n_spikes_per_neuron.group_by("delta").agg(
        (n_neurons - pl.n_unique("neuron")).alias("n_silent")
    )

    # Adjust time limits before comparison with the periodic reference
    time_lims = spikes.group_by("delta", "neuron").agg(
        (pl.col("time").first() - eps).alias("time_min"),
        (pl.col("time").last() + eps).alias("time_max"),
    )

    # Periodic extension of the reference
    spikes_ref = (
        spikes_ref.join(time_lims, on="neuron", how="left")
        .with_columns(
            rp.extend_periodically(
                pl.col("time"),
                pl.col("period"),
                pl.col("time_min").fill_null(0.0),
                pl.col("time_max")
                .clip(pl.col("time_min") + pl.col("period"))
                .fill_null(pl.col("period")),
            ).alias("time"),
        )
        .explode("time")
    )

    # Number of spikes per neuron in the reference and number of active neurons in the reference
    n_spikes_ref_per_neuron = spikes_ref.group_by("delta", "neuron").agg(pl.len())
    n_silent_ref_neurons = n_spikes_ref_per_neuron.group_by("delta").agg(
        (n_neurons - pl.n_unique("neuron")).alias("n_silent")
    )

    spikes_ref = spikes_ref.filter(pl.col("time") <= pl.col("time_max")).sort("time")

    measures = (
        spikes.join_asof(
            spikes_ref,
            by=["delta", "neuron"],
            on="time",
            strategy="nearest",
            suffix="_ref",
            tolerance=eps,
            check_sortedness=False,
            coalesce=False,
        )
        .drop_nulls(["time", "time_ref"])  # useless?
        .group_by("delta", "neuron")
        .agg(
            (1 - ((pl.col("time") - pl.col("time_ref")).abs() / eps))
            .sum()
            .alias("sum_per_neuron")
        )
    )

    precisions = (
        n_spikes_per_neuron.join(measures, on=["delta", "neuron"], how="left")
        .with_columns(
            (pl.col("sum_per_neuron") / pl.col("len")).alias("mean_per_neuron")
        )
        .group_by("delta")
        .agg((pl.col("mean_per_neuron").sum() / n_neurons).alias("precision"))
        .join(n_silent_neurons, on="delta")
        .select(
            pl.col("delta").alias("delta_precision"),
            (pl.col("precision") + pl.col("n_silent") / n_neurons).alias(
                "meas_precision"
            ),
        )
    )

    recalls = (
        n_spikes_ref_per_neuron.join(measures, on=["delta", "neuron"], how="left")
        .with_columns(
            (pl.col("sum_per_neuron") / pl.col("len")).alias("mean_per_neuron")
        )
        .group_by("delta")
        .agg((pl.col("mean_per_neuron").sum() / n_neurons).alias("recall"))
        .join(n_silent_ref_neurons, on="delta")
        .select(
            pl.col("delta").alias("delta_recall"),
            (pl.col("recall") + pl.col("n_silent") / n_neurons).alias("meas_recall"),
        )
    )

    return precisions, recalls
