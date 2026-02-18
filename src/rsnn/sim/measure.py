import numpy as np
import polars as pl

import rsnn_plugin as rp
from rsnn import REFRACTORY_PERIOD


def compute_upper_bounds(spikes_ref, spikes, n_neurons):
    n_spikes_per_neuron = (
        spikes_ref.group_by("neuron")
        .agg(pl.len().alias("n_spikes_ref"))
        .join(
            spikes.group_by("neuron").agg(pl.len().alias("n_spikes")),
            on="neuron",
            how="outer",
        )
        .with_columns(
            pl.min_horizontal(
                pl.col("n_spikes_ref").fill_null(0), pl.col("n_spikes").fill_null(0)
            ).alias("min_n_spikes"),
        )
    )

    pr_ub = (
        1
        - n_spikes_per_neuron.select(
            ((pl.col("n_spikes") - pl.col("min_n_spikes")) / pl.col("n_spikes")).sum()
        ).item()
        / n_neurons
    )
    rc_ub = (
        1
        - n_spikes_per_neuron.select(
            (
                (pl.col("n_spikes_ref") - pl.col("min_n_spikes"))
                / pl.col("n_spikes_ref")
            ).sum()
        ).item()
        / n_neurons
    )

    return pr_ub, rc_ub


# def compute_precision_upper_bound(spikes_ref, spikes, n_neurons):
#     n_spikes_per_neuron = (
#         spikes_ref.group_by("neuron")
#         .agg(pl.len().alias("n_spikes_ref"))
#         .join(
#             spikes.group_by("neuron").agg(pl.len().alias("n_spikes")),
#             on="neuron",
#             how="right",
#         )
#         .with_columns(
#             pl.min_horizontal(
#                 pl.col("n_spikes_ref").fill_null(0), pl.col("n_spikes").fill_null(0)
#             ).alias("min_n_spikes"),
#             # pl.max_horizontal(
#             #     [pl.col("n_spikes_ref").fill_null(0), pl.col("n_spikes").fill_null(0)]
#             # ).alias("max_n_spikes"),
#         )
#     )
#     # n_silent_neurons = n_neurons - n_spikes_per_neuron.height

#     return (
#         1
#         - n_spikes_per_neuron.select(
#             ((pl.col("n_spikes") - pl.col("min_n_spikes")) / pl.col("n_spikes")).sum()
#         ).item()
#         / n_neurons
#     )


# def compute_recall_upper_bound(spikes_ref, spikes, n_neurons):
#     n_spikes_per_neuron = (
#         spikes_ref.group_by("neuron")
#         .agg(pl.len().alias("n_spikes_ref"))
#         .join(
#             spikes.group_by("neuron").agg(pl.len().alias("n_spikes")),
#             on="neuron",
#             how="left",
#         )
#         .with_columns(
#             pl.min_horizontal(
#                 pl.col("n_spikes_ref").fill_null(0), pl.col("n_spikes").fill_null(0)
#             ).alias("min_n_spikes"),
#             # pl.max_horizontal(
#             #     [pl.col("n_spikes_ref").fill_null(0), pl.col("n_spikes").fill_null(0)]
#             # ).alias("max_n_spikes"),
#         )
#     )
#     # n_silent_neurons = n_neurons - n_spikes_per_neuron.height

#     return (
#         1
#         - n_spikes_per_neuron.select(
#             (
#                 (pl.col("n_spikes_ref") - pl.col("min_n_spikes"))
#                 / pl.col("n_spikes_ref")
#             ).sum()
#         ).item()
#         / n_neurons
#     )


# def compute_scores(spikes_ref, spikes, shifts, n_neurons, min_score=0.5):
#     """Compute similarity measures between shifted spike trains and the (periodic) reference.

#     Evaluates how well generated spike trains match reference periodic patterns
#     by computing similarity measures across different temporal shifts.
#     Uses nearest-neighbor matching with tolerance for comparing spike times.

#     Args:
#         spikes_ref (pl.DataFrame): Reference periodic spike patterns with columns
#             'neuron', 'time', 'period'.
#         spikes (pl.DataFrame): Generated spike trains with columns 'neuron', 'time'.
#         shifts (pl.DataFrame): Temporal shift values with column 'delta' for
#             alignment testing.
#         n_neurons (int): Total number of neurons in the network.
#         eps (float, optional): Tolerance for spike time matching, half of
#             refractory period by default.

#     Returns:
#         tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
#             - precisions: DataFrame with precision metrics per shift
#             - recalls: DataFrame with recall metrics per shift

#     Notes:
#         - Precision measures how many generated spikes match reference spikes
#         - Recall measures how many reference spikes are matched by generated spikes
#         - Silent neurons (no spikes) contribute positively to both metrics
#         - Reference patterns are extended periodically for comparison
#         - Uses temporal tolerance (eps) for nearest-neighbor matching
#     """
#     # Adjust spikes
#     # spikes = spikes.join(
#     #     pl.DataFrame({"neuron": np.arange(n_neurons, dtype=np.uint32)}),
#     #     on="neuron",
#     #     how="right",
#     # )

#     # Compute number of spikes per neuron
#     n_spikes_per_neuron = (
#         spikes_ref.group_by("neuron")
#         .agg(pl.len().alias("n_spikes_ref"))
#         .join(
#             spikes.group_by("neuron").agg(pl.len().alias("n_spikes")),
#             on="neuron",
#             how="outer",
#         )
#         .with_columns(
#             pl.min_horizontal(
#                 pl.col("n_spikes_ref").fill_null(0), pl.col("n_spikes").fill_null(0)
#             ).alias("min_n_spikes"),
#             pl.max_horizontal(
#                 pl.col("n_spikes_ref").fill_null(0), pl.col("n_spikes").fill_null(0)
#             ).alias("max_n_spikes"),
#         )
#     )
#     n_silent_neurons = n_neurons - n_spikes_per_neuron.height

#     upper_bound = (
#         n_silent_neurons
#         + n_spikes_per_neuron.select(
#             (pl.col("min_n_spikes") / pl.col("max_n_spikes")).sum()
#         ).item()
#     ) / n_neurons
#     if upper_bound < min_score:
#         logger.info(f"Early stopping: upper bound {upper_bound:.3f} < {min_score:.3f}")
#         return pl.DataFrame(
#             {
#                 "best_shift": [None],
#                 "best_score": [upper_bound],
#             }
#         )

#     spikes = (
#         spikes.join(shifts, how="cross")
#         .select(
#             pl.col("neuron"),
#             pl.col("shift"),
#             pl.col("time") + pl.col("shift"),
#         )
#         .sort("time")
#     )

#     # # Number of spikes per neuron and number of active neurons
#     # n_spikes_per_neuron = spikes.drop_nulls().group_by("shift", "neuron").agg(pl.len())
#     # n_silent_neurons = n_spikes_per_neuron.group_by("shift").agg(
#     #     (n_neurons - pl.n_unique("neuron")).alias("n_silent")
#     # )

#     # Adjust time limits before comparison with the periodic reference
#     time_lims = spikes.group_by("shift", "neuron").agg(
#         (pl.col("time").first() - REFRACTORY_PERIOD / 2).alias("time_min"),
#         (pl.col("time").last() + REFRACTORY_PERIOD / 2).alias("time_max"),
#     )

#     # Periodic extension of the reference
#     # min_time = spikes.select(pl.col("time").min()).item() - REFRACTORY_PERIOD / 2
#     # max_time = spikes.select(pl.col("time").max()).item() + REFRACTORY_PERIOD / 2
#     spikes_ref = (
#         spikes_ref.join(time_lims, on="neuron")
#         .with_columns(
#             rp.extend_periodically(
#                 pl.col("time"),
#                 pl.col("period"),
#                 pl.col("time_min"),
#                 pl.col("time_max"),
#             ).alias("time"),
#         )
#         .explode("time")
#         .sort("time")
#     )

#     # Number of spikes per neuron in the reference and number of active neurons in the reference
#     # n_spikes_ref_per_neuron = spikes_ref.group_by("shift", "neuron").agg(pl.len())
#     # n_silent_ref_neurons = n_spikes_ref_per_neuron.group_by("shift").agg(
#     #     (n_neurons - pl.n_unique("neuron")).alias("n_silent")
#     # )

#     # spikes_ref = spikes_ref.filter(pl.col("time") <= pl.col("time_max")).sort("time")

#     scores = (
#         spikes_ref.join_asof(
#             spikes,
#             by=["shift", "neuron"],
#             on="time",
#             strategy="nearest",
#             suffix="_ref",
#             tolerance=REFRACTORY_PERIOD / 2,
#             check_sortedness=False,
#             coalesce=False,
#         )
#         # .drop_nulls(["time", "time_ref"])  # useless?
#         .group_by("shift", "neuron")
#         .agg(
#             (1 - (2 * (pl.col("time") - pl.col("time_ref")).abs() / REFRACTORY_PERIOD))
#             .sum()
#             .alias("sum_per_neuron")
#         )
#         .join(n_spikes_per_neuron, on="neuron")
#         .group_by("shift")
#         .agg((pl.col("sum_per_neuron") / pl.col("max_n_spikes")).sum().alias("score"))
#     )
#     return scores.top_k(1, by="score").select(
#         pl.col("shift").alias("best_shift"),
#         ((pl.col("score") + n_silent_neurons) / n_neurons).alias("best_score"),
#     )

# precisions = (
#     n_spikes_per_neuron.join(measures, on=["shift", "neuron"], how="left")
#     .with_columns(
#         (pl.col("sum_per_neuron") / pl.col("len")).alias("mean_per_neuron")
#     )
#     .group_by("shift")
#     .agg((pl.col("mean_per_neuron").sum() / n_neurons).alias("precision"))
#     .join(n_silent_neurons, on="shift")
#     .select(
#         pl.col("shift").alias("delta_precision"),
#         (pl.col("precision") + pl.col("n_silent") / n_neurons).alias(
#             "meas_precision"
#         ),
#     )
# )

# recalls = (
#     n_spikes_ref_per_neuron.join(measures, on=["shift", "neuron"], how="left")
#     .with_columns(
#         (pl.col("sum_per_neuron") / pl.col("len")).alias("mean_per_neuron")
#     )
#     .group_by("shift")
#     .agg((pl.col("mean_per_neuron").sum() / n_neurons).alias("recall"))
#     .join(n_silent_ref_neurons, on="shift")
#     .select(
#         pl.col("shift").alias("delta_recall"),
#         (pl.col("recall") + pl.col("n_silent") / n_neurons).alias("meas_recall"),
#     )
# )

# return precisions, recalls


def compute_scores(
    true_spikes: pl.DataFrame,
    spikes: pl.DataFrame,
    n_neurons: int,
    shift: float,
    period: float,
):
    """Compute similarity measures between the provided spike train and the periodic reference, across different uniform time shifts.

    Uses optimal one-to-one assignment with nearest-neighbor matching and tolerance for comparing spike times.

    Args:
        true_spikes (pl.DataFrame): Reference periodic spike patterns with columns 'neuron', 'time'.
        spikes (pl.DataFrame): Actual spike trains with columns 'neuron', 'time'.
        n_neurons (int): Total number of neurons in the network.
        shift (float): Temporal shift to apply to the spike train for alignment with the reference.
        period (float): Period of the reference spike patterns.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
            - precisions: DataFrame with precision metrics per shift
            - recalls: DataFrame with recall metrics per shift

    Notes:
        - The reference spike train is periodic and satisfies a refractory period.
        - Precision measures how many generated spikes match reference spikes
        - Recall measures how many reference spikes are matched by generated spikes
    """
    # if which is None:
    #     which = "best"

    # if which not in {"best", "all"}:
    #     raise ValueError("which must be 'best' or 'all'")

    neurons = pl.DataFrame({"neuron": np.arange(n_neurons, dtype=np.uint32)})

    # Compute number of spikes per neuron in both spike trains
    n_spikes_per_neuron = (
        neurons.join(
            spikes.group_by("neuron").agg(pl.len().alias("n_spikes")),
            on="neuron",
            how="left",
        ).join(
            true_spikes.group_by("neuron").agg(pl.len().alias("n_true_spikes")),
            on="neuron",
            how="left",
        )
        # .fill_null(0)
    )

    # Construct the aligned spikes from the shifts
    spikes = spikes.select(
        pl.col("neuron"),
        (pl.col("time") + shift).mod(period).alias("time"),
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
    ).sort("time")

    # # Periodic extension of the reference to cover the range of spikes
    # time_lims = spikes.group_by("shift", "neuron").agg(
    #     (pl.col("time").first() - REFRACTORY_PERIOD / 2).alias("time_min"),
    #     (pl.col("time").last() + REFRACTORY_PERIOD / 2).alias("time_max"),
    # )

    ext_true_spikes = (
        true_spikes.with_columns(
            rp.extend_periodically(
                pl.col("time"),
                period=period,
                tmin=-REFRACTORY_PERIOD / 2,
                tmax=period + REFRACTORY_PERIOD / 2,
            ).alias("true_time"),
            pl.int_range(pl.len(), dtype=pl.UInt32).alias("true_id"),
        )
        .explode("true_time")
        .join(n_spikes_per_neuron, on="neuron")
        .select(pl.col("neuron"), pl.col("true_time"), pl.col("true_id"))
        .sort("true_time")
        # .match_to_schema(
        #     {
        #         "shift": pl.Float64,
        #         "neuron": pl.UInt32,
        #         "time": pl.Float64,
        #         # "n_spikes": pl.UInt32,
        #         "ref_id": pl.UInt32,
        #     }
        # )
    )
    # print(
    #     "comparison:",
    #     ext_true_spikes.join_asof(
    #         spikes,
    #         by="neuron",
    #         left_on="true_time",
    #         right_on="time",
    #         strategy="nearest",
    #         tolerance=REFRACTORY_PERIOD / 2,
    #         check_sortedness=False,
    #         coalesce=False,
    #     )
    #     .with_columns((pl.col("time") - pl.col("true_time")).abs().alias("delta"))
    #     .drop_nulls()
    #     .filter(pl.col("neuron") == 6),
    # )
    # print("extended spikes_ref:", ext_spikes_ref)

    scores = n_spikes_per_neuron.join(  # .drop_nulls("n_spikes")
        ext_true_spikes.join_asof(
            spikes,
            by="neuron",
            left_on="true_time",
            right_on="time",
            strategy="nearest",
            tolerance=REFRACTORY_PERIOD / 2,
            check_sortedness=False,
            coalesce=False,
        )
        .with_columns((pl.col("time") - pl.col("true_time")).abs().alias("delta"))
        .drop_nulls()
        .sort("delta")
        .unique(
            ["neuron", "true_id"], keep="first"
        )  # remove periodic duplicates, keeping the closest
        .group_by("neuron")
        .agg(
            (1.0 - 2.0 * pl.col("delta") / REFRACTORY_PERIOD)
            .sum()
            .alias("sum_per_neuron")
        ),
        on=["neuron"],
        how="left",
    )  # optimal one-to-one assignment between spikes

    precision = 1.0 - (
        scores.select(
            (1.0 - pl.col("sum_per_neuron").fill_null(0) / pl.col("n_spikes"))
        )
        .drop_nulls()
        .sum()
        .item()
        / n_neurons
    )
    recall = 1.0 - (
        scores.select(
            (1.0 - pl.col("sum_per_neuron").fill_null(0) / pl.col("n_true_spikes"))
        )
        .drop_nulls()
        .sum()
        .item()
        / n_neurons
    )

    # # print(precisions)

    # recalls = (
    #     scores.filter(pl.col("n_true_spikes") > 0)
    #     .group_by("shift")
    #     .agg(
    #         (1.0 - pl.col("sum_per_neuron").fill_null(0) / pl.col("n_true_spikes"))
    #         .drop_nulls()
    #         .sum()
    #         .alias("recall"),
    #     )
    #     .select(pl.col("shift"), (1.0 - pl.col("recall") / n_neurons).alias("recall"))
    # )
    # # print(recalls)

    return (precision, recall)
    # if which == "all":

    # return (
    #     precisions.top_k(1, by="precision"),
    #     recalls.top_k(1, by="recall"),
    # )


# def compute_precisions(spikes_ref, spikes, shifts, n_neurons, which=None):
#     """Compute similarity measures between shifted spike trains and the (periodic) reference.

#     Evaluates how well generated spike trains match reference periodic patterns
#     by computing similarity measures across different temporal shifts.
#     Uses nearest-neighbor matching with tolerance for comparing spike times.

#     Args:
#         spikes_ref (pl.DataFrame): Reference periodic spike patterns with columns
#             'neuron', 'time', 'period'.
#         spikes (pl.DataFrame): Generated spike trains with columns 'neuron', 'time'.
#         shifts (pl.DataFrame): Temporal shift values with column 'shift' for alignment testing.
#         n_neurons (int): Total number of neurons in the network.
#         which (str, optional): Which score to return, 'best' or 'all'.

#     Returns:
#         tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
#             - precisions: DataFrame with precision metrics per shift
#             - recalls: DataFrame with recall metrics per shift

#     Notes:
#         - Precision measures how many generated spikes match reference spikes
#         - Recall measures how many reference spikes are matched by generated spikes
#         - Silent neurons (no spikes) contribute positively to both metrics
#         - Reference patterns are extended periodically for comparison
#         - Uses temporal tolerance (eps) for nearest-neighbor matching
#     """
#     if which is None:
#         which = "best"

#     if which not in {"best", "all"}:
#         raise ValueError("which must be 'best' or 'all'")

#     neurons = pl.DataFrame({"neuron": np.arange(n_neurons, dtype=np.uint32)})

#     # Compute number of spikes per neuron
#     n_spikes_per_neuron = (
#         spikes.group_by("neuron")
#         .agg(pl.len().alias("n_spikes"))
#         .join(neurons, on="neuron", how="right")
#     )

#     spikes = (
#         spikes.join(shifts, how="cross")
#         .select(
#             pl.col("neuron"),
#             pl.col("shift"),
#             pl.col("time") + pl.col("shift"),
#         )
#         .sort("time")
#         .match_to_schema({"shift": pl.Float64, "neuron": pl.UInt32, "time": pl.Float64})
#     )

#     # Periodic extension of the reference to cover the range of spikes
#     time_lims = spikes.group_by("shift", "neuron").agg(
#         (pl.col("time").first() - REFRACTORY_PERIOD / 2).alias("time_min"),
#         (pl.col("time").last() + REFRACTORY_PERIOD / 2).alias("time_max"),
#     )
#     n_spikes_ref_per_neuron = spikes_ref.group_by("neuron").agg(
#         pl.len().alias("n_spikes_ref")
#     )
#     spikes_ref = (
#         spikes_ref.join(time_lims, on="neuron")
#         .with_columns(
#             rp.extend_periodically(
#                 pl.col("time"),
#                 pl.col("period"),
#                 pl.col("time_min"),
#                 pl.col("time_max"),
#             ).alias("time"),
#         )
#         .explode("time")
#         .sort("time")
#         .join(n_spikes_ref_per_neuron, on="neuron")
#         .select(
#             pl.col("shift"),
#             pl.col("neuron"),
#             pl.col("time").alias("time_ref"),
#             pl.arange(pl.len())
#             .mod(pl.col("n_spikes_ref"))
#             .over("neuron")
#             .alias("id_ref"),
#             # pl.col("n_spikes"),
#         )
#         # .match_to_schema(
#         #     {
#         #         "shift": pl.Float64,
#         #         "neuron": pl.UInt32,
#         #         "time": pl.Float64,
#         #         # "n_spikes": pl.UInt32,
#         #         "ref_id": pl.UInt32,
#         #     }
#         # )
#     )

#     precisions = (
#         n_spikes_per_neuron.drop_nulls("n_spikes")
#         .join(shifts, how="cross")
#         .join(
#             spikes_ref.join_asof(
#                 spikes,
#                 by=["shift", "neuron"],
#                 left_on="time_ref",
#                 right_on="time",
#                 strategy="nearest",
#                 tolerance=REFRACTORY_PERIOD / 2,
#                 check_sortedness=False,
#                 coalesce=False,
#             )
#             .with_columns(
#                 (pl.col("time") - pl.col("time_ref")).abs().alias("time_diff")
#             )
#             .sort("time_diff")
#             .unique(
#                 "id_ref", keep="first"
#             )  # remove periodic duplicates, keeping the closest
#             .group_by("shift", "neuron")
#             .agg(
#                 (1.0 - (2.0 * pl.col("time_diff") / REFRACTORY_PERIOD))
#                 .sum()
#                 .alias("sum_per_neuron")
#             ),
#             on=["shift", "neuron"],
#             how="left",
#         )
#         .group_by("shift")
#         .agg(
#             (1.0 - pl.col("sum_per_neuron").fill_null(0) / pl.col("n_spikes"))
#             .sum()
#             .alias("score")
#         )
#         .select(
#             pl.col("shift"),
#             (1.0 - pl.col("score") / n_neurons).alias("precision"),
#         )
#     )

#     if which == "all":
#         return precisions

#     return precisions.top_k(1, by="precision")


# def compute_recalls(spikes_ref, spikes, shifts, n_neurons, which=None):
#     """Compute similarity measures between shifted spike trains and the (periodic) reference.

#     Evaluates how well generated spike trains match reference periodic patterns
#     by computing similarity measures across different temporal shifts.
#     Uses nearest-neighbor matching with tolerance for comparing spike times.

#     Args:
#         spikes_ref (pl.DataFrame): Reference periodic spike patterns with columns
#             'neuron', 'time', 'period'.
#         spikes (pl.DataFrame): Generated spike trains with columns 'neuron', 'time'.
#         shifts (pl.DataFrame): Temporal shift values with column 'delta' for
#             alignment testing.
#         n_neurons (int): Total number of neurons in the network.
#         which (str, optional): Which score to return, 'best' or 'all'.

#     Returns:
#         tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
#             - precisions: DataFrame with precision metrics per shift
#             - recalls: DataFrame with recall metrics per shift

#     Notes:
#         - Precision measures how many generated spikes match reference spikes
#         - Recall measures how many reference spikes are matched by generated spikes
#         - Silent neurons (no spikes) contribute positively to both metrics
#         - Reference patterns are extended periodically for comparison
#         - Uses temporal tolerance (eps) for nearest-neighbor matching
#     """
#     if which is None:
#         which = "best"

#     neurons = pl.DataFrame({"neuron": np.arange(n_neurons, dtype=np.uint32)})

#     # Compute number of spikes per neuron in the reference
#     n_spikes_ref_per_neuron = (
#         spikes_ref.group_by("neuron")
#         .agg(pl.len().alias("n_spikes"))
#         .join(neurons, on="neuron", how="right")
#     )

#     spikes = (
#         spikes.join(shifts, how="cross")
#         .select(
#             pl.col("neuron"),
#             pl.col("shift"),
#             pl.col("time") + pl.col("shift"),
#         )
#         .sort("time")
#         .match_to_schema({"shift": pl.Float64, "neuron": pl.UInt32, "time": pl.Float64})
#     )

#     # Periodic extension of the reference to cover the range of spikes
#     time_lims = spikes.group_by("shift", "neuron").agg(
#         (pl.col("time").first() - REFRACTORY_PERIOD / 2).alias("time_min"),
#         (pl.col("time").last() + REFRACTORY_PERIOD / 2).alias("time_max"),
#     )
#     spikes_ref = (
#         spikes_ref.join(time_lims, on="neuron")
#         .with_columns(
#             rp.extend_periodically(
#                 pl.col("time"),
#                 pl.col("period"),
#                 pl.col("time_min"),
#                 pl.col("time_max"),
#             ).alias("time"),
#         )
#         .explode("time")
#         .join(n_spikes_ref_per_neuron, on="neuron")
#         .sort("time")
#         .select(
#             pl.col("shift"),
#             pl.col("neuron"),
#             pl.col("time").alias("time_ref"),
#             pl.arange(pl.len())
#             .mod(pl.col("n_spikes_ref"))
#             .over("neuron")
#             .alias("id_ref"),
#         )
#     )

# Number of spikes per neuron in the reference and number of active neurons in the reference
# n_spikes_ref_per_neuron = spikes_ref.group_by("shift", "neuron").agg(pl.len())
# n_silent_ref_neurons = n_spikes_ref_per_neuron.group_by("shift").agg(
#     (n_neurons - pl.n_unique("neuron")).alias("n_silent")
# )

# spikes_ref = spikes_ref.filter(pl.col("time") <= pl.col("time_max")).sort("time")

# recalls = (
#     spikes_ref.join_asof(
#         spikes,
#         by=["shift", "neuron"],
#         on="time",
#         strategy="nearest",
#         suffix="_ref",
#         tolerance=REFRACTORY_PERIOD / 2,
#         check_sortedness=False,
#         coalesce=False,
#     )
#     # .drop_nulls(["time", "time_ref"])  # useless?
#     .group_by("shift", "neuron")
#     .agg(
#         (1 - (2 * (pl.col("time") - pl.col("time_ref")).abs() / REFRACTORY_PERIOD))
#         .sum()
#         .alias("sum_per_neuron")
#     )
#     .join(n_spikes_ref_per_neuron, on="neuron", how="right")
#     .group_by("shift")
#     .agg(
#         (1 - pl.col("sum_per_neuron").fill_null(0) / pl.col("n_spikes_ref"))
#         .sum()
#         .alias("score")
#     )
#     .select(
#         pl.col("shift"),
#         (1 - pl.col("score") / n_neurons).alias("recall"),
#     )
# )

# recalls = (
#     n_spikes_ref_per_neuron.drop_nulls("n_spikes")
#     .join(shifts, how="cross")
#     .join(
#         spikes_ref.join_asof(
#             spikes,
#             by=["shift", "neuron"],
#             on="time",
#             strategy="nearest",
#             tolerance=REFRACTORY_PERIOD / 2,
#             check_sortedness=False,
#             coalesce=False,
#         )
#         .group_by("shift", "neuron")
#         .agg(
#             (
#                 1.0
#                 - (
#                     2.0
#                     * (pl.col("time_right") - pl.col("time")).abs()
#                     / REFRACTORY_PERIOD
#                 )
#             )
#             .sum()
#             .alias("sum_per_neuron")
#         ),
#         on=["shift", "neuron"],
#         how="left",
#     )
#     .group_by("shift")
#     .agg(
#         (1.0 - pl.col("sum_per_neuron").fill_null(0) / pl.col("n_spikes"))
#         .sum()
#         .alias("score")
#     )
#     .select(
#         pl.col("shift"),
#         (1.0 - pl.col("score") / n_neurons).alias("recall"),
#     )
# )

# if which == "all":
#     return recalls

# return recalls.top_k(1, by="recall")
