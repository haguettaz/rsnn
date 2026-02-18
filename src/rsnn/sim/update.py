import polars as pl

import rsnn_plugin as rp
from rsnn.optim.nrg import *

# def get_full_nrg_1st(syn_states: pl.DataFrame, last_only: bool):
#     """
#     Args:
#         syn_states (pl.DataFrame): with column 'end' and 'f_index'
#         which (_type_, optional): _description_. Defaults to None.

#     Raises:
#         ValueError: _description_
#     """

#     if last_only:
#         syn_states = syn_states.group_by(["neuron", "end", "in_index"]).agg(
#             pl.max("start").alias("start")
#         )

#     return (
#         syn_states.join(syn_states, on=["neuron", "end"])
#         .group_by(["neuron", "in_index", "in_index_right"])
#         .agg(
#             rp.inner_syn_1st(pl.col("start"), pl.col("start_right"), pl.col("end"))
#             .sum()
#             .alias("energy")
#         )
#     )


# def get_full_nrg_1st(syn_states: pl.DataFrame, last_only: bool) -> pl.DataFrame:

#     if last_only:
#         syn_states = syn_states.group_by(["f_index", "end", "in_index"]).agg(
#             pl.max("start").alias("start")
#         )

#     return (
#         syn_states.join(syn_states, on=["f_index", "end"])
#         .group_by(["f_index", "in_index", "in_index_right"])
#         .agg(
#             (
#                 rp.inner_syn_1st(
#                     pl.col("start"), pl.col("start_right"), pl.col("end")
#                 ).sum()
#             ).alias("energy")
#         )
#     )

# return (
#     syn_states.join(syn_states, on=["neuron", "index", "end"])
#     .group_by("neuron", "index", "end")
#     .agg(
#         (
#             (
#                 (-(pl.col("start") - pl.col("start_right")).abs()).exp()
#                 - (
#                     -(2 * pl.col("end") - pl.col("start") - pl.col("start_right"))
#                 ).exp()
#             ).sum()
#             + l2_reg
#         ).alias("energy")
#     )
# )


def get_syn_energies(syn_states, full, first_order, last_only):
    if full:
        if first_order:
            syn_energies = get_full_nrg_1st(syn_states, last_only=last_only)
        else:
            syn_energies = get_full_nrg_2nd(syn_states, last_only=last_only)

    else:
        if first_order:
            syn_energies = get_diag_nrg_1st(syn_states, last_only=last_only)
        else:
            syn_energies = get_diag_nrg_2nd(syn_states, last_only=last_only)

    return syn_energies


def get_norm_2nd_syn_signals(syn_states: pl.DataFrame) -> pl.DataFrame:
    """_summary_

    Args:
        syn_states (_type_): _description_

    Returns:
        pl.DataFrame: with columns 'in_index', 'f_index', and 'signal'

    Warning:
        - `syn_states` must not contain entries with `end` < `start`.
        - `syn_states` contains only one `end` value per `neuron`.

    """
    syn_signals = (
        syn_states.with_columns((pl.col("end") - pl.col("start")).alias("delta"))
        .group_by(["f_index", "in_index"])
        .agg((pl.col("delta") * (-pl.col("delta")).exp()).sum().alias("signal"))
    )

    return syn_signals.join(
        syn_signals.group_by("f_index").agg(
            pl.col("signal").pow(2).sum().sqrt().alias("norm")
        ),
        on="f_index",
    ).select(
        pl.col("f_index"),
        pl.col("in_index"),
        pl.col("signal") / pl.col("norm").alias("signal"),
    )


def update_weights(
    spikes: pl.DataFrame,
    synapses: pl.DataFrame,
    syn_states: pl.DataFrame,
    alpha: float,
    l2_reg: float,
    full: bool,
    first_order: bool,
    last_only: bool,
) -> pl.DataFrame:
    """_summary_

    Args:
        spikes (pl.DataFrame): _description_
        synapses (pl.DataFrame): with columns 'target', 'in_index', and 'weight'
        syn_states (pl.DataFrame): _description_
        alpha (float): _description_

    Returns:
        pl.DataFrame: _description_

    Warning:
        - `spikes` must contain at most one spike per 'neuron' group.
        - `syn_states` must be sorted by the 'start' key (within each 'neuron' group).
    """
    # select only the synapses of firing neurons (based on spikes)
    new_synapses = synapses.join(
        spikes, left_on="target", right_on="neuron", how="semi"
    ).select(pl.col("in_index"), pl.col("weight"))

    # syn_states = spikes.join(syn_states, on="neuron").select(
    #     pl.col("neuron"),
    #     pl.col("index"),
    #     pl.col("start"),
    #     pl.col("time").alias("end"),
    # )

    syn_states = (
        syn_states.join_asof(
            spikes,
            by="neuron",
            left_on="start",
            right_on="time",
            strategy="forward",
            coalesce=False,
            check_sortedness=False,
        )
        .drop_nulls()
        .select(
            pl.col("neuron").alias("f_index"),
            pl.col("in_index"),
            pl.col("start"),
            pl.col("time").alias("end"),
        )
    )

    syn_signals = get_norm_2nd_syn_signals(syn_states)
    syn_signals = syn_signals.join(new_synapses, on="in_index")

    # logger.debug(f"Synaptic states: {syn_states.filter(pl.col('index')==46855)}")
    syn_energies = get_syn_energies(syn_states, full, first_order, last_only)

    agg_syn_energies = (
        syn_signals.join(syn_energies, on="in_index")
        .group_by(["f_index", "in_index_right"])
        .agg((pl.col("energy") * pl.col("weight")).sum().alias("weighted_energy"))
        .select(
            pl.col("f_index"),
            pl.col("in_index_right").alias("in_index"),
            pl.col("weighted_energy"),
        )
    )
    syn_signals = syn_signals.join(agg_syn_energies, on="in_index").select(
        pl.col("f_index"),
        pl.col("in_index"),
        pl.col("weight"),
        pl.col("signal"),
        pl.col("weighted_energy") + l2_reg * pl.col("weight"),
    )
    syn_signals = syn_signals.join(
        syn_signals.group_by("f_index").agg(
            (pl.col("weighted_energy") * pl.col("signal"))
            .sum()
            .alias("proj_weighted_energy")
        ),
        on="f_index",
    )

    new_synapses = syn_signals.select(
        pl.col("in_index"),
        pl.col("weight")
        + alpha
        * (
            pl.col("proj_weighted_energy") * pl.col("signal")
            - pl.col("weighted_energy")
        ),
    )

    # logger.debug(f"Synaptic energies: {syn_energies.filter(pl.col('index')==46855)}")

    # new_synapses = (
    #     weighted_proj_syn_energies.join(syn_signals, on="in_index")
    #     .join(weighted_syn_energies, on="in_index")
    #     .with_columns(
    #         pl.col("weight")
    #         + alpha
    #         * (
    #             pl.col("proj_weighted_energy") * pl.col("signal")
    #             - pl.col("weighted_energy")
    #         )
    #     )
    # )

    # new_synapses = new_synapses.join(syn_signals, on=["neuron", "index"]).join(
    #     syn_energies, on=["neuron", "index"]
    # )
    # new_synapses = (
    #     new_synapses.group_by("neuron")
    #     .agg(
    #         (pl.col("weight") * pl.col("signal") * pl.col("energy"))
    #         .sum()
    #         .alias("weighted_energy")
    #     )
    #     .join(new_synapses, on="neuron")
    #     .with_columns(
    #         (
    #             pl.col("weight")
    #             + alpha
    #             * (
    #                 pl.col("weighted_energy") * pl.col("signal") / pl.col("norm")
    #                 - pl.col("energy") * pl.col("weight")
    #             )
    #         )
    #     )
    # )

    return synapses.update(new_synapses, on="in_index")
