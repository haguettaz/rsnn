# import numpy as np
# import polars as pl
# import polars.selectors as cs

# import rsnn_plugin as rp

# from .constants import REFRACTORY_RESET

# # @guvectorize(
# #     [(float64[:], float64[:], float64[:], float64[:], float64[:])],
# #     "(n),(n),(n),(n)->(n)",
# # )
# # def recursively_update_c0(length, weight_0, coef_1, coef_0, res):
# #     res[0] = coef_0[0]
# #     for n in range(1, len(length)):
# #         res[n] = weight_0[n] + (res[n - 1] + coef_1[n - 1] * length[n - 1]) * np.exp(
# #             -length[n - 1]
# #         )


# # @guvectorize([(float64[:], float64[:], float64[:], float64[:])], "(n),(n),(n)->(n)")
# # def recursively_update_c1(length, weight_1, coef_1, res):
# #     res[0] = coef_1[0]
# #     for n in range(1, len(length)):
# #         res[n] = weight_1[n] + res[n - 1] * np.exp(-length[n - 1])


# # def update_coef(states, over=None):
# #     # Warning: states should be sorted by start (according to the grouping given by over) before being passed to this function

# #     states = states.with_columns(
# #         pl.struct(["length", "weight_1", "coef_1"])
# #         .map_batches(
# #             lambda batch: recursively_update_c1(
# #                 batch.struct.field("length"),
# #                 batch.struct.field("weight_1"),
# #                 batch.struct.field("coef_1"),
# #             ),  # type: ignore
# #             return_dtype=pl.Float64,
# #         )
# #         .over(over)
# #         .alias("coef_1")
# #     )
# #     states = states.with_columns(
# #         pl.struct(["length", "weight_0", "coef_0", "coef_1"])
# #         .map_batches(
# #             lambda batch: recursively_update_c0(
# #                 batch.struct.field("length"),
# #                 batch.struct.field("weight_0"),
# #                 batch.struct.field("coef_1"),
# #                 batch.struct.field("coef_0"),
# #             ),  # type: ignore
# #             return_dtype=pl.Float64,
# #         )
# #         .over(over)
# #         .alias("coef_0")
# #     )

# #     return states


# # def update_length(states, over=None, fill_value=None):
# #     # Warning: states should be sorted by start
# #     states = states.with_columns(
# #         pl.col("start")
# #         .diff()
# #         .shift(-1, fill_value=fill_value)
# #         .over(over)
# #         .alias("length"),
# #     )
# #     return states


# # def scan_states(states, over, final_length):
# #     states = states.sort("start")
# #     states = states.with_columns(
# #         length=pl.col("start").diff().shift(-1, fill_value=final_length).over(over)
# #     )
# #     states = states.with_columns(
# #         coef_1=rp.scan_coef_1(pl.col("length").shift(), pl.col("weight_1")).over(over)
# #     )
# #     states = states.with_columns(
# #         coef_0=rp.scan_coef_0(
# #             pl.col("length").shift(), pl.col("coef_1").shift(), pl.col("weight_0")
# #         ).over(over)
# #     )
# #     return states


# def compute_max_violations(states, vmin, k=1, by=None, right_lim=1e-9):
#     states = states.with_columns(
#         (
#             pl.lit(0.0, pl.Float64).alias("left_delta"),
#             (1 - pl.col("coef_0") / pl.col("coef_1"))
#             .clip(0.0, (pl.col("length") - right_lim).clip(0.0))
#             .alias("crit_delta"),
#             (pl.col("length") - right_lim).clip(0.0).alias("right_delta"),
#         )
#     )
#     states = states.unpivot(
#         index=cs.exclude(cs.ends_with("_delta")), value_name="delta"
#     ).drop("variable")
#     states = states.with_columns(
#         (
#             (pl.col("coef_0") + pl.col("coef_1") * pl.col("delta"))
#             * (-pl.col("delta")).exp()
#         )
#         .fill_nan(0.0)  # Fading memory -> 0 at infinity
#         .alias("value")
#     )

#     states = states.filter(pl.col("value") > vmin)
#     if by is None:
#         max_violations = states.top_k(k=k, by="value")
#     else:
#         max_violations = (
#             states.group_by(by).agg(pl.all().top_k_by("value", k=k)).explode(cs.list())
#         )

#     max_violations = max_violations.with_columns(
#         (pl.col("start") + pl.col("delta")).alias("time")
#     )
#     return max_violations


# # def compute_rising_crossing_times(f_thresh, start, length, coef_0, coef_1):
# #     dt = np.where(
# #         coef_0 < f_thresh,
# #         np.where(
# #             coef_1 > 0.0,
# #             -(lambertw(-f_thresh / coef_1 * np.exp(-coef_0 / coef_1), 0)) - coef_0 / coef_1,
# #             np.where(
# #                 coef_1 < 0.0,
# #                 -(lambertw(-f_thresh / coef_1 * np.exp(-coef_0 / coef_1), -1)) - coef_0 / coef_1,
# #                 np.log(coef_0 / f_thresh),
# #             ),
# #         ),
# #         0.0,
# #     )
# #     dt[(dt < 0.0) | (dt >= length) | np.iscomplex(dt)] = np.nan
# #     return start + np.real(dt)


# # @guvectorize([(float64[:], float64[:], float64[:], float64[:])], "(n),(n),(n)->(n)")
# # def recursively_update_c0(length, coef_1, weight_0, coef_0):
# #     coef_0[0] = weight_0[0]
# #     for n in range(1, len(length)):
# #         coef_0[n] = weight_0[n] + (coef_0[n - 1] + coef_1[n - 1] * length[n - 1]) * np.exp(
# #             -length[n - 1]
# #         )


# # @guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->(n)")
# # def recursively_update_c1(length, weight_1, coef_1):
# #     coef_1[0] = weight_1[0]
# #     for n in range(1, len(length)):
# #         coef_1[n] = weight_1[n] + coef_1[n - 1] * np.exp(-length[n - 1])
