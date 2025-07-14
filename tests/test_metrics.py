from rsnn.metrics import *


def test_similarity_metric():
    precision, adjustment = compute_best_precision(
        [np.array([1.0, 3.0, 5.0])], [np.array([0.0, 2.0, 4.0])], 1
    )
    assert np.isclose(precision, 1.0)
    assert np.isclose(adjustment, 1.0)
    recall, adjustment = compute_best_recall(
        [np.array([1.0, 3.0, 5.0])], [np.array([0.0, 2.0, 4.0])], 1
    )
    assert np.isclose(recall, 1.0)
    assert np.isclose(adjustment, 1.0)

    precision, adjustment = compute_best_precision(
        [np.array([1.0, 3.0, 5.0])], [np.array([100.0, 102.0, 104.0])], 1
    )
    assert np.isclose(precision, 1.0)
    assert np.isclose(adjustment, -99.0)
    recall, adjustment = compute_best_recall(
        [np.array([1.0, 3.0, 5.0])], [np.array([100.0, 102.0, 104.0])], 1
    )
    assert np.isclose(recall, 1.0)
    assert np.isclose(adjustment, -99.0)

    precision, adjustment = compute_best_precision(
        [np.array([1.0, 3.0, 5.0])], [np.array([0.0, 2.0])], 1
    )
    assert np.isclose(precision, 1.0)
    assert np.isclose(adjustment, 1.0)
    recall, adjustment = compute_best_recall(
        [np.array([1.0, 3.0, 5.0])], [np.array([0.0, 2.0])], 1
    )
    assert np.isclose(recall, 2.0 / 3.0)
    assert np.isclose(adjustment, 1.0)

    precision, adjustment = compute_best_precision(
        [np.array([1.0, 3.0, 5.0])], [np.array([3.0, 5.0, 7.0, 10.0])], 1
    )
    assert np.isclose(precision, 3.0 / 4.0)
    assert np.isclose(adjustment, -2.0)
    recall, adjustment = compute_best_recall(
        [np.array([1.0, 3.0, 5.0])], [np.array([3.0, 5.0, 7.0, 10.0])], 1
    )
    assert np.isclose(recall, 1.0)
    assert np.isclose(adjustment, -2.0)


def test_similarity_metric_periodic():
    precision, adjustment = compute_best_precision(
        [np.array([1.0, 3.0, 5.0])], [np.array([0.0, 2.0, 4.0])], 1, 10.0
    )
    assert np.isclose(precision, 1.0)
    assert np.isclose(adjustment, 1.0)
    recall, adjustment = compute_best_recall(
        [np.array([1.0, 3.0, 5.0])], [np.array([0.0, 2.0, 4.0])], 1, 10.0
    )
    assert np.isclose(recall, 1.0)
    assert np.isclose(adjustment, 1.0)

    precision, adjustment = compute_best_precision(
        [np.array([1.0, 3.0, 5.0])], [np.array([100.0, 102.0, 104.0])], 1, 10.0
    )
    assert np.isclose(precision, 1.0)
    assert np.isclose(adjustment, 1.0)
    recall, adjustment = compute_best_recall(
        [np.array([1.0, 3.0, 5.0])], [np.array([100.0, 102.0, 104.0])], 1, 10.0
    )
    assert np.isclose(recall, 1.0)
    assert np.isclose(adjustment, 1.0)

    precision, adjustment = compute_best_precision(
        [np.array([1.0, 3.0, 5.0])], [np.array([0.0, 2.0])], 1, 10.0
    )
    assert np.isclose(precision, 1.0)
    assert np.isclose(adjustment, 1.0)
    recall, adjustment = compute_best_recall(
        [np.array([1.0, 3.0, 5.0])], [np.array([0.0, 2.0])], 1, 10.0
    )
    assert np.isclose(recall, 2.0 / 3.0)
    assert np.isclose(adjustment, 1.0)

    precision, adjustment = compute_best_precision(
        [np.array([1.0, 3.0, 5.0])], [np.array([3.0, 5.0, 7.0, 10.0])], 1, 10.0
    )
    assert np.isclose(precision, 3.0 / 4.0)
    assert np.isclose(adjustment, 8.0)
    recall, adjustment = compute_best_recall(
        [np.array([1.0, 3.0, 5.0])], [np.array([3.0, 5.0, 7.0, 10.0])], 1, 10.0
    )
    assert np.isclose(recall, 1.0)
    assert np.isclose(adjustment, 8.0)
