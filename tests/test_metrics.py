from rsnn.metrics import *


def test_similarity_metric():
    similarity_metric = SimilarityMetric(
        period=8.0,
        r_f_times=[np.array([1.0, 3.0, 5.0])],
    )

    precision, recall = similarity_metric.measure([np.array([0.0, 2.0, 4.0])])
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 1.0)

    precision, recall = similarity_metric.measure([np.array([0.0, 2.0])])
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 2.0 / 3.0)

    precision, recall = similarity_metric.measure([np.array([3.0, 5.0, 7.0, 1.0])])
    assert np.isclose(precision, 3.0 / 4.0)
    assert np.isclose(recall, 1.0)

    precision, recall = similarity_metric.measure([np.array([0.2, 1.9, 4.1])])
    assert np.isclose(precision, 0.8)
    assert np.isclose(recall, 0.8)

    precision, recall = similarity_metric.measure([np.array([])])
    assert np.isclose(precision, 0.0)
    assert np.isclose(recall, 0.0)

    precision, recall = similarity_metric.measure([np.array([1.0, 1.5, 5.0])])
    assert np.isnan(precision)
    assert np.isnan(recall)
