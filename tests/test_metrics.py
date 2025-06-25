from rsnn.metrics import *


def test_score():
    assert False


def test_similarity_metric():
    similarity_metric = SimilarityMetric(
        period=10.0,
        r_f_times=[np.array([1.0, 3.0, 5.0])],
    )

    precision, recall = similarity_metric.measure([np.array([0.0, 2.0, 8.0])])
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 1.0)

    precision, recall = similarity_metric.measure([np.array([0.2, 1.9, 8.4])])
    assert np.isclose(precision, 0.6666666666666663)
    assert np.isclose(recall, 0.6666666666666663)

    precision, recall = similarity_metric.measure([np.array([3.0, 5.0, 7.0, 9.0])])
    assert np.isclose(precision, 0.75)
    assert np.isclose(recall, 1.0)

    precision, recall = similarity_metric.measure([np.array([])])
    assert np.isclose(precision, 0.0)
    assert np.isclose(recall, 0.0)
