"""Tests for the sklearn-compatible MOMENT adapters.

The tests focus on the public adapter contract (shape coercion,
sklearn estimator API conformance, error messages).  The actual MOMENT
forward pass is exercised by an integration test that requires the
HuggingFace weights to be downloaded; that test is marked to skip by
default so CI does not need network access at test time.
"""
from __future__ import annotations

import numpy as np
import pytest

from momentfm.sklearn_adapter import (
    MOMENTAnomalyDetector,
    MOMENTEmbedder,
    MOMENTForecaster,
    _to_3d,
)


def test_to_3d_passthrough():
    X = np.zeros((4, 1, 512))
    out = _to_3d(X, n_channels=1, context_length=512)
    assert out.shape == (4, 1, 512)


def test_to_3d_2d_to_3d():
    X = np.arange(4 * 512).reshape(4, 512)
    out = _to_3d(X, n_channels=1, context_length=512)
    assert out.shape == (4, 1, 512)
    np.testing.assert_array_equal(out[0, 0], X[0])


def test_to_3d_multichannel_2d():
    X = np.arange(4 * 3 * 512).reshape(4, 3 * 512)
    out = _to_3d(X, n_channels=3, context_length=512)
    assert out.shape == (4, 3, 512)


def test_to_3d_rejects_4d():
    X = np.zeros((2, 2, 2, 512))
    with pytest.raises(ValueError, match="ndim"):
        _to_3d(X, n_channels=1, context_length=512)


def test_to_3d_rejects_wrong_2d_width():
    X = np.zeros((4, 500))
    with pytest.raises(ValueError, match="2D input"):
        _to_3d(X, n_channels=1, context_length=512)


def test_to_3d_rejects_wrong_3d_shape():
    X = np.zeros((4, 2, 512))
    with pytest.raises(ValueError, match="3D input"):
        _to_3d(X, n_channels=1, context_length=512)


@pytest.mark.parametrize(
    "estimator_cls",
    [MOMENTAnomalyDetector, MOMENTForecaster, MOMENTEmbedder],
)
def test_default_params(estimator_cls):
    """Adapters expose the standard sklearn estimator interface."""
    est = estimator_cls()
    params = est.get_params()
    assert params["model_name"] == "AutonLab/MOMENT-1-large"
    assert params["context_length"] == 512
    assert params["n_channels"] == 1
    assert params["batch_size"] == 64
    assert params["device"] == "auto"


@pytest.mark.parametrize(
    "estimator_cls",
    [MOMENTAnomalyDetector, MOMENTForecaster, MOMENTEmbedder],
)
def test_set_params(estimator_cls):
    est = estimator_cls()
    est.set_params(context_length=256, n_channels=2, batch_size=8)
    assert est.context_length == 256
    assert est.n_channels == 2
    assert est.batch_size == 8


def test_forecaster_requires_horizon():
    """Forecasting head needs ``forecast_horizon`` set on the estimator."""
    est = MOMENTForecaster()
    with pytest.raises(ValueError, match="forecast_horizon"):
        est.fit(np.zeros((2, 512)))


# ---------------------------------------------------------------------
# Integration tests — require HuggingFace weights (skipped by default)
# ---------------------------------------------------------------------

@pytest.mark.skipif(
    True,
    reason="Integration test — downloads MOMENT-1-small (~100 MB) "
    "and runs a forward pass.  Flip the skipif to False to enable.",
)
def test_anomaly_detector_smoke():
    """End-to-end smoke for the anomaly detector with real weights."""
    est = MOMENTAnomalyDetector(model_name="AutonLab/MOMENT-1-small")
    est.fit()
    X = np.random.RandomState(0).randn(3, 512).astype(np.float32)
    scores = est.transform(X)
    assert scores.shape == (3,)
    assert (scores >= 0).all()


@pytest.mark.skipif(
    True,
    reason="Integration test — downloads MOMENT-1-small (~100 MB).",
)
def test_anomaly_detector_3d_input_smoke():
    est = MOMENTAnomalyDetector(model_name="AutonLab/MOMENT-1-small")
    est.fit()
    X = np.random.RandomState(0).randn(3, 1, 512).astype(np.float32)
    scores = est.transform(X)
    assert scores.shape == (3,)


@pytest.mark.skipif(
    True,
    reason="Integration test — downloads MOMENT-1-large + 96-h horizon head.",
)
def test_forecaster_smoke():
    est = MOMENTForecaster(forecast_horizon=96)
    est.fit()
    X = np.random.RandomState(0).randn(2, 512).astype(np.float32)
    y_hat = est.predict(X)
    assert y_hat.shape == (2, 96)


@pytest.mark.skipif(
    True,
    reason="Integration test — downloads MOMENT-1-small (~100 MB).",
)
def test_embedder_smoke():
    est = MOMENTEmbedder(model_name="AutonLab/MOMENT-1-small")
    est.fit()
    X = np.random.RandomState(0).randn(2, 512).astype(np.float32)
    emb = est.transform(X)
    assert emb.ndim == 2
    assert emb.shape[0] == 2


@pytest.mark.skipif(
    True,
    reason="Integration test — exercises sklearn Pipeline wiring with "
    "downloaded weights.",
)
def test_sklearn_pipeline_integration():
    """The adapter is sklearn-Pipeline-compatible."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("embed", MOMENTEmbedder(model_name="AutonLab/MOMENT-1-small")),
        ("clf", LogisticRegression(max_iter=100)),
    ])
    X = np.random.RandomState(0).randn(8, 512).astype(np.float32)
    y = np.array([0, 1] * 4)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (8,)
