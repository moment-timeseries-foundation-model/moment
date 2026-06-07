"""sklearn-compatible adapters for MOMENT.

These wrappers let MOMENT be dropped into ``sklearn.pipeline.Pipeline``
and any other tooling (ColumnTransformer, GridSearchCV, etc.) that
follows the sklearn estimator API.

Three adapters are provided:

- :class:`MOMENTAnomalyDetector` — returns reconstruction MSE as an
  anomaly score (compatible with ``OutlierMixin``-style usage).
- :class:`MOMENTForecaster` — returns the forecast horizon as the
  prediction (``RegressorMixin``).
- :class:`MOMENTEmbedder` — returns the encoder embedding as a feature
  matrix (``TransformerMixin``).

All three accept either 2D ``(n_samples, n_channels * context_length)``
or 3D ``(n_samples, n_channels, context_length)`` input.  The 2D path
makes pipeline integration possible (sklearn pipelines require 2D X);
the 3D path matches MOMENT's native shape.

``fit`` is a no-op by default (zero-shot use).  Pre-loaded weights are
preserved across pipeline operations because ``fit`` does not reload
the model.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

try:
    import torch
    from sklearn.base import (
        BaseEstimator,
        RegressorMixin,
        TransformerMixin,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "momentfm.sklearn_adapter requires torch and scikit-learn. "
        "Install them via `pip install torch scikit-learn`."
    ) from exc

from momentfm.models.moment import MOMENTPipeline
from momentfm.utils.utils import control_randomness


# ---------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------

def _to_3d(
    X: np.ndarray, n_channels: int, context_length: int
) -> np.ndarray:
    """Coerce a 2D or 3D array into MOMENT's expected 3D shape
    ``(n_samples, n_channels, context_length)``.

    Raises ``ValueError`` if the input cannot be reshaped without
    losing or fabricating data.
    """
    X = np.asarray(X)
    if X.ndim == 3:
        if X.shape[1:] != (n_channels, context_length):
            raise ValueError(
                f"3D input has shape {X.shape}; expected "
                f"(_, {n_channels}, {context_length})."
            )
        return X
    if X.ndim == 2:
        expected = n_channels * context_length
        if X.shape[1] != expected:
            raise ValueError(
                f"2D input has shape {X.shape}; expected "
                f"(_, {expected}) = (_, n_channels * context_length)."
            )
        return X.reshape(X.shape[0], n_channels, context_length)
    raise ValueError(
        f"Input must be 2D or 3D; got ndim={X.ndim}."
    )


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------
# Base estimator
# ---------------------------------------------------------------------

class _MOMENTBaseEstimator(BaseEstimator):
    """Shared infrastructure for the three task-specific adapters."""

    _task_name: str = "reconstruction"

    def __init__(
        self,
        model_name: str = "AutonLab/MOMENT-1-large",
        context_length: int = 512,
        n_channels: int = 1,
        batch_size: int = 64,
        device: str = "auto",
        random_state: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.forecast_horizon = forecast_horizon
        self._pipeline = None
        self._resolved_device = None

    # sklearn convention: load the model in `fit` so estimators that
    # serialise unfitted lazily (joblib/grid-search) stay light.
    def fit(self, X=None, y=None):  # noqa: D401, ARG002
        """Load the MOMENT pipeline.  Zero-shot — does not train."""
        if self.random_state is not None:
            control_randomness(self.random_state)
        model_kwargs: dict = {"task_name": self._task_name}
        if self._task_name == "forecasting":
            if self.forecast_horizon is None:
                raise ValueError(
                    "forecast_horizon must be set on the estimator "
                    "before calling fit() with task=forecasting."
                )
            model_kwargs["forecast_horizon"] = int(self.forecast_horizon)
        pipeline = MOMENTPipeline.from_pretrained(
            self.model_name, model_kwargs=model_kwargs
        )
        pipeline.init()
        device = _resolve_device(self.device)
        pipeline.to(device)
        pipeline.eval()
        self._pipeline = pipeline
        self._resolved_device = device
        return self

    def _ensure_pipeline(self) -> MOMENTPipeline:
        if self._pipeline is None:
            self.fit()
        return self._pipeline

    def _forward(self, X: np.ndarray):
        """Run MOMENT forward pass in batches, return the raw output
        list (one element per batch)."""
        pipeline = self._ensure_pipeline()
        X3 = _to_3d(X, self.n_channels, self.context_length)
        device = self._resolved_device or "cpu"
        outputs: list = []
        with torch.no_grad():
            for start in range(0, X3.shape[0], self.batch_size):
                batch = X3[start : start + self.batch_size]
                x_enc = torch.tensor(
                    batch, dtype=torch.float32, device=device
                )
                outputs.append(pipeline(x_enc=x_enc))
        return outputs


# ---------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------

class MOMENTAnomalyDetector(_MOMENTBaseEstimator, TransformerMixin):
    """sklearn-compatible MOMENT anomaly detector.

    Uses reconstruction MSE as the anomaly score: high score → high
    reconstruction error → anomalous.  ``transform`` returns per-sample
    mean MSE (shape ``(n_samples,)``).  Use ``score_samples`` for the
    sklearn outlier-detector convention (higher = more normal).
    """

    _task_name = "reconstruction"

    def transform(self, X) -> np.ndarray:
        """Return per-sample mean reconstruction MSE."""
        outputs = self._forward(X)
        scores: list[np.ndarray] = []
        for batch_out in outputs:
            recon = getattr(batch_out, "reconstruction", None)
            if recon is None:
                raise RuntimeError(
                    "MOMENT output has no `reconstruction` field; "
                    "did you instantiate with task_name='reconstruction'?"
                )
            # Reconstruct against the SAME input that produced it.
            input_tensor = recon.detach()
            actual_tensor = self._last_x_enc
            mse = (input_tensor - actual_tensor) ** 2
            # mean over channel + timestep → (n_samples,)
            scores.append(mse.mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(scores)

    def _forward(self, X: np.ndarray):
        pipeline = self._ensure_pipeline()
        X3 = _to_3d(X, self.n_channels, self.context_length)
        device = self._resolved_device or "cpu"
        outputs: list = []
        # Anomaly mode needs both the forward output AND the input that
        # produced it, so we stash the input alongside the output.
        with torch.no_grad():
            for start in range(0, X3.shape[0], self.batch_size):
                batch = X3[start : start + self.batch_size]
                x_enc = torch.tensor(
                    batch, dtype=torch.float32, device=device
                )
                self._last_x_enc = x_enc  # used inside transform()
                outputs.append(pipeline(x_enc=x_enc))
        return outputs

    def score_samples(self, X) -> np.ndarray:
        """sklearn outlier-detector convention: higher = more normal."""
        return -self.transform(X)


# ---------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------

class MOMENTForecaster(_MOMENTBaseEstimator, RegressorMixin):
    """sklearn-compatible MOMENT forecaster.

    ``predict`` returns the forecast horizon as a flat 2D array of
    shape ``(n_samples, n_channels * forecast_horizon)`` so it slots
    into sklearn pipelines.  The 3D reshape is straightforward:
    ``y_3d = y.reshape(n_samples, n_channels, forecast_horizon)``.
    """

    _task_name = "forecasting"

    def predict(self, X) -> np.ndarray:
        outputs = self._forward(X)
        chunks: list[np.ndarray] = []
        for batch_out in outputs:
            forecast = getattr(batch_out, "forecast", None)
            if forecast is None:
                raise RuntimeError(
                    "MOMENT output has no `forecast` field; ensure "
                    "task_name='forecasting' and that the pipeline "
                    "is reload after changing horizon."
                )
            np_out = forecast.detach().cpu().numpy()
            # (batch, channels, horizon) → (batch, channels*horizon)
            chunks.append(np_out.reshape(np_out.shape[0], -1))
        return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------

class MOMENTEmbedder(_MOMENTBaseEstimator, TransformerMixin):
    """sklearn-compatible MOMENT embedder.

    ``transform`` returns the encoder embeddings as a 2D feature matrix
    of shape ``(n_samples, embedding_dim)`` so downstream sklearn
    estimators (LogisticRegression, KMeans, t-SNE, etc.) can consume
    them directly.
    """

    _task_name = "embedding"

    def transform(self, X) -> np.ndarray:
        outputs = self._forward(X)
        chunks: list[np.ndarray] = []
        for batch_out in outputs:
            emb = getattr(batch_out, "embeddings", None)
            if emb is None:
                raise RuntimeError(
                    "MOMENT output has no `embeddings` field; ensure "
                    "task_name='embedding'."
                )
            np_out = emb.detach().cpu().numpy()
            chunks.append(np_out.reshape(np_out.shape[0], -1))
        return np.concatenate(chunks, axis=0)


__all__ = [
    "MOMENTAnomalyDetector",
    "MOMENTForecaster",
    "MOMENTEmbedder",
]
