from __future__ import annotations
from collections.abc import Callable
import numpy as np

def validate_horizon(horizon: int) -> int:
    return max(1, min(512, int(horizon)))

def statistical_fallback(values: np.ndarray, horizon: int):
    window = min(len(values), max(10, min(60, len(values)//2)))
    y = values[-window:]
    x = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * np.arange(window, window + horizon) + intercept
    residual = y - (slope*x + intercept)
    spread = max(float(np.std(residual)), float(np.mean(np.abs(y))*0.03), 1e-6)
    radius = 1.28 * spread * np.sqrt(np.arange(1, horizon+1))
    return pred, np.column_stack((pred-radius, pred+radius))

def forecast_series(values, horizon: int, model_loader: Callable | None = None):
    values = np.asarray(values, dtype=np.float32)
    horizon = validate_horizon(horizon)
    if values.ndim != 1 or len(values) < 10 or not np.isfinite(values).all():
        raise ValueError("La série doit contenir au moins 10 valeurs numériques finies.")
    try:
        if model_loader is None:
            import torch, timesfm
            torch.set_float32_matmul_precision("high")
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=512, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=False, fix_quantile_crossing=True))
        else:
            model = model_loader()
        raw, quantiles = model.forecast(horizon=horizon, inputs=[values])
        pred = np.asarray(raw[0], dtype=float)[:horizon]
        if len(pred) != horizon or not np.isfinite(pred).all():
            raise ValueError("Le modèle a renvoyé des valeurs de prévision invalides.")
        bounds = None
        if quantiles is not None:
            q = np.asarray(quantiles)
            if q.ndim == 3 and q.shape[1] >= horizon and q.shape[2] >= 9:
                q = q[0, :horizon, :]
                bounds = np.column_stack((q[:, 1], q[:, -1]))
        return pred, bounds, "TimesFM 2.5 — modèle officiel"
    except Exception as exc:
        pred, bounds = statistical_fallback(values, horizon)
        return pred, bounds, f"Mode statistique de secours — {type(exc).__name__}: {str(exc)[:120]}"
