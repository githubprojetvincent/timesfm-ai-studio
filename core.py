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


def _ml_features(history: np.ndarray, window: int) -> np.ndarray:
    """Features causales uniquement : retards et statistiques des fenêtres passées."""
    y = np.asarray(history, dtype=float)
    lags = [1, 2, 3, 7, 14, 28]
    values = [float(y[-lag]) if len(y) >= lag else float(y[0]) for lag in lags]
    for width in (3, 7, 14, 28):
        sample = y[-min(width, len(y)):]
        values.extend([float(np.mean(sample)), float(np.std(sample))])
    values.append(float(len(y)))
    return np.asarray(values, dtype=float)


def _make_supervised(values: np.ndarray, min_lag: int = 14):
    X, target = [], []
    for end in range(min_lag, len(values)):
        X.append(_ml_features(values[:end], end))
        target.append(values[end])
    return np.asarray(X, dtype=float), np.asarray(target, dtype=float)


def _recursive_ml_predict(model, history: np.ndarray, horizon: int) -> np.ndarray:
    context = list(np.asarray(history, dtype=float))
    predictions = []
    for _ in range(horizon):
        x = _ml_features(np.asarray(context, dtype=float), len(context)).reshape(1, -1)
        value = float(np.asarray(model.predict(x)).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("Le modèle ML a produit une valeur non finie.")
        predictions.append(value)
        context.append(value)
    return np.asarray(predictions, dtype=float)


def machine_learning_forecast(values, horizon: int = 30, min_train_size: int | None = None):
    """Compare plusieurs modèles par validation chronologique puis réentraîne le meilleur."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(values, dtype=float)
    horizon = validate_horizon(horizon)
    if y.ndim != 1 or len(y) < 40 or not np.isfinite(y).all():
        raise ValueError("Le machine learning nécessite au moins 40 observations numériques finies.")
    min_train = min_train_size or max(30, int(len(y) * 0.55))
    min_train = min(min_train, len(y) - 8)
    eval_size = min(max(3, horizon), max(3, (len(y) - min_train) // 3), 14)
    split_points = sorted(set([min_train, min_train + max(1, (len(y)-min_train-eval_size)//2)]))
    candidates = {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=160, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=160, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=42),
    }
    scores = []
    for name, factory in candidates.items():
        actuals, forecasts = [], []
        for split in split_points:
            train = y[:split]
            stop = min(len(y), split + eval_size)
            if len(train) < 30 or stop <= split:
                continue
            X_train, y_train = _make_supervised(train)
            if len(X_train) < 10:
                continue
            model = factory()
            model.fit(X_train, y_train)
            pred = _recursive_ml_predict(model, train, stop - split)
            forecasts.extend(pred.tolist())
            actuals.extend(y[split:stop].tolist())
        if actuals:
            err = np.asarray(actuals) - np.asarray(forecasts)
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            scores.append({"model": name, "mae": mae, "rmse": rmse, "n": len(actuals)})
    if not scores:
        raise RuntimeError("Pas assez de fenêtres pour comparer les modèles ML.")
    scores.sort(key=lambda item: (item["mae"], item["rmse"]))
    best = scores[0]
    X_all, y_all = _make_supervised(y)
    final_model = candidates[best["model"]]()
    final_model.fit(X_all, y_all)
    predictions = _recursive_ml_predict(final_model, y, horizon)
    return {
        "forecast": predictions,
        "best_model": best["model"],
        "scores": scores,
        "mae": best["mae"],
        "rmse": best["rmse"],
        "validation_points": best["n"],
        "history_points": len(y),
        "horizon": horizon,
    }
