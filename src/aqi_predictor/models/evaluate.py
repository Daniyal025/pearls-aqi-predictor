"""Regression metrics used to compare models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # r2 is undefined for a single sample; guard it.
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}
