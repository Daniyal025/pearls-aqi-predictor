"""
SHAP explainability. Produces feature-importance values for tree models and
saves a bar chart to artifacts/shap/.

TODO (advanced): cache explainers, support KernelExplainer for the TF model,
and store SHAP arrays in MongoDB for the dashboard instead of reading PNGs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aqi_predictor.logging_config import get_logger
from aqi_predictor.models.registry import get_active_model

logger = get_logger(__name__)

SHAP_DIR = Path("artifacts/shap")


def explain_horizon(X_background: np.ndarray, horizon: str = "24h") -> dict[str, Any]:
    """Compute mean |SHAP| per feature for the active model of a horizon."""
    import shap  # imported lazily; shap is heavy

    meta = get_active_model(horizon)
    if meta is None:
        raise RuntimeError(f"No active model for horizon {horizon}.")
    if not meta.get("artifact_path", "").endswith(".joblib"):
        raise RuntimeError("SHAP explain currently supports sklearn (joblib) models.")

    bundle = joblib.load(meta["artifact_path"])
    model, feats = bundle["model"], bundle["features"]

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X_background)

    values = explainer.shap_values(X_background)
    importance = np.abs(np.asarray(values)).mean(axis=0)
    ranked = sorted(zip(feats, importance.tolist()), key=lambda x: x[1], reverse=True)

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r[0] for r in ranked][:15][::-1]
        vals = [r[1] for r in ranked][:15][::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(names, vals)
        plt.xlabel("mean |SHAP value|")
        plt.title(f"Feature importance ({horizon})")
        plt.tight_layout()
        out_png = SHAP_DIR / f"shap_{horizon}.png"
        plt.savefig(out_png, dpi=120)
        plt.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render SHAP plot: %s", exc)
        out_png = None

    return {
        "horizon": horizon,
        "importance": ranked,
        "plot_path": str(out_png) if out_png else None,
    }
