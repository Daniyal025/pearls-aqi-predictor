"""
Canonical collection names for the custom feature store.

MongoDB is NOT a dedicated ML feature store (like Hopsworks). Here we build a
feature-store-LIKE layer using plain collections plus deterministic ids and
indexes for dedup, point-in-time-ish reads, versioning, and logging.
"""

RAW_WEATHER = "raw_weather_data"
RAW_POLLUTION = "raw_pollution_data"
AQI_FEATURES = "aqi_features"
TRAINING_DATASETS = "training_datasets"
MODEL_REGISTRY = "model_registry"
MODEL_METRICS = "model_metrics"
PREDICTIONS = "predictions"
ALERTS = "alerts"

ALL_COLLECTIONS = [
    RAW_WEATHER,
    RAW_POLLUTION,
    AQI_FEATURES,
    TRAINING_DATASETS,
    MODEL_REGISTRY,
    MODEL_METRICS,
    PREDICTIONS,
    ALERTS,
]
