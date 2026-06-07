"""
Optional GridFS helpers for storing model artifacts inside MongoDB.

By default the project saves models to artifacts/models/ on disk and stores
only the path in the registry (simpler, faster). Use GridFS if you deploy
across machines that don't share a filesystem (e.g. CI trains, Render serves).
"""
from __future__ import annotations

import gridfs

from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.logging_config import get_logger

logger = get_logger(__name__)


def _fs() -> "gridfs.GridFS":
    return gridfs.GridFS(get_db())


def upload_artifact(local_path: str, filename: str) -> str:
    """Store a file in GridFS, return the file id as a string."""
    with open(local_path, "rb") as fh:
        file_id = _fs().put(fh, filename=filename)
    logger.info("Uploaded %s to GridFS (%s).", filename, file_id)
    return str(file_id)


def download_artifact(file_id: str, dest_path: str) -> str:
    """Download a GridFS file by id to dest_path."""
    from bson import ObjectId

    data = _fs().get(ObjectId(file_id)).read()
    with open(dest_path, "wb") as fh:
        fh.write(data)
    return dest_path
