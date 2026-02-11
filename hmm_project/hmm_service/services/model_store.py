"""
model_store.py
==============
In-memory store for trained HMM results, keyed by UUID.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional


class ModelStore:
    """Thread-safe (GIL-protected) in-memory model store."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def save(self, result_dict: dict[str, Any]) -> str:
        """Persist a training result and return its UUID key."""
        model_id = str(uuid.uuid4())
        self._store[model_id] = result_dict
        return model_id

    def get(self, model_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a result by ID, or ``None`` if not found."""
        return self._store.get(model_id)

    def list_ids(self) -> list[str]:
        """Return all stored model IDs."""
        return list(self._store.keys())


# Module-level singleton.
store = ModelStore()
