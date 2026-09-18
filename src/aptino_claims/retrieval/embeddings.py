"""Thin wrapper around fastembed's local ONNX dense embedding model.

Runs fully offline/free -- no API key -- so retrieval quality does not
depend on an external embeddings API being configured.
"""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from ..config import settings

_model_cache: dict[str, TextEmbedding] = {}


def _get_model(model_name: str) -> TextEmbedding:
    if model_name not in _model_cache:
        _model_cache[model_name] = TextEmbedding(model_name=model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Return an (N, D) L2-normalized float32 embedding matrix for passages/chunks."""
    model = _get_model(model_name or settings.dense_model)
    vecs = np.array(list(model.passage_embed(texts)), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_query(query: str, model_name: str | None = None) -> np.ndarray:
    model = _get_model(model_name or settings.dense_model)
    vec = np.array(list(model.query_embed([query]))[0], dtype="float32")
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec
