"""Encoded tensor bundle + save/load helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from .exceptions import SchemaError

REQUIRED_KEYS = (
    "node_features",
    "coords",
    "edge_index",
    "edge_features",
    "edge_dist",
    "schema",
)


@dataclass
class EncodedProtein:
    """Model-ready tensors from ``encode`` (dict-compatible for older call sites)."""

    node_features: np.ndarray
    coords: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_dist: np.ndarray
    schema: dict = field(repr=False)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return REQUIRED_KEYS

    def as_dict(self) -> dict:
        return {
            "node_features": self.node_features,
            "coords": self.coords,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "edge_dist": self.edge_dist,
            "schema": self.schema,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EncodedProtein:
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise SchemaError(f"Encoded protein missing keys: {missing}")
        return cls(
            node_features=np.asarray(data["node_features"]),
            coords=np.asarray(data["coords"]),
            edge_index=np.asarray(data["edge_index"]),
            edge_features=np.asarray(data["edge_features"]),
            edge_dist=np.asarray(data["edge_dist"]),
            schema=dict(data["schema"]),
        )


EncodedLike = Union[EncodedProtein, Mapping[str, Any]]


def validate_encoded(
    encoded: EncodedLike, *, expected_version: str | None = None
) -> EncodedProtein:
    """Check required keys and basic shape consistency; return an EncodedProtein."""
    enc = encoded if isinstance(encoded, EncodedProtein) else EncodedProtein.from_mapping(encoded)
    schema = enc.schema
    if not isinstance(schema, dict):
        raise SchemaError("schema must be a dict")

    if expected_version is not None and schema.get("version") != expected_version:
        raise SchemaError(
            f"schema version {schema.get('version')!r} != expected {expected_version!r}"
        )

    n = enc.node_features.shape[0]
    node_dim = int(schema.get("node_dim", -1))
    edge_dim = int(schema.get("edge_dim", -1))

    if enc.node_features.ndim != 2:
        raise SchemaError("node_features must be 2-D (N, D)")
    if node_dim > 0 and enc.node_features.shape[1] != node_dim:
        raise SchemaError(
            f"node_features width {enc.node_features.shape[1]} != schema node_dim {node_dim}"
        )
    if enc.coords.shape != (n, 3):
        raise SchemaError(f"coords shape {enc.coords.shape} != ({n}, 3)")
    if enc.edge_index.ndim != 2 or enc.edge_index.shape[0] != 2:
        raise SchemaError("edge_index must be shape (2, E)")
    e = enc.edge_index.shape[1]
    if enc.edge_features.shape[0] != e:
        raise SchemaError("edge_features rows must match edge_index columns")
    if edge_dim > 0 and enc.edge_features.shape[1] != edge_dim:
        raise SchemaError(
            f"edge_features width {enc.edge_features.shape[1]} != schema edge_dim {edge_dim}"
        )
    if enc.edge_dist.shape[0] != e:
        raise SchemaError("edge_dist length must match number of edges")
    for key in ("node_layout", "edge_layout", "chain_ids", "res_ids", "restypes", "ss_types"):
        if key not in schema:
            raise SchemaError(f"schema missing {key!r}")
    if len(schema["chain_ids"]) != n or len(schema["res_ids"]) != n:
        raise SchemaError("schema chain_ids/res_ids length must equal N")
    return enc


def save_encoded(out_dir: str | Path, encoded: EncodedLike) -> Path:
    """Write ``encoded.npz`` + ``schema.json`` under ``out_dir``."""
    enc = validate_encoded(encoded)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "encoded.npz",
        node_features=enc.node_features,
        coords=enc.coords,
        edge_index=enc.edge_index,
        edge_features=enc.edge_features,
        edge_dist=enc.edge_dist,
    )
    with open(out / "schema.json", "w", encoding="utf-8") as f:
        json.dump(enc.schema, f, indent=2)
    return out


def load_encoded(out_dir: str | Path, *, expected_version: str | None = None) -> EncodedProtein:
    """Load tensors + schema written by ``save_encoded``."""
    out = Path(out_dir)
    npz_path = out / "encoded.npz"
    schema_path = out / "schema.json"
    if not npz_path.is_file() or not schema_path.is_file():
        raise SchemaError(f"missing encoded.npz or schema.json under {out}")
    data = np.load(npz_path)
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    enc = EncodedProtein(
        node_features=data["node_features"],
        coords=data["coords"],
        edge_index=data["edge_index"],
        edge_features=data["edge_features"],
        edge_dist=data["edge_dist"],
        schema=schema,
    )
    return validate_encoded(enc, expected_version=expected_version)


def report_to_jsonable(report: Mapping[str, Any]) -> dict:
    """JSON-safe copy of a validation report (numpy scalars -> Python)."""
    out = {}
    for k, v in report.items():
        if isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
