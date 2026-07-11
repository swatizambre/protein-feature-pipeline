"""Stage 3: turn encoded tensors back into named features.

Reads the schema and recovers residue type / SS (argmax), angles (atan2),
physchem + burial (undo norms), and edge distances (from RBF). Discrete +
geometry are exact; RBF distances are a bit approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import constants as C
from ..core import geometry as G
from ..core.io import EncodedLike, validate_encoded
from ..s2_encoding.encoder import SCHEMA_VERSION


@dataclass
class DecodedProtein:
    """Named residue-level fields you get back from ``decode``."""

    one_letter: list
    chain_ids: list
    res_ids: list
    ca_coords: np.ndarray
    phi_deg: np.ndarray
    psi_deg: np.ndarray
    omega_deg: np.ndarray
    ss: list
    burial: np.ndarray
    physchem: np.ndarray
    edge_index: np.ndarray
    edge_dist_recovered: np.ndarray


def _slice(layout_entry):
    """Turn a [start, stop] schema entry into a Python slice."""
    a, b = layout_entry
    return slice(a, b)


def decode(encoded: EncodedLike) -> DecodedProtein:
    """Undo ``encode``: tensors back to named biological quantities."""
    enc = validate_encoded(encoded, expected_version=SCHEMA_VERSION)
    schema = enc.schema
    x = enc.node_features
    layout = schema["node_layout"]
    restypes = schema["restypes"]
    ss_types = schema["ss_types"]

    # Discrete fields: pick the column with the 1
    rt = x[:, _slice(layout["restype_onehot"])]
    one_letter = [restypes[i] for i in rt.argmax(axis=1)]

    ssb = x[:, _slice(layout["ss_onehot"])]
    ss = [ss_types[i] for i in ssb.argmax(axis=1)]

    # Angles: (sin, cos, mask) -> degrees via atan2 (vectorised)
    dh = x[:, _slice(layout["dihedral_sincos"])]
    angles = []
    for k in range(3):
        s = dh[:, 3 * k]
        c = dh[:, 3 * k + 1]
        m = dh[:, 3 * k + 2]
        ang = np.where(m > 0.5, np.arctan2(s, c), np.nan).astype(np.float32)
        angles.append(np.degrees(ang))
    phi_deg, psi_deg, omega_deg = angles

    # Undo fixed physchem + burial normalisation
    physchem = C.denormalise_physchem(x[:, _slice(layout["physchem_norm"])])
    burial_norm = x[:, _slice(layout["burial_norm"])][:, 0]
    burial = burial_norm * C.BURIAL_STD + C.BURIAL_MEAN

    # Only approximate step: RBF fingerprint -> distance
    rbf = enc.edge_features[:, _slice(schema["edge_layout"]["rbf"])]
    edge_dist_recovered = G.rbf_decode(rbf)

    return DecodedProtein(
        one_letter=one_letter,
        chain_ids=list(schema["chain_ids"]),
        res_ids=list(schema["res_ids"]),
        ca_coords=enc.coords.copy(),
        phi_deg=phi_deg,
        psi_deg=psi_deg,
        omega_deg=omega_deg,
        ss=ss,
        burial=burial,
        physchem=physchem,
        edge_index=enc.edge_index,
        edge_dist_recovered=edge_dist_recovered,
    )
