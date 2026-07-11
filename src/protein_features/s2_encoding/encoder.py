"""Stage 2: turn ExtractedProtein into tensors a model can eat.

Builds a fixed-width node feature matrix and a kNN edge graph (distance,
sequence separation, relative orientation). Same layout every time; the schema
tags each column so decode knows what it's looking at.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core import constants as C
from ..core import geometry as G
from ..core.exceptions import ResourceLimitError
from ..core.io import EncodedProtein
from ..s1_feature_extraction.extractor import ExtractedProtein

log = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"

# --- node feature layout (41 numbers per residue) ---
# Hand out consecutive column ranges so the layout stays consistent.
_n = 0


def _slot(width):
    """Reserve the next ``width`` columns in the node vector."""
    global _n
    s = slice(_n, _n + width)
    _n += width
    return s


RESTYPE_SLICE = _slot(C.NUM_RESTYPES)  # 21: one-hot AA (+ UNK)
PHYSCHEM_SLICE = _slot(C.NUM_PHYSCHEM)  # 7: chemistry (normalised)
SS_SLICE = _slot(C.NUM_SS)  # 3: H / E / C
DIHEDRAL_SLICE = _slot(9)  # 9: phi, psi, omega as (sin, cos, mask)
BURIAL_SLICE = _slot(1)  # 1: how buried the residue is
NODE_DIM = _n

NODE_LAYOUT = {
    "restype_onehot": [RESTYPE_SLICE.start, RESTYPE_SLICE.stop],
    "physchem_norm": [PHYSCHEM_SLICE.start, PHYSCHEM_SLICE.stop],
    "ss_onehot": [SS_SLICE.start, SS_SLICE.stop],
    "dihedral_sincos": [DIHEDRAL_SLICE.start, DIHEDRAL_SLICE.stop],
    "burial_norm": [BURIAL_SLICE.start, BURIAL_SLICE.stop],
}

# --- edge feature layout (26 numbers per neighbour link) ---
# RBF_NUM is 16, so: distance (16) + seq gap (1) + same-chain (1) + orientation (8)
EDGE_RBF_SLICE = slice(0, C.RBF_NUM)
EDGE_SEQSEP_SLICE = slice(C.RBF_NUM, C.RBF_NUM + 1)
EDGE_SAMECHAIN_SLICE = slice(C.RBF_NUM + 1, C.RBF_NUM + 2)
EDGE_ORIENT_SLICE = slice(C.RBF_NUM + 2, C.RBF_NUM + 10)
EDGE_DIM = C.RBF_NUM + 10


def encode(prot: ExtractedProtein) -> EncodedProtein:
    """Turn extracted features into node/edge tensors plus a schema.

    Returns an ``EncodedProtein`` (dict-compatible) with node_features, coords,
    edge_index, edge_features, edge_dist, and schema.
    """
    n = prot.num_residues
    if n > C.MAX_RESIDUES:
        raise ResourceLimitError(
            f"Too many residues ({n} > {C.MAX_RESIDUES}); raise MAX_RESIDUES if intentional"
        )
    x = np.zeros((n, NODE_DIM), dtype=np.float32)

    # Residue type: put a 1 in the matching column (unknown -> UNK)
    for i, aa in enumerate(prot.one_letter):
        col = RESTYPE_SLICE.start + C.RESTYPE_TO_INDEX.get(aa, C.UNK_INDEX)
        x[i, col] = 1.0

    # Chemistry: hydropathy, charge, volume, polar/aromatic/H-bond flags
    physchem = np.array([C.physchem_vector(aa) for aa in prot.one_letter], dtype=np.float32)
    x[:, PHYSCHEM_SLICE] = C.normalise_physchem(physchem)

    # Secondary structure: H / E / C one-hot
    for i, s in enumerate(prot.ss):
        x[i, SS_SLICE.start + C.SS_TO_INDEX[s]] = 1.0

    # Backbone angles as (sin, cos, mask) so wrap-around is smooth
    for k, ang in enumerate((prot.phi, prot.psi, prot.omega)):
        base = DIHEDRAL_SLICE.start + 3 * k
        for i in range(n):
            s, c, m = G.angle_to_sincos(ang[i])
            x[i, base : base + 3] = (s, c, m)

    # Burial: rescale with fixed mean/std (same for every protein)
    x[:, BURIAL_SLICE.start] = (prot.burial - C.BURIAL_MEAN) / C.BURIAL_STD

    # Neighbour graph in 3D
    edge_index, edge_dist = G.knn_graph(prot.ca_coords, C.KNN_K)
    edge_features = np.zeros((edge_index.shape[1], EDGE_DIM), dtype=np.float32)
    edge_features[:, EDGE_RBF_SLICE] = G.rbf_expand(edge_dist)

    src, dst = edge_index
    res_ids = np.array(prot.res_ids)
    seqsep = res_ids[dst].astype(np.float32) - res_ids[src].astype(np.float32)
    # Signed log gap along the sequence (keeps large gaps from dominating)
    edge_features[:, EDGE_SEQSEP_SLICE.start] = np.sign(seqsep) * np.log1p(np.abs(seqsep))

    same_chain = np.array(
        [prot.chain_ids[s] == prot.chain_ids[d] for s, d in zip(src, dst)],
        dtype=np.float32,
    )
    edge_features[:, EDGE_SAMECHAIN_SLICE.start] = same_chain

    if prot.frames is not None and edge_index.shape[1] > 0:
        orient = G.orientation_edge_features(
            prot.ca_coords, prot.frames, prot.frame_mask, edge_index
        )
        edge_features[:, EDGE_ORIENT_SLICE] = orient

    # Schema travels with the tensors so decode needs no extra state
    schema = {
        "version": SCHEMA_VERSION,
        "node_dim": NODE_DIM,
        "edge_dim": EDGE_DIM,
        "node_layout": NODE_LAYOUT,
        "edge_layout": {
            "rbf": [EDGE_RBF_SLICE.start, EDGE_RBF_SLICE.stop],
            "seqsep": [EDGE_SEQSEP_SLICE.start, EDGE_SEQSEP_SLICE.stop],
            "same_chain": [EDGE_SAMECHAIN_SLICE.start, EDGE_SAMECHAIN_SLICE.stop],
            "orientation": [EDGE_ORIENT_SLICE.start, EDGE_ORIENT_SLICE.stop],
        },
        "knn_k": C.KNN_K,
        "rbf": {"min": C.RBF_MIN, "max": C.RBF_MAX, "num": C.RBF_NUM},
        "ss_source": prot.ss_source,
        "restypes": C.RESTYPES + ["X"],
        "ss_types": C.SS_TYPES,
        "chain_ids": list(prot.chain_ids),
        "res_ids": list(prot.res_ids),
        "source": prot.source,
        "extraction_meta": prot.meta,
    }

    log.debug("Encoded nodes=%s edges=%d edge_dim=%d", x.shape, edge_index.shape[1], EDGE_DIM)
    return EncodedProtein(
        node_features=x,
        coords=prot.ca_coords.copy(),
        edge_index=edge_index,
        edge_features=edge_features,
        edge_dist=edge_dist,
        schema=schema,
    )
