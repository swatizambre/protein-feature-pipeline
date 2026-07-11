"""Shared constants for the pipeline.

AA tables, physchem properties, SS labels, and fixed normalisation / graph
settings. Hard-coded so every protein lands in the same numeric space.
"""

from __future__ import annotations

import numpy as np

# --- amino-acid names ---
# fmt: off
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
# fmt: on
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

RESTYPES = list("ARNDCQEGHILKMFPSTWYV")
UNK_INDEX = len(RESTYPES)
NUM_RESTYPES = len(RESTYPES) + 1
RESTYPE_TO_INDEX = {aa: i for i, aa in enumerate(RESTYPES)}


# --- per-residue chemistry tables ---
# fmt: off
_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
_CHARGE = {aa: 0.0 for aa in RESTYPES}
_CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})
_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}
# fmt: on
_POLAR = set("RNDQEHKSTYC")
_AROMATIC = set("FWYH")
_HBOND_DONOR = set("RKWNQHSTYC")
_HBOND_ACCEPTOR = set("DENQHSTY")

PHYSCHEM_KEYS = ["hydropathy", "charge", "volume", "polar", "aromatic", "hbond_don", "hbond_acc"]
NUM_PHYSCHEM = len(PHYSCHEM_KEYS)

# Fixed mean/std so every protein uses the same scale
_PHYSCHEM_MEAN = np.array([-0.49, 0.0, 137.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_PHYSCHEM_STD = np.array([2.90, 0.55, 39.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def physchem_vector(one_letter: str) -> np.ndarray:
    """Raw 7-D physchem vector for a 1-letter code. Zeros if unknown."""
    aa = one_letter if one_letter in _HYDROPATHY else None
    if aa is None:
        return np.zeros(NUM_PHYSCHEM, dtype=np.float32)
    return np.array(
        [
            _HYDROPATHY[aa],
            _CHARGE[aa],
            _VOLUME[aa],
            1.0 if aa in _POLAR else 0.0,
            1.0 if aa in _AROMATIC else 0.0,
            1.0 if aa in _HBOND_DONOR else 0.0,
            1.0 if aa in _HBOND_ACCEPTOR else 0.0,
        ],
        dtype=np.float32,
    )


def normalise_physchem(vec: np.ndarray) -> np.ndarray:
    """Standardise physchem features with the fixed mean/std."""
    return (vec - _PHYSCHEM_MEAN) / _PHYSCHEM_STD


def denormalise_physchem(vec: np.ndarray) -> np.ndarray:
    """Undo ``normalise_physchem`` (exact inverse)."""
    return vec * _PHYSCHEM_STD + _PHYSCHEM_MEAN


SS_TYPES = ["H", "E", "C"]
NUM_SS = len(SS_TYPES)
SS_TO_INDEX = {s: i for i, s in enumerate(SS_TYPES)}


# --- graph / geometry hyper-parameters ---
KNN_K = 16
BURIAL_RADIUS = 10.0
RBF_MIN = 0.0
RBF_MAX = 22.0
RBF_NUM = 16

BURIAL_MEAN = 15.0
BURIAL_STD = 8.0

# --- resource limits (web / defensive CLI) ---
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_RESIDUES = 20_000
JOB_TTL_SECONDS = 6 * 3600  # purge web job dirs older than this
