"""protein_features: extract, encode, and decode protein structures.

Takes a PDB, builds a residue-level geometric graph (tensors), and can turn
those tensors back. Stages are s1_..s6_; shared helpers live in core.
"""

import logging

from .core import constants, geometry
from .core._log import configure_logging
from .core.exceptions import (
    InvalidPDBError,
    ProteinFeatureError,
    ResourceLimitError,
    SchemaError,
)
from .core.io import EncodedProtein, load_encoded, save_encoded
from .s1_feature_extraction.extractor import ExtractedProtein, extract
from .s2_encoding.encoder import EDGE_DIM, NODE_DIM, NODE_LAYOUT, encode
from .s3_decoding.decoder import DecodedProtein, decode
from .s4_validation.validate import roundtrip_report
from .s6_pocket_ligand import pocket
from .tools import synthetic

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "extract",
    "ExtractedProtein",
    "encode",
    "EncodedProtein",
    "NODE_DIM",
    "EDGE_DIM",
    "NODE_LAYOUT",
    "decode",
    "DecodedProtein",
    "roundtrip_report",
    "save_encoded",
    "load_encoded",
    "configure_logging",
    "constants",
    "geometry",
    "pocket",
    "synthetic",
    "ProteinFeatureError",
    "InvalidPDBError",
    "SchemaError",
    "ResourceLimitError",
]
__version__ = "1.1.0"
