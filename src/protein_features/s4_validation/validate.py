"""Stage 4: check that encode -> decode keeps the features.

Compares original vs decoded (type, SS, coords, angles, burial, physchem, edge
distances) and returns a report with a ``passed`` flag for the lossless parts.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..core import constants as C
from ..core.io import EncodedLike
from ..s1_feature_extraction.extractor import ExtractedProtein
from ..s3_decoding.decoder import DecodedProtein


def roundtrip_report(
    prot: ExtractedProtein, dec: DecodedProtein, encoded: EncodedLike
) -> dict:
    """Compare original vs decoded features. Returns errors and a ``passed`` flag."""
    report: dict = {
        "num_residues": int(prot.num_residues),
        "ss_source": prot.ss_source,
        "schema_version": encoded["schema"].get("version")
        if isinstance(encoded, Mapping)
        else encoded.schema.get("version"),
        "num_edges": int(encoded["edge_index"].shape[1]),
    }

    # Exact matches for discrete fields
    report["restype_exact_match"] = bool(prot.one_letter == dec.one_letter)
    report["ss_exact_match"] = bool(prot.ss == dec.ss)
    report["coords_max_abs_error"] = float(np.max(np.abs(prot.ca_coords - dec.ca_coords)))

    def ang_err(orig_rad, dec_deg):
        """Max absolute angle error in degrees (handles wrap-around)."""
        o = np.degrees(orig_rad)
        mask = ~np.isnan(o) & ~np.isnan(dec_deg)
        if mask.sum() == 0:
            return 0.0
        d = np.abs(((o[mask] - dec_deg[mask]) + 180) % 360 - 180)
        return float(d.max())

    report["phi_max_abs_error_deg"] = ang_err(prot.phi, dec.phi_deg)
    report["psi_max_abs_error_deg"] = ang_err(prot.psi, dec.psi_deg)
    report["omega_max_abs_error_deg"] = ang_err(prot.omega, dec.omega_deg)

    report["burial_max_abs_error"] = float(np.max(np.abs(prot.burial - dec.burial)))

    orig_phys = np.array([C.physchem_vector(aa) for aa in prot.one_letter], dtype=np.float32)
    report["physchem_max_abs_error"] = float(np.max(np.abs(orig_phys - dec.physchem)))

    # RBF distance recovery (expected to be slightly lossy)
    true_d = encoded["edge_dist"]
    rec_d = dec.edge_dist_recovered
    report["edge_dist_mean_abs_error_A"] = float(np.mean(np.abs(true_d - rec_d)))
    report["edge_dist_max_abs_error_A"] = float(np.max(np.abs(true_d - rec_d)))

    # Exact distances are still available from the stored coordinates
    src, dst = encoded["edge_index"]
    geom_d = np.linalg.norm(dec.ca_coords[src] - dec.ca_coords[dst], axis=1)
    report["edge_dist_from_coords_max_error_A"] = float(np.max(np.abs(true_d - geom_d)))

    report["passed"] = bool(
        report["restype_exact_match"]
        and report["ss_exact_match"]
        and report["coords_max_abs_error"] < 1e-4
        and report["phi_max_abs_error_deg"] < 1e-2
        and report["psi_max_abs_error_deg"] < 1e-2
        and report["omega_max_abs_error_deg"] < 1e-2
        and report["burial_max_abs_error"] < 1e-3
        and report["physchem_max_abs_error"] < 1e-4
    )
    return report
