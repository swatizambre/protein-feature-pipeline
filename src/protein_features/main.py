#!/usr/bin/env python3
"""CLI for the full pipeline.

Runs extract -> encode -> decode -> validate on a PDB, then writes tensors,
schema, validation report, and the feature plot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from protein_features import configure_logging, decode, encode, extract, roundtrip_report
from protein_features.core.exceptions import (
    InvalidPDBError,
    ProteinFeatureError,
    ResourceLimitError,
)
from protein_features.core.io import report_to_jsonable, save_encoded

log = logging.getLogger(__name__)


def main(argv=None):
    """Run extract -> encode -> decode -> validate and write the outputs."""
    ap = argparse.ArgumentParser(description="Protein feature enc/dec pipeline")
    ap.add_argument("--pdb", required=True, help="input PDB file (any structure)")
    ap.add_argument(
        "--out",
        default=None,
        help="output directory (default: output/pipeline_<pdb-name>)",
    )
    ap.add_argument("--no-viz", action="store_true", help="skip visualisation")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)
    configure_logging(args.verbose)

    if not os.path.isfile(args.pdb):
        raise SystemExit(f"PDB file not found: {args.pdb}")
    stem = os.path.splitext(os.path.basename(args.pdb))[0]
    out = args.out or os.path.join("output", f"pipeline_{stem}")
    os.makedirs(out, exist_ok=True)

    try:
        log.info("Extracting features from %s", args.pdb)
        prot = extract(args.pdb)

        log.info("Encoding")
        enc = encode(prot)
        log.info(
            "node_features %s, edges %d, edge_features %s",
            enc["node_features"].shape,
            enc["edge_index"].shape[1],
            enc["edge_features"].shape,
        )

        log.info("Decoding")
        dec = decode(enc)

        log.info("Validating round-trip")
        report = roundtrip_report(prot, dec, enc)
        for k, v in report.items():
            log.info("  %s: %s", k, v)

        save_encoded(out, enc)
        with open(os.path.join(out, "validation_report.json"), "w", encoding="utf-8") as f:
            json.dump(report_to_jsonable(report), f, indent=2)
        log.info("Saved tensors -> %s", os.path.join(out, "encoded.npz"))

        if not args.no_viz:
            try:
                from protein_features.s5_visualization.visualize import plot_summary

                png = plot_summary(prot, enc, os.path.join(out, "features.png"))
                log.info("Saved visualisation -> %s", png)
            except Exception as e:
                log.warning("Visualisation skipped: %s", e)

        log.info("Done. Round-trip passed: %s", report["passed"])
        if not report["passed"]:
            return 1
        return 0
    except (InvalidPDBError, ResourceLimitError) as e:
        log.error("%s", e)
        return 1
    except ProteinFeatureError as e:
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
