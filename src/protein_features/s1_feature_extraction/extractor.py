"""Stage 1: parse a PDB into per-residue features.

Tries BioPython first; if that isn't around, uses a small built-in parser.
Returns residue type, Ca coords, backbone angles, SS, burial, and local frames
as an ExtractedProtein.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from ..core import constants as C
from ..core import geometry as G
from ..core.exceptions import InvalidPDBError

log = logging.getLogger(__name__)
BACKBONE = ("N", "CA", "C", "O")


@dataclass
class ExtractedProtein:
    """Per-residue features from a PDB, before encoding."""

    one_letter: list
    chain_ids: list
    res_ids: list
    ca_coords: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    omega: np.ndarray
    ss: list
    burial: np.ndarray
    frames: np.ndarray = None
    frame_mask: np.ndarray = None
    ss_source: str = "ramachandran"
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def num_residues(self) -> int:
        """Number of residues kept after parsing."""
        return len(self.one_letter)


# --- parsers ---


def _parse_with_biopython(pdb_path: str):
    """Parse with BioPython. Skips hetero groups and residues that have no CA."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())
    residues, skipped_het, skipped_no_ca = [], 0, 0
    for chain in model:
        for res in chain:
            hetflag, _, _ = res.id
            if hetflag.strip():
                skipped_het += 1
                continue
            atoms = {a.get_name(): np.array(a.get_coord(), dtype=np.float32) for a in res}
            if "CA" not in atoms:
                skipped_no_ca += 1
                continue
            residues.append(
                {
                    "resname": res.get_resname().strip(),
                    "chain": chain.id,
                    "resseq": res.id[1],
                    "atoms": atoms,
                }
            )
    meta = {"parser": "biopython", "skipped_het": skipped_het, "skipped_no_ca": skipped_no_ca}
    return residues, meta


def _parse_minimal(pdb_path: str):
    """Read ATOM columns by hand when BioPython isn't installed."""
    residues, seen, current = [], {}, None
    het_keys = set()
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "ENDMDL":
                break  # first model only
            if rec == "HETATM":
                het_keys.add((line[21], line[22:27]))
                continue
            if rec != "ATOM":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip() or "A"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            try:
                x, y, z = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            key = (chain, resseq, icode)
            if key not in seen:
                current = {"resname": resname, "chain": chain, "resseq": int(resseq), "atoms": {}}
                seen[key] = current
                residues.append(current)
            current["atoms"][atom] = np.array([x, y, z], dtype=np.float32)
    kept = [r for r in residues if "CA" in r["atoms"]]
    meta = {
        "parser": "minimal",
        "skipped_het": len(het_keys),
        "skipped_no_ca": len(residues) - len(kept),
    }
    return kept, meta


def _parse_pdb(pdb_path: str):
    """Try BioPython; fall back to the minimal parser if BioPython is missing or fails."""
    try:
        import Bio.PDB  # noqa: F401
    except ImportError:
        log.debug("BioPython not installed; using minimal PDB parser")
        return _parse_minimal(pdb_path)

    try:
        return _parse_with_biopython(pdb_path)
    except (OSError, ValueError) as e:
        log.warning("BioPython parse failed (%s); using minimal parser", e)
        return _parse_minimal(pdb_path)
    except Exception as e:
        # BioPython sometimes raises library-specific errors on corrupt files
        log.warning("BioPython parse failed (%s: %s); using minimal parser", type(e).__name__, e)
        return _parse_minimal(pdb_path)


# Map DSSP's 8 states down to helix / strand / coil
_DSSP3 = {
    "H": "H",
    "G": "H",
    "I": "H",
    "E": "E",
    "B": "E",
    "T": "C",
    "S": "C",
    "-": "C",
    " ": "C",
    "P": "C",
}


def _assign_dssp(pdb_path, chain_ids, res_ids):
    """Try real DSSP if mkdssp + BioPython are around; otherwise return None."""
    try:
        from Bio.PDB import DSSP, PDBParser

        model = PDBParser(QUIET=True).get_structure("p", pdb_path)[0]
        dssp = DSSP(model, pdb_path)
        mp = {}
        for key in dssp.keys():
            chain = key[0]
            resseq = key[1][1]
            ss8 = dssp[key][2]
            mp[(chain, resseq)] = _DSSP3.get(ss8, "C")
        return [mp.get((c, r), "C") for c, r in zip(chain_ids, res_ids)]
    except ImportError as e:
        log.debug("DSSP unavailable (%s); falling back to internal SS", e)
        return None
    except (OSError, ValueError, KeyError, IndexError) as e:
        log.debug("DSSP failed (%s); falling back to internal SS", e)
        return None
    except Exception as e:
        log.debug("DSSP failed (%s: %s); falling back to internal SS", type(e).__name__, e)
        return None


def extract(pdb_path: str) -> ExtractedProtein:
    """Parse ``pdb_path`` and return the per-residue structural features."""
    if not os.path.isfile(pdb_path):
        raise InvalidPDBError(f"PDB file not found: {pdb_path}")
    residues, meta = _parse_pdb(pdb_path)
    if not residues:
        raise InvalidPDBError(f"No standard protein residues with CA found in {pdb_path}")

    n = len(residues)
    one_letter = [C.AA_3TO1.get(r["resname"], "X") for r in residues]
    chain_ids = [r["chain"] for r in residues]
    res_ids = [r["resseq"] for r in residues]
    ca_coords = np.array([r["atoms"]["CA"] for r in residues], dtype=np.float32)

    def atom(i, name):
        return residues[i]["atoms"].get(name) if 0 <= i < n else None

    def bonded(lo, hi):
        # Only true for consecutive residues on the same chain (no fake angles across gaps)
        return (
            0 <= lo < n
            and 0 <= hi < n
            and chain_ids[lo] == chain_ids[hi]
            and res_ids[hi] - res_ids[lo] == 1
        )

    # Backbone dihedrals (NaN at chain ends / breaks)
    phi = np.full(n, np.nan, dtype=np.float32)
    psi = np.full(n, np.nan, dtype=np.float32)
    omega = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if bonded(i - 1, i):
            phi[i] = G.dihedral(atom(i - 1, "C"), atom(i, "N"), atom(i, "CA"), atom(i, "C"))
            omega[i] = G.dihedral(atom(i - 1, "CA"), atom(i - 1, "C"), atom(i, "N"), atom(i, "CA"))
        if bonded(i, i + 1):
            psi[i] = G.dihedral(atom(i, "N"), atom(i, "CA"), atom(i, "C"), atom(i + 1, "N"))

    backbone = {
        name: np.array(
            [residues[i]["atoms"].get(name, np.full(3, np.nan, np.float32)) for i in range(n)],
            dtype=np.float32,
        )
        for name in BACKBONE
    }

    # SS: real DSSP -> H-bond method -> Ramachandran fallback
    ss_dssp = _assign_dssp(pdb_path, chain_ids, res_ids)
    if ss_dssp is not None and len(ss_dssp) == n:
        ss, ss_source = ss_dssp, "dssp"
    else:
        try:
            ss = G.secondary_structure_from_hbonds(
                backbone["N"], backbone["CA"], backbone["C"], backbone["O"], chain_ids, res_ids
            )
            ss_source = "hbond"
        except (ValueError, RuntimeError, FloatingPointError) as e:
            log.warning("H-bond SS failed (%s); using Ramachandran fallback", e)
            ss = [G.classify_secondary_structure(phi[i], psi[i]) for i in range(n)]
            ss_source = "ramachandran"

    burial = G.coordination_number(ca_coords)
    frames, frame_mask = G.local_frames(backbone["N"], backbone["CA"], backbone["C"])

    chain_breaks = sum(
        1
        for i in range(1, n)
        if chain_ids[i] == chain_ids[i - 1] and res_ids[i] - res_ids[i - 1] != 1
    )
    meta.update(
        {
            "num_residues": n,
            "chains": sorted(set(chain_ids)),
            "chain_breaks": chain_breaks,
            "ss_source": ss_source,
            "valid_frames": int(frame_mask.sum()),
        }
    )
    log.info(
        "Extracted %d residues from %s (parser=%s, chains=%s)",
        n,
        pdb_path,
        meta.get("parser"),
        meta["chains"],
    )
    log.info(
        "Skipped %d hetero group(s), %d residue(s) without CA; %d chain break(s); SS=%s",
        meta.get("skipped_het", 0),
        meta.get("skipped_no_ca", 0),
        chain_breaks,
        ss_source,
    )

    return ExtractedProtein(
        one_letter=one_letter,
        chain_ids=chain_ids,
        res_ids=res_ids,
        ca_coords=ca_coords,
        phi=phi,
        psi=psi,
        omega=omega,
        ss=ss,
        burial=burial,
        frames=frames,
        frame_mask=frame_mask,
        ss_source=ss_source,
        source=pdb_path,
        meta=meta,
    )
