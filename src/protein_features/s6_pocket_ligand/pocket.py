"""Stage 6: ligand + binding pocket (drug-discovery side).

Reads a PDB, pulls out the ligand (atoms/bonds, optional RDKit SMILES), nearby
pocket atoms, and typed contacts (H-bond / hydrophobic / polar, plus covalent
links), then packs them into tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..core import constants as C

log = logging.getLogger(__name__)

WATER = {"HOH", "WAT", "DOD"}
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "OTHER"]
ELEM_INDEX = {e: i for i, e in enumerate(ELEMENTS)}
DONOR_ACCEPTOR = {"N", "O"}


@dataclass
class Complex:
    """Ligand, pocket atoms, and typed contacts from one PDB."""

    ligand_resname: str
    ligand_elements: list
    ligand_names: list
    ligand_coords: np.ndarray
    ligand_bonds: list
    covalent_links: list
    pocket_elements: list
    pocket_names: list
    pocket_restypes: list
    pocket_res: list
    pocket_coords: np.ndarray
    pocket_is_backbone: np.ndarray
    contacts: list
    cutoff: float
    contact_cutoff: float
    meta: dict = field(default_factory=dict)

    @property
    def pocket_residues(self):
        """Unique pocket residues as (chain, resseq, resname) tuples."""
        seen, out = set(), []
        for r in self.pocket_res:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out


def _element(line, atom_name):
    """Element from PDB cols 77-78, or guess from the atom name."""
    e = line[76:78].strip()
    if e:
        return e[0].upper() + e[1:].lower() if len(e) == 2 else e.upper()
    nm = atom_name.strip()
    if nm[:2] in ("CL", "BR"):
        return nm[:1] + nm[1:2].lower()
    return nm[0]


def _parse(pdb_path):
    """Read ATOM/HETATM atoms and CONECT bonds from a PDB."""
    protein, hetero, conect = [], [], {}
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "ENDMDL":
                break
            if rec in ("ATOM", "HETATM"):
                if line[16] not in (" ", "A"):
                    continue
                serial = int(line[6:11])
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip() or "A"
                try:
                    resseq = int(line[22:26])
                    xyz = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=np.float32,
                    )
                except ValueError:
                    continue
                rec_atom = {
                    "serial": serial,
                    "name": name,
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                    "coord": xyz,
                    "element": _element(line, name),
                }
                (protein if rec == "ATOM" else hetero).append(rec_atom)
            elif rec == "CONECT":
                nums = [line[i : i + 5] for i in range(6, len(line.rstrip()), 5)]
                nums = [int(x) for x in nums if x.strip().isdigit()]
                if nums:
                    conect.setdefault(nums[0], []).extend(nums[1:])
    return protein, hetero, conect


def extract_complex(pdb_path, cutoff=5.0, contact_cutoff=4.0, ligand_resname=None):
    """Pull ligand, pocket atoms, and typed contacts out of ``pdb_path``."""
    protein, hetero, conect = _parse(pdb_path)
    ligands = [a for a in hetero if a["resname"] not in WATER]
    if not ligands:
        raise ValueError(f"No non-water ligand found in {pdb_path}")

    # Pick the largest non-water hetero group (or the named ligand if given)
    groups = {}
    for a in ligands:
        groups.setdefault((a["chain"], a["resseq"], a["resname"]), []).append(a)
    if ligand_resname:
        groups = {k: v for k, v in groups.items() if k[2] == ligand_resname}
        if not groups:
            raise ValueError(f"Ligand {ligand_resname} not found")
    key = max(groups, key=lambda k: len(groups[k]))
    lig = groups[key]
    lig_resname = key[2]
    lig_serials = {a["serial"]: i for i, a in enumerate(lig)}
    lig_coords = np.array([a["coord"] for a in lig], dtype=np.float32)

    # Bonds inside the ligand + any covalent link to the protein
    prot_by_serial = {a["serial"]: a for a in protein}
    bonds, covalent = set(), []
    for a in lig:
        for nb in conect.get(a["serial"], []):
            if nb in lig_serials:
                i, j = lig_serials[a["serial"]], lig_serials[nb]
                bonds.add((min(i, j), max(i, j)))
            elif nb in prot_by_serial:
                p = prot_by_serial[nb]
                covalent.append(
                    (lig_serials[a["serial"]], (p["chain"], p["resseq"], p["resname"], p["name"]))
                )

    # Pocket = protein heavy atoms within cutoff of any ligand atom
    prot_heavy = [a for a in protein if a["element"] != "H"]
    pkt = []
    for a in prot_heavy:
        d = np.linalg.norm(lig_coords - a["coord"], axis=1).min()
        if d <= cutoff:
            pkt.append(a)
    pkt_coords = np.array([a["coord"] for a in pkt], dtype=np.float32)
    backbone = np.array([a["name"] in ("N", "CA", "C", "O") for a in pkt], dtype=bool)

    # Typed close contacts
    contacts = []
    for li, a in enumerate(lig):
        if a["element"] == "H":
            continue
        for pi, b in enumerate(pkt):
            dist = float(np.linalg.norm(a["coord"] - b["coord"]))
            if dist <= contact_cutoff:
                if a["element"] in DONOR_ACCEPTOR and b["element"] in DONOR_ACCEPTOR:
                    ctype = "hbond"
                elif a["element"] == "C" and b["element"] == "C":
                    ctype = "hydrophobic"
                else:
                    ctype = "polar"
                contacts.append({"lig": li, "pocket": pi, "dist": round(dist, 2), "type": ctype})

    cx = Complex(
        ligand_resname=lig_resname,
        ligand_elements=[a["element"] for a in lig],
        ligand_names=[a["name"] for a in lig],
        ligand_coords=lig_coords,
        ligand_bonds=sorted(bonds),
        covalent_links=covalent,
        pocket_elements=[a["element"] for a in pkt],
        pocket_names=[a["name"] for a in pkt],
        pocket_restypes=[C.AA_3TO1.get(a["resname"], "X") for a in pkt],
        pocket_res=[(a["chain"], a["resseq"], a["resname"]) for a in pkt],
        pocket_coords=pkt_coords,
        pocket_is_backbone=backbone,
        contacts=contacts,
        cutoff=cutoff,
        contact_cutoff=contact_cutoff,
        meta={"n_ligand_atoms": len(lig), "n_pocket_atoms": len(pkt), "source": pdb_path},
    )
    log.info(
        "Ligand %s: %d atoms, %d bonds; pocket: %d atoms in %d residues; %d contacts",
        lig_resname,
        len(lig),
        len(bonds),
        len(pkt),
        len(cx.pocket_residues),
        len(contacts),
    )
    if covalent:
        log.info(
            "Covalent link(s) to protein: %s",
            ", ".join(f"{c[1][2]}{c[1][1]}:{c[1][3]}" for c in covalent),
        )
    return cx


def _elem_onehot(elements):
    x = np.zeros((len(elements), len(ELEMENTS)), dtype=np.float32)
    for i, e in enumerate(elements):
        x[i, ELEM_INDEX.get(e, ELEM_INDEX["OTHER"])] = 1.0
    return x


def encode_complex(cx: Complex) -> dict:
    """Turn a ``Complex`` into ligand/pocket/interaction tensors + schema."""
    # Ligand atom graph
    lig_x = _elem_onehot(cx.ligand_elements)
    lig_edges = (
        np.array(cx.ligand_bonds, dtype=np.int64).T
        if cx.ligand_bonds
        else np.zeros((2, 0), np.int64)
    )

    # Pocket atoms: element + residue type + backbone flag
    pkt_elem = _elem_onehot(cx.pocket_elements)
    pkt_rt = np.zeros((len(cx.pocket_restypes), C.NUM_RESTYPES), dtype=np.float32)
    for i, aa in enumerate(cx.pocket_restypes):
        pkt_rt[i, C.RESTYPE_TO_INDEX.get(aa, C.UNK_INDEX)] = 1.0
    pkt_x = np.concatenate(
        [pkt_elem, pkt_rt, cx.pocket_is_backbone[:, None].astype(np.float32)], axis=1
    )

    ctypes = {"hbond": 0, "hydrophobic": 1, "polar": 2}
    inter_index = (
        np.array([[c["lig"], c["pocket"]] for c in cx.contacts], dtype=np.int64).T
        if cx.contacts
        else np.zeros((2, 0), np.int64)
    )
    inter_feat = np.zeros((len(cx.contacts), 4), dtype=np.float32)
    for i, c in enumerate(cx.contacts):
        inter_feat[i, 0] = c["dist"]
        inter_feat[i, 1 + ctypes[c["type"]]] = 1.0

    return {
        "ligand_node_features": lig_x,
        "ligand_coords": cx.ligand_coords,
        "ligand_edge_index": lig_edges,
        "pocket_node_features": pkt_x,
        "pocket_coords": cx.pocket_coords,
        "interaction_edge_index": inter_index,
        "interaction_edge_features": inter_feat,
        "schema": {
            "elements": ELEMENTS,
            "ligand_node_dim": lig_x.shape[1],
            "pocket_node_dim": pkt_x.shape[1],
            "interaction_types": list(ctypes),
            "cutoff": cx.cutoff,
            "contact_cutoff": cx.contact_cutoff,
            "ligand_resname": cx.ligand_resname,
        },
    }


def summarize(cx: Complex) -> dict:
    """Plain-language summary: ligand, pocket residues, contact counts."""
    from collections import Counter

    ctype_counts = Counter(c["type"] for c in cx.contacts)
    formula = "".join(
        f"{e}{n}" for e, n in sorted(Counter(x for x in cx.ligand_elements if x != "H").items())
    )
    out = {
        "ligand": cx.ligand_resname,
        "ligand_atoms": len(cx.ligand_elements),
        "ligand_heavy_formula": formula,
        "ligand_bonds": len(cx.ligand_bonds),
        "covalent_to_protein": [f"{c[1][2]}{c[1][1]}:{c[1][3]}" for c in cx.covalent_links],
        "pocket_atoms": len(cx.pocket_elements),
        "pocket_residues": [f"{r[2]}{r[1]}" for r in cx.pocket_residues],
        "contacts_total": len(cx.contacts),
        "contacts_by_type": dict(ctype_counts),
        "cutoff_A": cx.cutoff,
    }
    try:
        smiles = _rdkit_smiles(cx)
        if smiles:
            out["ligand_smiles"] = smiles
    except Exception as e:
        log.debug("RDKit SMILES unavailable (%s); reporting formula only", e)
    return out


def _rdkit_smiles(cx: Complex):
    from rdkit import Chem  # pyright: ignore[reportMissingImports]
    from rdkit.Chem import Atom, Conformer, RWMol  # pyright: ignore[reportMissingImports]
    from rdkit.Geometry import Point3D  # pyright: ignore[reportMissingImports]

    mol = RWMol()
    for e in cx.ligand_elements:
        mol.AddAtom(Atom(e if e in ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I") else "C"))
    for i, j in cx.ligand_bonds:
        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    conf = Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(cx.ligand_coords):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in xyz]))
    mol.AddConformer(conf)
    m = mol.GetMol()
    Chem.SanitizeMol(m, catchErrors=True)
    return Chem.MolToSmiles(m)


def visualize(cx: Complex, out_path: str) -> str:
    """Save a 3D pocket/ligand scatter with coloured contacts to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*cx.pocket_coords.T, s=12, c="tab:gray", alpha=0.5, label="pocket atoms")
    ax.scatter(*cx.ligand_coords.T, s=30, c="tab:red", label=f"ligand {cx.ligand_resname}")
    for c in cx.contacts:
        p = np.stack([cx.ligand_coords[c["lig"]], cx.pocket_coords[c["pocket"]]])
        col = {"hbond": "tab:blue", "hydrophobic": "tab:green", "polar": "tab:orange"}[c["type"]]
        ax.plot(*p.T, c=col, lw=0.8, alpha=0.7)
    ax.set_title(
        f"{cx.ligand_resname} binding pocket "
        f"({len(cx.pocket_residues)} residues, {len(cx.contacts)} contacts)"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    """CLI for binding-pocket + ligand extraction."""
    import argparse
    import json
    import os

    from protein_features import configure_logging

    ap = argparse.ArgumentParser(description="Binding-pocket + ligand extraction")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--ligand", default=None, help="ligand resname (default: largest)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    a = ap.parse_args()
    configure_logging(a.verbose)
    out = a.out or os.path.join(
        "output", f"pocket_{os.path.splitext(os.path.basename(a.pdb))[0]}"
    )
    os.makedirs(out, exist_ok=True)

    cx = extract_complex(a.pdb, cutoff=a.cutoff, ligand_resname=a.ligand)
    enc = encode_complex(cx)
    summary = summarize(cx)
    np.savez_compressed(
        os.path.join(out, "complex_encoded.npz"),
        ligand_node_features=enc["ligand_node_features"],
        ligand_coords=enc["ligand_coords"],
        ligand_edge_index=enc["ligand_edge_index"],
        pocket_node_features=enc["pocket_node_features"],
        pocket_coords=enc["pocket_coords"],
        interaction_edge_index=enc["interaction_edge_index"],
        interaction_edge_features=enc["interaction_edge_features"],
    )
    with open(os.path.join(out, "complex_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    try:
        png = visualize(cx, os.path.join(out, "pocket.png"))
        log.info("Saved visualisation -> %s", png)
    except Exception as e:
        log.warning("Visualisation skipped: %s", e)
    log.info(
        "Done. Ligand %s (%s atoms), covalent=%s -> %s",
        summary.get("ligand"),
        summary.get("ligand_atoms"),
        summary.get("covalent_to_protein"),
        out,
    )


if __name__ == "__main__":
    main()
