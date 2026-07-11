#!/usr/bin/env python3
"""Build a small valid test PDB.

Places backbone atoms with ideal bond lengths/angles so tests have a structure
whose recomputed angles match the inputs.
"""

import numpy as np

# Ideal backbone bond lengths (A) and angles (deg)
BL = {"N_CA": 1.458, "CA_C": 1.525, "C_N": 1.329, "C_O": 1.231, "CA_CB": 1.521}
BA = {"N_CA_C": 111.2, "CA_C_N": 116.2, "C_N_CA": 121.7, "CA_C_O": 120.8, "N_CA_CB": 110.4}


def place(a, b, c, length, angle_deg, dih_deg):
    """Place the next atom from three previous ones (NeRF-style)."""
    ang, dih = np.radians(angle_deg), np.radians(dih_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array(
        [
            -length * np.cos(ang),
            length * np.sin(ang) * np.cos(dih),
            length * np.sin(ang) * np.sin(dih),
        ]
    )
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def build(seq, phis, psis, omega=180.0):
    """Build backbone (+ CB) coords for ``seq`` from ideal geometry."""
    N = np.array([0.0, 0.0, 0.0])
    CA = np.array([BL["N_CA"], 0.0, 0.0])
    C = place(np.array([-1.0, 1.0, 0.0]), N, CA, BL["CA_C"], BA["N_CA_C"], 0.0)
    residues = [{"N": N, "CA": CA, "C": C}]
    for i in range(1, len(seq)):
        pN, pCA, pC = residues[i - 1]["N"], residues[i - 1]["CA"], residues[i - 1]["C"]
        n = place(pN, pCA, pC, BL["C_N"], BA["CA_C_N"], psis[i - 1])
        ca = place(pCA, pC, n, BL["N_CA"], BA["C_N_CA"], omega)
        c = place(pC, n, ca, BL["CA_C"], BA["N_CA_C"], phis[i])
        residues.append({"N": n, "CA": ca, "C": c})
    for i, r in enumerate(residues):
        psi = psis[i] if i < len(psis) else 0.0
        r["O"] = place(r["N"], r["CA"], r["C"], BL["C_O"], BA["CA_C_O"], psi + 180.0)
        if seq[i] != "G":
            r["CB"] = place(r["C"], r["N"], r["CA"], BL["CA_CB"], BA["N_CA_CB"], -122.0)
    return residues


def write_pdb(seq, residues, path):
    """Write a minimal PDB with ATOM records for the built residues."""
    # fmt: off
    three = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
             "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
             "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
             "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
    # fmt: on
    order = ["N", "CA", "C", "O", "CB"]
    serial = 1
    with open(path, "w") as f:
        for i, r in enumerate(residues):
            for atom in order:
                if atom not in r:
                    continue
                x, y, z = r[atom]
                elem = atom[0]
                name4 = (" " + atom) if len(atom) < 4 else atom
                line = (
                    f"ATOM  {serial:>5} "
                    f"{name4:<4}"
                    " "
                    f"{three[seq[i]]:>3} "
                    "A"
                    f"{i+1:>4}"
                    " "
                    "   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{1.00:6.2f}{0.00:6.2f}"
                    "          "
                    f"{elem:>2}\n"
                )
                f.write(line)
                serial += 1
        f.write("END\n")


def main():
    """CLI: write a small helix+strand test PDB with ideal backbone geometry."""
    import argparse

    ap = argparse.ArgumentParser(description="Generate a valid test PDB")
    ap.add_argument("--out", default="test_structure.pdb")
    a = ap.parse_args()
    seq = "MKTAYIAGLDEHVWFSNGPRQK"
    n = len(seq)
    phis = [-60.0] * 11 + [-120.0] * (n - 11)
    psis = [-45.0] * 11 + [130.0] * (n - 11)
    write_pdb(seq, build(seq, phis, psis), a.out)
    print(f"Wrote {a.out} ({n} residues)")


if __name__ == "__main__":
    main()
