"""
Round-trip and structural tests. Runs under pytest, or standalone:
    python tests/test_roundtrip.py
"""

import os
import tempfile

import numpy as np

from protein_features import EDGE_DIM, NODE_DIM, decode, encode, extract, roundtrip_report
from protein_features import synthetic as mk


def _make_pdb(path):
    seq = "MKTAYIAGLDEHVWFSNGPRQK"
    n = len(seq)
    phis = [-60.0] * 11 + [-120.0] * (n - 11)
    psis = [-45.0] * 11 + [130.0] * (n - 11)
    mk.write_pdb(seq, mk.build(seq, phis, psis), path)
    return seq


def test_roundtrip_lossless_parts():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.pdb")
        seq = _make_pdb(path)
        prot = extract(path)
        assert prot.num_residues == len(seq)
        assert "".join(prot.one_letter) == seq

        enc = encode(prot)
        assert enc["node_features"].shape == (len(seq), NODE_DIM)
        assert enc["edge_features"].shape[1] == EDGE_DIM
        assert enc["edge_index"].shape[0] == 2

        dec = decode(enc)
        rep = roundtrip_report(prot, dec, enc)
        assert rep["restype_exact_match"]
        assert rep["ss_exact_match"]
        assert rep["coords_max_abs_error"] < 1e-4
        assert rep["phi_max_abs_error_deg"] < 1e-2
        assert rep["psi_max_abs_error_deg"] < 1e-2
        assert rep["edge_dist_from_coords_max_error_A"] < 1e-4
        assert rep["passed"]


def test_determinism():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.pdb")
        _make_pdb(path)
        prot = extract(path)
        a, b = encode(prot), encode(prot)
        assert np.array_equal(a["node_features"], b["node_features"])
        assert np.array_equal(a["edge_index"], b["edge_index"])
        assert np.array_equal(a["edge_features"], b["edge_features"])


def test_unknown_residue_bucket():
    # residue name outside the standard 20 must land in the UNK one-hot slot
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.pdb")
        _make_pdb(path)
        with open(path) as f:
            lines = f.readlines()
        lines = [
            ln.replace("MET A   1", "XYZ A   1") if ln.startswith("ATOM") and " A   1" in ln else ln
            for ln in lines
        ]
        with open(path, "w") as f:
            f.writelines(lines)
        prot = extract(path)
        assert prot.one_letter[0] == "X"
        enc = encode(prot)
        dec = decode(enc)
        assert dec.one_letter[0] == "X"


def test_chain_gap_no_bogus_dihedral():
    # deleting a residue mid-chain must NOT invent a backbone angle across the gap
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.pdb")
        _make_pdb(path)
        lines = [
            ln for ln in open(path) if not (ln.startswith("ATOM") and ln[22:26].strip() == "6")
        ]
        with open(path, "w") as f:
            f.writelines(lines)
        prot = extract(path)
        assert 6 not in prot.res_ids
        assert prot.meta["chain_breaks"] == 1
        i5 = prot.res_ids.index(5)
        i7 = prot.res_ids.index(7)
        assert np.isnan(prot.psi[i5])  # residue before the gap
        assert np.isnan(prot.phi[i7])  # residue after the gap
        # pipeline should still run and round-trip despite the gap
        enc = encode(prot)
        dec = decode(enc)
        assert roundtrip_report(prot, dec, enc)["passed"]


def test_orientation_features_se3_invariant():
    # relative-orientation edge features must not change under a global
    # rotation + translation of the whole structure
    from protein_features import geometry as G

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.pdb")
        _make_pdb(path)
        prot = extract(path)
        enc = encode(prot)
        ei = enc["edge_index"]
        orient1 = G.orientation_edge_features(prot.ca_coords, prot.frames, prot.frame_mask, ei)

        # random rotation + translation
        rng = np.random.default_rng(0)
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)  # orthonormal
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1  # ensure a proper rotation
        t = rng.normal(size=3).astype(np.float32)
        coords2 = (prot.ca_coords @ Q.T + t).astype(np.float32)
        frames2 = np.einsum("ij,njk->nik", Q, prot.frames).astype(np.float32)
        orient2 = G.orientation_edge_features(coords2, frames2, prot.frame_mask, ei)

        assert np.allclose(orient1, orient2, atol=1e-4)


def test_pocket_extraction_7rfw():
    # binding-pocket + ligand extraction on the bundled real structure
    import pytest

    from protein_features import pocket

    pdb = os.path.join(os.path.dirname(__file__), "..", "data", "7rfw.pdb")
    if not os.path.exists(pdb):
        pytest.skip("data/7rfw.pdb not present")
    cx = pocket.extract_complex(pdb, cutoff=5.0)
    assert cx.ligand_resname == "4WI"  # nirmatrelvir
    assert cx.meta["n_ligand_atoms"] == 68
    # covalent inhibitor: must find the bond to the catalytic Cys145
    cov = [f"{c[1][2]}{c[1][1]}" for c in cx.covalent_links]
    assert "CYS145" in cov
    # active-site residues must be in the pocket (catalytic dyad His41/Cys145)
    res = {f"{r[2]}{r[1]}" for r in cx.pocket_residues}
    assert "HIS41" in res and "CYS145" in res
    # encoding produces the three graphs
    enc = pocket.encode_complex(cx)
    assert enc["ligand_node_features"].shape[0] == 68
    assert enc["pocket_node_features"].shape[0] == len(cx.pocket_elements)
    assert enc["interaction_edge_index"].shape[0] == 2


def test_save_load_encoded_roundtrip():
    from protein_features import load_encoded, save_encoded

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.pdb")
        _make_pdb(path)
        enc = encode(extract(path))
        out = os.path.join(d, "out")
        save_encoded(out, enc)
        loaded = load_encoded(out)
        assert np.array_equal(loaded.node_features, enc.node_features)
        assert loaded.schema["version"] == enc.schema["version"]


def test_decode_rejects_bad_schema():
    import pytest

    from protein_features import SchemaError

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.pdb")
        _make_pdb(path)
        enc = encode(extract(path))
        bad = enc.as_dict()
        bad["schema"] = dict(bad["schema"])
        bad["schema"]["node_dim"] = 999
        with pytest.raises(SchemaError):
            decode(bad)


def test_7rfw_roundtrip():
    import pytest

    pdb = os.path.join(os.path.dirname(__file__), "..", "data", "7rfw.pdb")
    if not os.path.exists(pdb):
        pytest.skip("data/7rfw.pdb not present")
    prot = extract(pdb)
    enc = encode(prot)
    dec = decode(enc)
    rep = roundtrip_report(prot, dec, enc)
    assert prot.num_residues == 306
    assert rep["passed"]
    assert rep["burial_max_abs_error"] < 1e-3
    assert rep["physchem_max_abs_error"] < 1e-4


if __name__ == "__main__":
    test_roundtrip_lossless_parts()
    test_determinism()
    test_unknown_residue_bucket()
    test_chain_gap_no_bogus_dihedral()
    test_orientation_features_se3_invariant()
    test_pocket_extraction_7rfw()
    test_save_load_encoded_roundtrip()
    test_decode_rejects_bad_schema()
    test_7rfw_roundtrip()
    print("All tests passed.")
