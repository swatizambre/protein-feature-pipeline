"""Geometry helpers used across the pipeline.

Dihedrals, sin/cos angles, RBF distances, kNN graph, burial, secondary
structure, local frames, and orientation edge features. Pure numpy on coords.
"""

from __future__ import annotations

import numpy as np

from . import constants as C

# --- angles ---


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed torsion (radians) for four points. NaN if any atom is missing."""
    if any(p is None for p in (p0, p1, p2, p3)):
        return np.nan
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.arctan2(y, x))


def angle_to_sincos(angle_rad: float) -> tuple[float, float, float]:
    """Turn an angle into (sin, cos, mask). Zeros if the angle is undefined."""
    if angle_rad is None or np.isnan(angle_rad):
        return 0.0, 0.0, 0.0
    return float(np.sin(angle_rad)), float(np.cos(angle_rad)), 1.0


def sincos_to_angle(sin_v: float, cos_v: float, mask: float) -> float:
    """Recover an angle from (sin, cos, mask). NaN when the mask is off."""
    if mask < 0.5:
        return np.nan
    return float(np.arctan2(sin_v, cos_v))


# --- distances (RBF) ---


def rbf_centers() -> np.ndarray:
    """Evenly spaced RBF centres over ``[RBF_MIN, RBF_MAX]``."""
    return np.linspace(C.RBF_MIN, C.RBF_MAX, C.RBF_NUM, dtype=np.float32)


def rbf_expand(dist: np.ndarray) -> np.ndarray:
    """Turn distances into a Gaussian RBF feature matrix."""
    dist = np.clip(dist, C.RBF_MIN, C.RBF_MAX)
    centers = rbf_centers()
    width = (C.RBF_MAX - C.RBF_MIN) / C.RBF_NUM
    diff = dist[:, None] - centers[None, :]
    return np.exp(-(diff**2) / (2.0 * width**2)).astype(np.float32)


def rbf_decode(rbf: np.ndarray) -> np.ndarray:
    """Approximate distances from RBF features (weighted mean of the centres)."""
    centers = rbf_centers()
    weights = rbf / (rbf.sum(axis=1, keepdims=True) + 1e-8)
    return (weights * centers[None, :]).sum(axis=1).astype(np.float32)


# --- graph + burial ---


def knn_graph(coords: np.ndarray, k: int = C.KNN_K, radius: float | None = C.RBF_MAX):
    """kNN edges from coords, capped by ``radius``.

    Uses ``scipy.spatial.cKDTree`` when SciPy is installed (O(N log N)); otherwise
    falls back to a dense pairwise distance matrix.
    """
    n = coords.shape[0]
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    kk = min(k, n - 1)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        # workers=1 keeps behaviour portable across SciPy / OS builds
        dists, nbr = tree.query(coords, k=kk + 1, workers=1)
        # column 0 is self (distance 0); drop it
        dists = np.asarray(dists[:, 1:], dtype=np.float32)
        nbr = np.asarray(nbr[:, 1:], dtype=np.int64)
        src_list, dst_list, dist_list = [], [], []
        for i in range(n):
            order = np.argsort(dists[i], kind="stable")
            for t in order:
                j = int(nbr[i, t])
                d = float(dists[i, t])
                if radius is None or d <= radius:
                    src_list.append(i)
                    dst_list.append(j)
                    dist_list.append(d)
        if not src_list:
            for i in range(n):
                j = int(nbr[i, 0])
                src_list.append(i)
                dst_list.append(j)
                dist_list.append(float(dists[i, 0]))
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_dist = np.array(dist_list, dtype=np.float32)
        return edge_index, edge_dist
    except ImportError:
        pass

    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt((diff**2).sum(-1) + 1e-8)
    np.fill_diagonal(dmat, np.inf)
    nbr = np.argsort(dmat, axis=1, kind="stable")[:, :kk]
    src_list, dst_list = [], []
    for i in range(n):
        for j in nbr[i]:
            if radius is None or dmat[i, j] <= radius:
                src_list.append(i)
                dst_list.append(int(j))
    if not src_list:
        for i in range(n):
            j = int(nbr[i, 0])
            src_list.append(i)
            dst_list.append(j)
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_dist = dmat[edge_index[0], edge_index[1]].astype(np.float32)
    return edge_index, edge_dist


def coordination_number(coords: np.ndarray, radius: float = C.BURIAL_RADIUS):
    """Count neighbours within ``radius`` (burial proxy; self not counted)."""
    diff = coords[:, None, :] - coords[None, :, :]
    dmat = np.sqrt((diff**2).sum(-1) + 1e-8)
    within = (dmat < radius).sum(axis=1) - 1
    return within.astype(np.float32)


# --- secondary structure ---


def classify_secondary_structure(phi: float, psi: float) -> str:
    """Rough H/E/C from Ramachandran regions. Used when DSSP/H-bonds aren't available."""
    if phi is None or psi is None or np.isnan(phi) or np.isnan(psi):
        return "C"
    phid, psid = np.degrees(phi), np.degrees(psi)
    if -160.0 <= phid <= -20.0 and -120.0 <= psid <= 40.0:
        return "H"
    if -180.0 <= phid <= -40.0 and (90.0 <= psid <= 180.0 or -180.0 <= psid <= -150.0):
        return "E"
    return "C"


def secondary_structure_from_hbonds(N, CA, C, O, chain_ids, res_ids):  # noqa: E741
    """Assign H/E/C from backbone H-bonds (DSSP-style energy and patterns)."""
    n = CA.shape[0]
    if n == 0:
        return []

    H = np.full((n, 3), np.nan, dtype=np.float32)
    for i in range(1, n):
        if (
            chain_ids[i] == chain_ids[i - 1]
            and res_ids[i] - res_ids[i - 1] == 1
            and not np.any(np.isnan(N[i]))
            and not np.any(np.isnan(C[i - 1]))
            and not np.any(np.isnan(O[i - 1]))
        ):
            co = C[i - 1] - O[i - 1]
            nco = np.linalg.norm(co)
            if nco > 1e-6:
                H[i] = N[i] + 1.01 * co / nco

    Q = 0.084 * 332.0
    hbond = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if np.any(np.isnan(H[i])) or np.any(np.isnan(N[i])):
            continue
        for j in range(n):
            if i == j or abs(i - j) < 1:
                continue
            if np.any(np.isnan(O[j])) or np.any(np.isnan(C[j])):
                continue
            r_ON = np.linalg.norm(O[j] - N[i])
            if r_ON > 5.0:
                continue
            r_CH = np.linalg.norm(C[j] - H[i])
            r_OH = np.linalg.norm(O[j] - H[i])
            r_CN = np.linalg.norm(C[j] - N[i])
            E = Q * (1.0 / r_ON + 1.0 / r_CH - 1.0 / r_OH - 1.0 / r_CN)
            if E < -0.5:
                hbond[i, j] = True

    def dssp_hb(a, b):
        return 0 <= a < n and 0 <= b < n and hbond[b, a]

    helix = np.zeros(n, dtype=bool)
    for turn in (4, 3, 5):
        for i in range(n - turn - 1):
            if dssp_hb(i, i + turn) and dssp_hb(i + 1, i + 1 + turn):
                helix[i + 1 : i + turn + 1] = True

    strand = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(i + 3, n):
            anti = (dssp_hb(i, j) and dssp_hb(j, i)) or (
                dssp_hb(i - 1, j + 1) and dssp_hb(j - 1, i + 1)
            )
            para = (dssp_hb(i - 1, j) and dssp_hb(j, i + 1)) or (
                dssp_hb(j - 1, i) and dssp_hb(i, j + 1)
            )
            if anti or para:
                strand[i] = True
                strand[j] = True

    ss = []
    for i in range(n):
        if helix[i]:
            ss.append("H")
        elif strand[i]:
            ss.append("E")
        else:
            ss.append("C")
    return ss


# --- local frames + orientation edges ---


def local_frames(N: np.ndarray, CA: np.ndarray, C: np.ndarray):
    """Per-residue backbone frame from N/CA/C. Returns frames + a validity mask."""
    n = CA.shape[0]
    R = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if np.any(np.isnan(N[i])) or np.any(np.isnan(C[i])) or np.any(np.isnan(CA[i])):
            mask[i] = False
            continue
        v1 = C[i] - CA[i]
        v2 = N[i] - CA[i]
        n1 = np.linalg.norm(v1)
        if n1 < 1e-6:
            mask[i] = False
            continue
        e1 = v1 / n1
        u2 = v2 - np.dot(v2, e1) * e1
        n2 = np.linalg.norm(u2)
        if n2 < 1e-6:
            mask[i] = False
            continue
        e2 = u2 / n2
        e3 = np.cross(e1, e2)
        R[i] = np.stack([e1, e2, e3], axis=1)
    return R.astype(np.float32), mask


def rotation_to_quaternion(Rm: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z)."""
    m = Rm
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def orientation_edge_features(coords, frames, frame_mask, edge_index):
    """Local direction + relative quaternion per edge (SE(3)-invariant)."""
    src, dst = edge_index
    E = src.shape[0]
    out = np.zeros((E, 8), dtype=np.float32)
    for e in range(E):
        i, j = int(src[e]), int(dst[e])
        if not (frame_mask[i] and frame_mask[j]):
            continue
        Ri = frames[i]
        d = coords[j] - coords[i]
        dn = np.linalg.norm(d)
        direction = (Ri.T @ d) / (dn + 1e-8)
        Rrel = Ri.T @ frames[j]
        quat = rotation_to_quaternion(Rrel)
        out[e, :3] = direction
        out[e, 3:7] = quat
        out[e, 7] = 1.0
    return out
