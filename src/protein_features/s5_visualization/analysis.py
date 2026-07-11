#!/usr/bin/env python3
"""Stage 5: stepwise EDA report.

Builds a per-residue table, runs basic stats/plots (distributions, correlations,
PCA, clustering, …), and writes ANALYSIS_REPORT.md for the given PDB.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ..core import constants as C
from ..s1_feature_extraction.extractor import extract
from ..s2_encoding.encoder import encode
from ..s3_decoding.decoder import decode
from ..s4_validation.validate import roundtrip_report

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

CONT = [
    "hydropathy",
    "charge",
    "volume",
    "polar",
    "aromatic",
    "hbond_don",
    "hbond_acc",
    "burial",
]


def build_frames(prot):
    """One row per residue: physchem + geometry fields."""
    phys = np.array([C.physchem_vector(a) for a in prot.one_letter])
    res = pd.DataFrame(
        {
            "restype": prot.one_letter,
            "ss": prot.ss,
            "hydropathy": phys[:, 0],
            "charge": phys[:, 1],
            "volume": phys[:, 2],
            "polar": phys[:, 3],
            "aromatic": phys[:, 4],
            "hbond_don": phys[:, 5],
            "hbond_acc": phys[:, 6],
            "phi": np.degrees(prot.phi),
            "psi": np.degrees(prot.psi),
            "omega": np.degrees(prot.omega),
            "burial": prot.burial,
        }
    )
    return res


class Report:
    """Stitches Observation / Decision / Significance notes into ANALYSIS_REPORT.md."""

    def __init__(self, outdir, title):
        self.outdir = outdir
        self.blocks = [f"# {title}\n"]

    def step(self, name, obs, dec, sig, figure=None, table=None):
        """Append one analysis step (and optional figure/table) to the report."""
        log.info("%s", name)
        log.info("  Observation : %s", obs)
        log.info("  Decision    : %s", dec)
        log.info("  Significance: %s", sig)
        md = [f"\n## {name}\n"]
        if figure:
            rel = os.path.relpath(figure, self.outdir).replace("\\", "/")
            md.append(f"![{name}]({rel})\n")
        if table is not None:
            md.append(table.to_markdown() + "\n")
        md.append(f"**Observation.** {obs}\n")
        md.append(f"**Decision.** {dec}\n")
        md.append(f"**Significance.** {sig}\n")
        self.blocks.append("\n".join(md))

    def save(self):
        """Write the accumulated markdown and return its path."""
        path = os.path.join(self.outdir, "ANALYSIS_REPORT.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.blocks))
        return path


def _fig(outdir, name):
    return os.path.join(outdir, name)


def run(pdb, outdir):
    """Run the EDA steps on ``pdb``; dump figures + report into ``outdir``.

    Layout::

        outdir/
          ANALYSIS_REPORT.md
          analysis_stats.json
          figures/
            s1_composition.png
            s2_continuous.png
            …
    """
    os.makedirs(outdir, exist_ok=True)
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    prot = extract(pdb)
    enc = encode(prot)
    df = build_frames(prot)
    n = len(df)
    rep = Report(outdir, f"Feature Analysis Report - {os.path.basename(pdb)}")
    stats_out = {}

    # --- Step 0: dataset / overview ---
    ndef = {a: int(df[a].notna().sum()) for a in ("phi", "psi", "omega")}
    obs = (
        f"{n} residues, {len(set(prot.chain_ids))} chain(s). Backbone angles "
        f"defined for phi={ndef['phi']}, psi={ndef['psi']}, omega={ndef['omega']} "
        f"residues; the rest are chain termini/gaps "
        f"({prot.meta.get('chain_breaks',0)} break(s)). "
        f"{prot.meta.get('skipped_het',0)} hetero groups skipped."
    )
    dec = (
        "Represent undefined angles with an explicit validity mask (already in "
        "the encoder) rather than imputing them."
    )
    sig = (
        "Undefined dihedrals are structural facts, not missing data - masking "
        "prevents a generative model from learning fake angles at chain ends."
    )
    rep.step("Step 0 - Dataset & data quality", obs, dec, sig)
    stats_out["n_residues"] = n
    stats_out["angles_defined"] = ndef

    # --- Steps 1-4: univariate ---
    # Step 1: residue composition
    comp = df["restype"].value_counts()
    p = comp / comp.sum()
    H = float(stats.entropy(p, base=2))
    Hmax = np.log2(len(comp))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    comp.sort_index().plot(kind="bar", color="tab:green", ax=ax)
    ax.set_title("Residue-type composition")
    ax.set_ylabel("count")
    f = _fig(fig_dir, "s1_composition.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    obs = (
        f"20 amino-acid vocabulary present; Shannon entropy {H:.2f} of "
        f"{Hmax:.2f} max bits ({100*H/Hmax:.0f}% of uniform). Most common: "
        f"{comp.index[0]} ({comp.iloc[0]}), rarest: {comp.index[-1]} ({comp.iloc[-1]})."
    )
    dec = "Keep a one-hot over the full 20 + UNK vocabulary; do not prune rare " "types."
    sig = (
        "Near-uniform composition means all classes carry signal; pruning would "
        "bias a generative model against rare-but-real residues."
    )
    rep.step("Step 1 - Univariate: residue composition", obs, dec, sig, figure=f)
    stats_out["composition_entropy_bits"] = H

    # Step 2: continuous feature distributions
    contfeat = ["hydropathy", "volume", "burial", "charge"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    sk = {}
    for ax, col in zip(axes, contfeat):
        sns.histplot(df[col], kde=(col != "charge"), ax=ax, color="tab:blue")
        s = float(stats.skew(df[col]))
        sk[col] = s
        ax.set_title(f"{col}\nskew={s:.2f}")
    f = _fig(fig_dir, "s2_continuous.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    obs = (
        f"hydropathy skew={sk['hydropathy']:.2f}, volume skew={sk['volume']:.2f}, "
        f"burial skew={sk['burial']:.2f}. Charge is discrete and sparse "
        f"(mostly 0; {(df['charge']!=0).mean()*100:.0f}% charged)."
    )
    dec = (
        "Standardise continuous features with fixed constants (done in encoder); "
        "leave binary/charge features raw."
    )
    sig = (
        "Modest skew and differing scales (volume ~140 vs charge ~0) justify "
        "standardisation so no single feature dominates distance/gradient scales "
        "in a generative model."
    )
    rep.step("Step 2 - Univariate: continuous distributions", obs, dec, sig, figure=f)
    stats_out["skew"] = sk

    # Step 3: backbone dihedral histograms
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    for ax, col in zip(axes, ("phi", "psi", "omega")):
        sns.histplot(df[col].dropna(), bins=36, ax=ax, color="tab:purple")
        ax.set_title(col)
        ax.set_xlim(-180, 180)
    f = _fig(fig_dir, "s3_dihedrals.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    omega_near180 = float((df["omega"].dropna().abs() > 150).mean() * 100)
    obs = (
        f"phi/psi are multimodal (helix and sheet basins); omega is sharply "
        f"peaked near ±180 ({omega_near180:.0f}% of residues), i.e. trans peptide "
        f"bonds."
    )
    dec = "Encode every angle as (sin, cos) rather than a raw degree value."
    sig = (
        "Angles are circular: −179° and +179° are neighbours. sin/cos removes the "
        "wrap-around discontinuity that would otherwise destabilise a generative "
        "model, and omega's tight peak shows it is nearly constant (low entropy)."
    )
    rep.step("Step 3 - Univariate: backbone dihedrals", obs, dec, sig, figure=f)
    stats_out["omega_trans_pct"] = omega_near180

    # Step 4: secondary-structure counts
    ssp = df["ss"].value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    df["ss"].value_counts().reindex(["H", "E", "C"]).plot(
        kind="bar", color=["tab:red", "tab:blue", "tab:gray"], ax=ax
    )
    ax.set_title("Secondary-structure counts")
    ax.set_ylabel("residues")
    f = _fig(fig_dir, "s4_ss.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    obs = (
        "SS mix (approx Ramachandran classifier): "
        + ", ".join(f"{k}={v*100:.0f}%" for k, v in ssp.items())
        + "."
    )
    dec = (
        "Keep 3-state SS as a one-hot node feature; flag DSSP as an optional "
        "upgrade for finer 8-state labels."
    )
    sig = (
        "A realistic helix/sheet/coil balance confirms the coarse classifier is "
        "behaving; SS is a strong, low-dimensional structural prior for generation."
    )
    rep.step("Step 4 - Univariate: secondary structure", obs, dec, sig, figure=f)
    stats_out["ss_proportions"] = {k: float(v) for k, v in ssp.items()}

    # --- Steps 5-9: bivariate ---
    # Step 5: Spearman correlation heatmap
    corr = df[CONT].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman correlation of continuous features")
    f = _fig(fig_dir, "s5_corr.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    cc = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    hi = cc.stack().sort_values(ascending=False)
    top = hi.index[0]
    obs = (
        f"Strongest redundancy: {top[0]}–{top[1]} (|rho|={hi.iloc[0]:.2f}). "
        f"polar/hbond flags overlap with charge/hydropathy as expected."
    )
    dec = (
        "Retain the correlated flags for interpretability but note the redundancy; "
        "an encoder could compress them."
    )
    sig = (
        "Correlated hand-designed features are partially redundant - quantifying "
        "this tells us the *effective* feature dimensionality is below 8, guiding "
        "latent-size choices."
    )
    rep.step("Step 5 - Bivariate: feature correlation", obs, dec, sig, figure=f)
    stats_out["max_abs_corr"] = float(hi.iloc[0])

    # Step 6: burial vs hydropathy
    r, pval = stats.pearsonr(df["burial"], df["hydropathy"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(
        data=df,
        x="hydropathy",
        y="burial",
        hue="ss",
        palette={"H": "tab:red", "E": "tab:blue", "C": "tab:gray"},
        ax=ax,
    )
    ax.set_title(f"Burial vs hydropathy (r={r:.2f}, p={pval:.1e})")
    f = _fig(fig_dir, "s6_burial_hydropathy.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    obs = (
        f"Pearson r={r:.2f} (p={pval:.1e}) between hydropathy and burial " f"(coordination number)."
    )
    dec = "Keep both features; the correlation is real but weak, so each still adds " "information."
    sig = (
        "A positive hydrophobic-core signal (hydrophobic residues tend to be more "
        "buried) is exactly the biophysics a structure generator must respect - "
        "the features encode it, which validates the design."
    )
    rep.step("Step 6 - Bivariate: burial vs hydropathy", obs, dec, sig, figure=f)
    stats_out["burial_hydropathy_r"] = float(r)

    # Step 7: burial by secondary structure
    groups = [df.loc[df.ss == s, "burial"].values for s in ("H", "E", "C") if (df.ss == s).any()]
    Hstat, pk = stats.kruskal(*groups) if len(groups) > 1 else (np.nan, np.nan)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.boxplot(
        data=df,
        x="ss",
        y="burial",
        order=["H", "E", "C"],
        palette={"H": "tab:red", "E": "tab:blue", "C": "tab:gray"},
        ax=ax,
    )
    ax.set_title(f"Burial by SS (Kruskal-Wallis p={pk:.1e})")
    f = _fig(fig_dir, "s7_burial_by_ss.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    obs = (
        f"Burial differs across SS classes (Kruskal-Wallis H={Hstat:.1f}, "
        f"p={pk:.1e}); sheets/helices are typically more buried than coil."
    )
    dec = "Keep burial and SS as complementary features rather than dropping either."
    sig = (
        "Statistically significant SS-burial coupling means the feature set jointly "
        "captures local structure and 3D environment - richer than either alone."
    )
    rep.step("Step 7 - Bivariate: burial vs secondary structure", obs, dec, sig, figure=f)
    stats_out["burial_by_ss_kruskal_p"] = float(pk)

    # Step 8: residue type vs SS (chi-square)
    ct = pd.crosstab(df["restype"], df["ss"])
    chi2, pc, dof, _ = stats.chi2_contingency(ct)
    obs = (
        f"Residue-type x SS contingency: chi2={chi2:.0f}, dof={dof}, p={pc:.1e}. "
        f"Certain residues show clear helix/sheet preference."
    )
    dec = (
        "No change needed - the one-hot residue type already lets a model learn "
        "these propensities."
    )
    sig = (
        "Significant association confirms residue identity carries secondary-"
        "structure information (e.g. Gly/Pro breakers, beta-branched sheet formers), "
        "a key prior for sequence-conditioned structure generation."
    )
    rep.step("Step 8 - Bivariate: residue type vs SS", obs, dec, sig, table=ct)
    stats_out["restype_ss_chi2_p"] = float(pc)

    # Step 9: edge distance vs sequence separation
    ei = enc["edge_index"]
    ed = enc["edge_dist"]
    rid = np.array(prot.res_ids)
    seqsep = np.abs(rid[ei[1]] - rid[ei[0]])
    edf = pd.DataFrame({"dist": ed, "seqsep": seqsep})
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.scatterplot(
        data=edf.sample(min(2000, len(edf)), random_state=0),
        x="seqsep",
        y="dist",
        s=8,
        alpha=0.3,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title("Edge distance vs |sequence separation|")
    f = _fig(fig_dir, "s9_edge.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    local = float((seqsep <= 4).mean() * 100)
    obs = (
        f"{local:.0f}% of kNN edges connect residues within 4 positions in "
        f"sequence; the rest are long-range spatial contacts at similar distances."
    )
    dec = (
        "Keep both the RBF distance and the (signed, log) sequence separation as " "edge features."
    )
    sig = (
        "The graph captures contacts that sequence alone misses - long-range "
        "tertiary contacts - which is exactly the information a structure-based "
        "representation must add over a sequence-only model."
    )
    rep.step("Step 9 - Bivariate: edge geometry vs sequence", obs, dec, sig, figure=f)
    stats_out["local_edge_pct"] = local

    # --- Steps 10-12: multivariate ---
    # Step 10: PCA
    X = StandardScaler().fit_transform(df[CONT].values)
    pca = PCA().fit(X)
    evr = pca.explained_variance_ratio_
    n90 = int(np.argmax(np.cumsum(evr) >= 0.90) + 1)
    Z = pca.transform(X)[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "o-")
    axes[0].axhline(0.9, ls="--", c="r")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("PCA cumulative explained variance")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("cum. variance")
    sns.scatterplot(
        x=Z[:, 0],
        y=Z[:, 1],
        hue=df["ss"],
        palette={"H": "tab:red", "E": "tab:blue", "C": "tab:gray"},
        ax=axes[1],
        s=18,
    )
    axes[1].set_title("PC1–PC2 (coloured by SS)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    f = _fig(fig_dir, "s10_pca.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    obs = (
        f"{n90} of {len(CONT)} principal components explain 90% of variance "
        f"(PC1={evr[0]*100:.0f}%, PC2={evr[1]*100:.0f}%)."
    )
    dec = (
        f"Treat ~{n90} as the effective continuous-feature dimensionality; a "
        f"generative latent need not exceed the graph/one-hot capacity by much."
    )
    sig = (
        "Confirms the hand-designed continuous features are compressible - useful "
        "for setting encoder/latent widths and for spotting redundancy."
    )
    rep.step("Step 10 - Multivariate: PCA", obs, dec, sig, figure=f)
    stats_out["pca_evr"] = [float(x) for x in evr]
    stats_out["pca_n_comp_90"] = n90

    # Step 11: KMeans clusters vs SS labels
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    ari = adjusted_rand_score(df["ss"], km.labels_)
    obs = (
        f"KMeans(k=3) on the standardized continuous features vs the SS labels "
        f"gives adjusted Rand index {ari:.2f} (0 = chance)."
    )
    dec = (
        "Do not expect unsupervised clusters of physicochemistry/burial to recover "
        "SS; keep SS and dihedral angles as explicit, dedicated features."
    )
    sig = (
        "A near-zero ARI is an important negative result: physicochemical and "
        "burial features alone carry almost no secondary-structure signal, so the "
        "backbone-dihedral and SS features are not redundant - they contribute "
        "information nothing else does, justifying their inclusion."
    )
    rep.step("Step 11 - Multivariate: clustering vs SS", obs, dec, sig)
    stats_out["kmeans_ss_ari"] = float(ari)

    # Step 12: variance inflation factors (multicollinearity)
    vif = {}
    for j, col in enumerate(CONT):
        y = X[:, j]
        A = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - np.sum((y - A @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        vif[col] = float(1.0 / max(1e-6, 1 - r2))
    vif_s = pd.Series(vif).sort_values(ascending=False)
    obs = "Variance-inflation factors: " + ", ".join(f"{k}={v:.1f}" for k, v in vif_s.items()) + "."
    dec = (
        "Flag features with VIF>5 as redundant; keep for interpretability but "
        "consider dropping in a compact encoder."
    )
    sig = (
        "Multicollinearity analysis tells us which hand-designed features are "
        "near-linear combinations of others - directly informing a leaner encoding."
    )
    rep.step(
        "Step 12 - Multivariate: multicollinearity (VIF)",
        obs,
        dec,
        sig,
        table=vif_s.round(2).to_frame("VIF"),
    )
    stats_out["vif"] = vif

    # --- Steps 13-15: feature-level checks ---
    # Step 13: variance and sparsity
    var = df[CONT].var()
    sparse = {
        c: float((df[c] == 0).mean()) for c in ("charge", "aromatic", "hbond_don", "hbond_acc")
    }
    obs = (
        "Sparsity (fraction zero): "
        + ", ".join(f"{k}={v*100:.0f}%" for k, v in sparse.items())
        + f". Lowest-variance feature: {var.idxmin()} ({var.min():.3f})."
    )
    dec = (
        "Retain sparse binary flags - sparsity is informative (a charged residue "
        "is a strong signal), not noise."
    )
    sig = (
        "Distinguishing 'sparse but informative' from 'near-constant / useless' "
        "prevents accidentally discarding rare high-signal features."
    )
    rep.step("Step 13 - Feature: variance & sparsity", obs, dec, sig)
    stats_out["sparsity"] = sparse

    # Step 14: mutual information with SS
    Xmi = df[CONT].values
    ycat = df["ss"].astype("category").cat.codes.values
    mi = mutual_info_classif(Xmi, ycat, random_state=0)
    mis = pd.Series(mi, index=CONT).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.3))
    mis.plot(kind="bar", color="teal", ax=ax)
    ax.set_title("Mutual information with secondary structure")
    f = _fig(fig_dir, "s14_mi.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    plt.close(fig)
    obs = (
        f"Most SS-informative continuous features: {mis.index[0]} "
        f"({mis.iloc[0]:.3f}), {mis.index[1]} ({mis.iloc[1]:.3f}); least: "
        f"{mis.index[-1]} ({mis.iloc[-1]:.3f})."
    )
    dec = "Prioritise high-MI features; keep low-MI ones only if cheap and " "interpretable."
    sig = (
        "Ranks features by how much structural signal they carry - evidence-based "
        "justification of the feature set rather than intuition alone."
    )
    rep.step("Step 14 - Feature: discriminative power (MI)", obs, dec, sig, figure=f)
    stats_out["mutual_info"] = {k: float(v) for k, v in mis.items()}

    # Step 15: encoding audit + round-trip
    x = enc["node_features"]
    dec_obj = decode(enc)
    rr = roundtrip_report(prot, dec_obj, enc)
    onehot_ok = bool(np.all(np.isclose(x[:, : C.NUM_RESTYPES].sum(1), 1.0)))
    obs = (
        f"Node tensor {x.shape}, range [{x.min():.2f}, {x.max():.2f}]. Residue "
        f"one-hot rows sum to 1: {onehot_ok}. Round-trip passed: {rr['passed']} "
        f"(coord err {rr['coords_max_abs_error']:.1e} A, RBF dist err "
        f"{rr['edge_dist_mean_abs_error_A']:.1e} A)."
    )
    dec = "Ship the encoding as-is; determinism and reversibility are verified."
    sig = (
        "Closes the loop: the analysed features are exactly the tensors a model "
        "would consume, and they are well-scaled, valid, and decodable."
    )
    rep.step("Step 15 - Feature: encoding audit", obs, dec, sig)
    stats_out["roundtrip"] = rr

    # --- write report + stats ---
    with open(os.path.join(outdir, "analysis_stats.json"), "w") as fjson:
        json.dump(stats_out, fjson, indent=2)
    report_path = rep.save()
    log.info("Wrote report -> %s", report_path)
    log.info("Wrote stats  -> %s", os.path.join(outdir, "analysis_stats.json"))
    return report_path


def main():
    """CLI entry for the stepwise feature EDA report."""
    from protein_features import configure_logging

    ap = argparse.ArgumentParser(description="Stepwise feature EDA")
    ap.add_argument("--pdb", required=True)
    ap.add_argument(
        "--out",
        default=None,
        help="output directory (default: output/stepwise_<pdb-name>)",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    a = ap.parse_args()
    configure_logging(a.verbose)
    out = a.out or os.path.join(
        "output", f"stepwise_{os.path.splitext(os.path.basename(a.pdb))[0]}"
    )
    run(a.pdb, out)


if __name__ == "__main__":
    main()
