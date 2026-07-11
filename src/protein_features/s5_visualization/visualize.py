"""Stage 5: 4-panel feature summary plot.

Ramachandran by SS, residue composition, burial along the chain, and kNN
edge-distance histogram. Uses the Agg backend so it runs without a display.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..s1_feature_extraction.extractor import ExtractedProtein


def plot_summary(prot: ExtractedProtein, encoded: dict, out_path: str) -> str:
    """Write a 4-panel feature summary to ``out_path`` and return that path."""
    fig = plt.figure(figsize=(13, 9))

    # 1) Ramachandran (phi/psi coloured by SS)
    # Colours: red=H (helix), blue=E (sheet), grey=C (coil). Helix sits in the
    # classic lower-left basin — legend order is H/E/C, not a colour gradient.
    ax1 = fig.add_subplot(2, 2, 1)
    phi, psi = np.degrees(prot.phi), np.degrees(prot.psi)
    colours = {"H": "tab:red", "E": "tab:blue", "C": "tab:gray"}
    labels = {"H": "H helix", "E": "E sheet", "C": "C coil"}
    for s in ("H", "E", "C"):
        m = np.array([x == s for x in prot.ss])
        ax1.scatter(phi[m], psi[m], s=14, c=colours[s], label=labels[s], alpha=0.8)
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    ax1.axhline(0, lw=0.5, c="k")
    ax1.axvline(0, lw=0.5, c="k")
    ax1.set_xlabel("phi (deg)")
    ax1.set_ylabel("psi (deg)")
    ax1.set_title("Ramachandran (coloured by 3-state SS)")
    ax1.legend(fontsize=8, title="SS")

    # 2) Amino-acid counts
    ax2 = fig.add_subplot(2, 2, 2)
    aas, counts = np.unique(prot.one_letter, return_counts=True)
    ax2.bar(aas, counts, color="tab:green")
    ax2.set_title("Residue-type composition")
    ax2.set_xlabel("amino acid")
    ax2.set_ylabel("count")

    # 3) Burial along the chain
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(prot.burial, color="tab:purple")
    ax3.set_title("Burial proxy (CA coordination number)")
    ax3.set_xlabel("residue index")
    ax3.set_ylabel("neighbours < 10 A")

    # 4) Neighbour-edge distance histogram
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(encoded["edge_dist"], bins=30, color="tab:orange")
    ax4.set_title("kNN edge-distance distribution")
    ax4.set_xlabel("CA-CA distance (A)")
    ax4.set_ylabel("edge count")

    fig.suptitle(f"Feature summary - {prot.num_residues} residues " f"({prot.source})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
