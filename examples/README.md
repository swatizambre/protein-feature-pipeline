# Example outputs

Committed results from `data/7rfw.pdb` so you can inspect artifacts without
re-running the pipeline. Keep demos here; use gitignored `output/` for scratch.

## Folders

| Folder | Command | What’s inside |
|--------|---------|----------------|
| `pipeline_7rfw/` | `protein-features --pdb data/7rfw.pdb --out examples/pipeline_7rfw` | `encoded.npz`, `schema.json`, `validation_report.json` (`"passed": true`), `features.png` |
| `stepwise_7rfw/` | `protein-features-analyze --pdb data/7rfw.pdb --out examples/stepwise_7rfw` | `ANALYSIS_REPORT.md`, `analysis_stats.json`, `figures/s1_….png` … |
| `pocket_7rfw/` | `protein-features-pocket --pdb data/7rfw.pdb --out examples/pocket_7rfw` | `complex_encoded.npz`, `complex_summary.json` (4WI + Cys145), `pocket.png` |

Regenerate all three:

```bash
make run analyze pocket
```

## Layout (stepwise)

```text
examples/stepwise_7rfw/
  ANALYSIS_REPORT.md
  analysis_stats.json
  figures/
    s1_composition.png
    s2_continuous.png
    s3_dihedrals.png
    s4_ss.png
    s5_corr.png
    s6_burial_hydropathy.png
    s7_burial_by_ss.png
    s9_edge.png
    s10_pca.png
    s14_mi.png
```

## Scratch vs demos

| Path | Purpose |
|------|---------|
| `examples/` | Committed demo outputs |
| `output/` | Gitignored local runs when you omit `--out` |

Quick checks without install: open `pipeline_7rfw/validation_report.json` and
`pocket_7rfw/complex_summary.json`.
