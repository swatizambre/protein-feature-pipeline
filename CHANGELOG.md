# Changelog

Notable changes. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is semantic.

## [1.1.0]

### Added
- Typed `EncodedProtein` return from `encode`, plus `save_encoded` / `load_encoded`.
- Schema/shape validation before `decode`; physchem restored on decode and
  checked in `roundtrip_report`.
- Typed errors (`InvalidPDBError`, `SchemaError`, `ResourceLimitError`).
- Web UI hardening: upload size cap, job TTL cleanup, `/api/health`, clearer
  HTTP status codes.
- Optional SciPy `cKDTree` path for kNN when SciPy is installed.
- `.[all]` extra and README Quick start for one-command setup.
- Broader tests (I/O, schema reject, 7RFW round-trip, web) and CI CLI smoke.

### Changed
- CLI exits non-zero when round-trip fails or the PDB is unusable.
- Example outputs under `examples/` refreshed on 7RFW.

## [1.0.0]

### Added
- Logging via the standard library: quiet module loggers by default, CLI console
  setup, `-v/--verbose` for DEBUG.
- **Binding-pocket + ligand extraction** (`protein_features.pocket`,
  `protein-features-pocket`): ligand atom graph (CONECT, optional RDKit SMILES),
  pocket atom graph, typed contacts, covalent-link detection. Checked on 7RFW
  (nirmatrelvir, Cys145 link, Mpro active site).
- DSSP-style H-bond secondary structure as the default (no external binary
  required). Real DSSP used when `mkdssp` + BioPython are available.
- Residue-level **extractor** (BioPython optional, numpy fallback): type, coords,
  dihedrals, SS, burial, physicochemical properties.
- Deterministic **encoder**: node tensor, coords, kNN graph, edge features (RBF
  distance, sequence separation, same-chain, relative orientation) + schema.
- **Decoder** with a clear objective: lossless for type, SS, angles, coords;
  bounded-lossy for RBF distances.
- Relative-orientation edge features from backbone frames (direction +
  quaternion).
- Optional real **DSSP**, with Ramachandran fallback if needed.
- Handling for messy PDBs: skip HETATM/water, gap-aware dihedrals, altLoc,
  multi-model, unknown residues.
- **Validation** (round-trip) and **visualisation**.
- Stepwise **EDA** (`protein-features-analyze`) with an auto-written report.
- Tests: round-trip, determinism, UNK bucket, chain gap, SE(3) invariance.
- Packaging (`pyproject.toml`, src layout, console entry points), CI, Makefile.
