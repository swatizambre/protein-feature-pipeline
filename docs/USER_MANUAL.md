# User Manual — Backend & Frontend

How to install and run the **CLI backend** and the **web frontend + API**.
Python **3.9+** required. Commands assume you are at the repo root.

---

## 1. One-time setup

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -e ".[all]"
```

Check that console scripts are available:

```bash
protein-features --help
protein-features-web --help
```

---

## 2. Backend (CLI pipeline)

The backend is the extract → encode → decode → validate codec.
No separate database or cloud service is required.

### Full pipeline

```bash
protein-features --pdb data/7rfw.pdb
```

Default output: `output/pipeline_<name>/`

| File | Meaning |
|------|---------|
| `encoded.npz` | Node / edge tensors |
| `schema.json` | Column layout + ids |
| `validation_report.json` | Round-trip checks |
| `features.png` | Summary plot (if viz installed) |

Useful flags:

```bash
protein-features --pdb path/to/file.pdb --out output/my_run
protein-features --pdb data/7rfw.pdb --no-viz
protein-features --pdb data/7rfw.pdb -v          # debug logs
```

### Other backend commands

```bash
# Stepwise EDA report + figures/
protein-features-analyze --pdb data/7rfw.pdb
# → output/stepwise_<name>/

# Pocket + ligand path
protein-features-pocket --pdb data/7rfw.pdb
# → output/pocket_<name>/

# Regenerate committed demos
make run analyze pocket
# → examples/pipeline_7rfw/, examples/stepwise_7rfw/, examples/pocket_7rfw/
```

### Library API (same backend in Python)

```python
from protein_features import extract, encode, decode, roundtrip_report

protein = extract("data/7rfw.pdb")
encoded = encode(protein)
decoded = decode(encoded)
print(roundtrip_report(protein, decoded, encoded))
```

### Tests

```bash
pytest -q
```

---

## 3. Frontend + API (web console)

Frontend and API run from **one process**. The UI is static HTML served by
FastAPI; the browser calls the same host’s REST endpoints.

### Start the server

```bash
protein-features-web
# Open http://127.0.0.1:8000
```

If port **8000** is busy:

```bash
protein-features-web --port 8001
protein-features-web --auto-port   # pick next free port
```

Stop with `Ctrl+C`.

### Use the UI (frontend)

1. Open **http://127.0.0.1:8000** in a browser.  
2. Confirm header shows **ONLINE** (health check).  
3. Choose or drag-drop a `.pdb` / `.ent` file.  
4. Click **Execute pipeline**.  
5. Review metrics and round-trip **PASS / FAIL**.  
6. Download artifacts (tensors, schema, validation report, plot).

Processing is **local** to your machine. Job folders are temporary and purged
after the configured TTL.

UI source: `src/protein_features/web/static/index.html`.

### Call the API (backend HTTP)

Base URL: `http://127.0.0.1:8000`

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Liveness + upload limit |
| `POST` | `/api/run` | Run pipeline on uploaded PDB |
| `GET` | `/api/download/{job_id}/{filename}` | Download artifact |

**Health**

```bash
curl http://127.0.0.1:8000/api/health
# {"status":"ok","max_upload_mb":50}
```

**Run pipeline**

```bash
curl -X POST http://127.0.0.1:8000/api/run \
  -F "file=@data/7rfw.pdb"
```

Response includes `job_id`, shapes, `validation`, and `downloads` URLs.

**Download** (example filenames):  
`encoded.npz` · `schema.json` · `validation_report.json` · `features.png`

```bash
curl -O http://127.0.0.1:8000/api/download/<job_id>/validation_report.json
```

| HTTP code | Meaning |
|-----------|---------|
| `400` | Bad or unreadable file |
| `413` | File over upload limit |
| `404` | Unknown job / filename |
| `500` | Internal pipeline error |

Interactive docs (FastAPI): **http://127.0.0.1:8000/docs**

---

## 4. Backend vs frontend at a glance

| Layer | How to run | What it does |
|-------|------------|--------------|
| Backend CLI | `protein-features …` | Pipeline on disk; writes `output/` or `--out` |
| Backend API | inside `protein-features-web` | Same pipeline via `POST /api/run` |
| Frontend UI | browser → `:8000` | Upload form, status, downloads |

You do **not** need two terminals for web: one `protein-features-web` process
serves both UI and API.

---

## 5. Troubleshooting

| Problem | Fix |
|---------|-----|
| `protein-features-web` / command not found | Activate `.venv`, re-run `pip install -e ".[all]"` |
| Port 8000 in use | `--port 8001` or `--auto-port` |
| No plot in UI | Install viz extras (`.[all]`); pipeline still returns tensors |
| Upload rejected | Use `.pdb`/`.ent`; stay under `/api/health` → `max_upload_mb` |
| Old UI after edits | Hard-refresh the browser (Ctrl+F5) |

---

## 6. Related docs

| Doc | Content |
|-----|---------|
| [`../README.md`](../README.md) | Feature design, encode/decode, results |
| This file | How to run CLI backend + web UI/API |

Other files under `docs/` (reports, walkthrough, references) are local-only and
are not part of the published repository.
