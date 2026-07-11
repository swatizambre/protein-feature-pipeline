"""FastAPI app: pick a PDB in the browser (or POST it) and run the pipeline."""

import errno
import json
import logging
import shutil
import socket
import tempfile
import time
import uuid
from pathlib import Path

from protein_features.core import constants as C
from protein_features.core.exceptions import (
    InvalidPDBError,
    ProteinFeatureError,
    ResourceLimitError,
    SchemaError,
)
from protein_features.core.io import report_to_jsonable, save_encoded

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
JOBS_ROOT = Path(tempfile.gettempdir()) / "protein_features_jobs"
ALLOWED_DOWNLOADS = frozenset(
    {"encoded.npz", "schema.json", "validation_report.json", "features.png"}
)


def _purge_old_jobs(root: Path = JOBS_ROOT, ttl: int = C.JOB_TTL_SECONDS) -> None:
    """Delete job directories older than ``ttl`` seconds."""
    if not root.is_dir():
        return
    now = time.time()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            age = now - child.stat().st_mtime
            if age > ttl:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def _read_upload_capped(upload_file, dest: Path, max_bytes: int) -> int:
    """Copy upload to ``dest``; raise ResourceLimitError if over ``max_bytes``."""
    written = 0
    chunk = 1024 * 1024
    with open(dest, "wb") as out:
        while True:
            block = upload_file.file.read(chunk)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise ResourceLimitError(
                    f"Upload exceeds {max_bytes // (1024 * 1024)} MB limit"
                )
            out.write(block)
    return written


def create_app():
    """Build the FastAPI application (lazy import so core package stays light)."""
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from protein_features import configure_logging, decode, encode, extract, roundtrip_report

    configure_logging(False)
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="Protein Feature Pipeline",
        description=(
            "Production feature layer: upload a PDB, run extract → encode → "
            "decode → validate, and download deterministic tensors."
        ),
        version="1.1.0",
    )

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="UI page missing")
        return FileResponse(index_path)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "max_upload_mb": C.MAX_UPLOAD_BYTES // (1024 * 1024)}

    @app.post("/api/run")
    async def run_pipeline(file: UploadFile = File(...)):
        _purge_old_jobs()
        name = file.filename or "upload.pdb"
        if not name.lower().endswith((".pdb", ".ent")):
            raise HTTPException(
                status_code=400,
                detail="Please upload a .pdb (or .ent) file",
            )

        job_id = uuid.uuid4().hex
        job_dir = JOBS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = job_dir / Path(name).name

        try:
            _read_upload_capped(file, pdb_path, C.MAX_UPLOAD_BYTES)

            prot = extract(str(pdb_path))
            enc = encode(prot)
            dec = decode(enc)
            report = report_to_jsonable(roundtrip_report(prot, dec, enc))

            save_encoded(job_dir, enc)
            with open(job_dir / "validation_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            files = ["encoded.npz", "schema.json", "validation_report.json"]
            plot_ok = False
            try:
                from protein_features.s5_visualization.visualize import plot_summary

                plot_summary(prot, enc, str(job_dir / "features.png"))
                files.append("features.png")
                plot_ok = True
            except Exception as e:
                log.warning("Visualisation skipped: %s", e)

            downloads = {fn: f"/api/download/{job_id}/{fn}" for fn in files}
            return {
                "job_id": job_id,
                "filename": name,
                "num_residues": int(prot.num_residues),
                "ss_source": prot.ss_source,
                "meta": prot.meta,
                "shapes": {
                    "node_features": list(enc["node_features"].shape),
                    "coords": list(enc["coords"].shape),
                    "edge_index": list(enc["edge_index"].shape),
                    "edge_features": list(enc["edge_features"].shape),
                },
                "validation": report,
                "plot": plot_ok,
                "downloads": downloads,
            }
        except ResourceLimitError as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=413, detail=str(e)) from e
        except (InvalidPDBError, SchemaError, ValueError) as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ProteinFeatureError as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception:
            log.exception("Pipeline failed for job %s", job_id)
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500, detail="Internal pipeline error; see server logs"
            ) from None

    @app.get("/api/download/{job_id}/{filename}")
    def download(job_id: str, filename: str):
        if filename not in ALLOWED_DOWNLOADS:
            raise HTTPException(status_code=404, detail="Unknown file")
        if not job_id.isalnum() or len(job_id) != 32:
            raise HTTPException(status_code=404, detail="Unknown job")
        path = JOBS_ROOT / job_id / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        media = "image/png" if filename.endswith(".png") else "application/octet-stream"
        return FileResponse(path, media_type=media, filename=filename)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = None

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
_PORT_SCAN_LIMIT = 50


def port_available(host: str, port: int) -> bool:
    """Return True if ``(host, port)`` can be bound for listening."""
    if not (0 < port < 65536):
        return False
    # Prefer IPv4 for typical 127.0.0.1 / 0.0.0.0 binds; IPv6 literal otherwise.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        # Do not set SO_REUSEADDR: we want a real "is this free?" check.
        if family == socket.AF_INET6:
            sock.bind((host, port, 0, 0))
        else:
            sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def resolve_listen_port(
    host: str,
    preferred: int,
    *,
    auto_port: bool = False,
    scan_limit: int = _PORT_SCAN_LIMIT,
) -> int:
    """Pick a listen port.

    Production default: fail fast if ``preferred`` is taken (never silently
    remaps ports under a load balancer / reverse proxy). Local convenience:
    pass ``auto_port=True`` to scan upward for the next free port.
    """
    if port_available(host, preferred):
        return preferred

    hint = (
        f"Port {preferred} on {host} is already in use.\n"
        f"  Stop the other process, or choose another port:\n"
        f"    protein-features-web --port {preferred + 1}\n"
        f"    protein-features-web --auto-port"
    )
    if not auto_port:
        raise SystemExit(hint)

    for port in range(preferred + 1, preferred + 1 + scan_limit):
        if port_available(host, port):
            log.warning(
                "Port %s on %s is busy; binding to %s instead (--auto-port)",
                preferred,
                host,
                port,
            )
            return port
    raise SystemExit(
        f"{hint}\n"
        f"  Also scanned {scan_limit} ports above {preferred}; none were free."
    )


def main(argv=None):
    """Start the local web server (protein-features-web)."""
    import argparse

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            'Web deps missing. Install with: pip install -e ".[web,viz]"'
        ) from e

    ap = argparse.ArgumentParser(description="Protein feature pipeline web UI")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"bind address (default {DEFAULT_HOST})")
    ap.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"listen port (default {DEFAULT_PORT})",
    )
    ap.add_argument(
        "--auto-port",
        action="store_true",
        help="if --port is busy, use the next free port instead of exiting",
    )
    args = ap.parse_args(argv)

    if not (0 < args.port < 65536):
        raise SystemExit(f"Invalid --port {args.port}; expected 1–65535")

    port = resolve_listen_port(args.host, args.port, auto_port=args.auto_port)

    global app
    app = create_app()
    url = f"http://{args.host}:{port}"
    print(f"Open {url}  (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host=args.host, port=port, log_level="info")
    except OSError as e:
        # Race: port was free at check time, taken before bind.
        winerror = getattr(e, "winerror", None)
        busy = e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", 10048)) or winerror == 10048
        if busy:
            raise SystemExit(
                f"Failed to bind {args.host}:{port} (address already in use).\n"
                f"  Retry with: protein-features-web --port {port + 1}\n"
                f"           or: protein-features-web --auto-port"
            ) from e
        raise


if __name__ == "__main__":
    main()
