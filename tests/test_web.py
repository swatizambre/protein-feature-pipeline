"""Web API smoke tests (requires fastapi + httpx)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from protein_features.web.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_run_rejects_non_pdb(client):
    r = client.post("/api/run", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_run_synthetic_pdb(client, tmp_path):
    from protein_features import synthetic as mk

    seq = "MKTAYIA"
    phis = [-60.0] * len(seq)
    psis = [-45.0] * len(seq)
    pdb_path = tmp_path / "tiny.pdb"
    mk.write_pdb(seq, mk.build(seq, phis, psis), str(pdb_path))
    with open(pdb_path, "rb") as f:
        r = client.post("/api/run", files={"file": ("tiny.pdb", f, "chemical/x-pdb")})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["num_residues"] == len(seq)
    assert data["validation"]["passed"] is True
    assert "encoded.npz" in data["downloads"]
    dl = client.get(data["downloads"]["encoded.npz"])
    assert dl.status_code == 200
    assert len(dl.content) > 0


def test_upload_size_limit(client, monkeypatch):
    from protein_features.core import constants as C

    monkeypatch.setattr(C, "MAX_UPLOAD_BYTES", 64)
    r = client.post(
        "/api/run",
        files={"file": ("big.pdb", b"ATOM" + b"x" * 200, "chemical/x-pdb")},
    )
    assert r.status_code == 413


def test_resolve_listen_port_fail_fast_when_busy():
    import socket

    from protein_features.web.app import port_available, resolve_listen_port

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    host, port = holder.getsockname()[:2]
    try:
        assert port_available(host, port) is False
        with pytest.raises(SystemExit) as ei:
            resolve_listen_port(host, port, auto_port=False)
        assert "already in use" in str(ei.value)
        next_port = resolve_listen_port(host, port, auto_port=True)
        assert next_port != port
        assert port_available(host, next_port) is True
    finally:
        holder.close()


def test_resolve_listen_port_uses_preferred_when_free():
    from protein_features.web.app import resolve_listen_port

    # Ephemeral bind to discover a free port, then release and resolve it.
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()[:2]
    probe.close()
    assert resolve_listen_port(host, port, auto_port=False) == port
