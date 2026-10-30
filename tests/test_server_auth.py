"""Optional access token (Bearer or session cookie), per-IP upload limits, quota."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_agentic_rag.server.__main__ import _is_loopback, _parse_args, _upload_quota_from_env
from local_agentic_rag.server.app import SESSION_COOKIE, UploadLimiter
from tests.test_server import PDF, FakeLibrary, make_client, parse_sse, upload

TOKEN = "s3cret-token"
BEARER = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def lib(tmp_path: Path):
    store = tmp_path / "store"
    store.mkdir()
    fake = FakeLibrary(store)
    fake.add_ready("d1", "Annual report")
    yield fake
    fake.release.set()


@pytest.fixture
def secured(lib, tmp_path):
    with make_client(lib, tmp_path, token=TOKEN) as c:
        yield c


@pytest.fixture
def signed_in(secured):
    assert secured.post("/api/session", json={"token": TOKEN}).status_code == 204
    return secured


# ------------------------------------------------------------------ no token = as before


def test_without_token_nothing_changes(lib, tmp_path) -> None:
    with make_client(lib, tmp_path) as c:
        assert c.get("/api/documents").status_code == 200
        assert c.get("/api/health").status_code == 200
        assert c.get("/").status_code == 200
        # the session routes only exist in token mode
        assert c.post("/api/session", json={"token": "x"}).status_code == 404
        assert c.delete("/api/session").status_code == 404


def test_empty_token_means_no_auth(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, token="") as c:
        assert c.get("/api/documents").status_code == 200


# ------------------------------------------------------------------ token set


def test_api_requires_auth(secured) -> None:
    for method, url in [
        ("GET", "/api/documents"),
        ("GET", "/api/documents/d1"),
        ("GET", "/api/documents/d1/events"),
        ("GET", "/api/documents/d1/pages/1.png"),
        ("POST", "/api/documents"),
        ("POST", "/api/ask"),
        ("POST", "/api/ask/stream"),
        ("DELETE", "/api/documents/d1"),
        ("GET", "/api/nonexistent"),
    ]:
        r = secured.request(method, url)
        assert r.status_code == 401, (method, url)
        assert r.json() == {"detail": "Authentication required. Send the access token."}
        assert r.headers["www-authenticate"] == "Bearer"
        assert r.headers["x-content-type-options"] == "nosniff"  # headers middleware still wraps it
    assert secured.get("/api/documents").status_code == 401


def test_bearer_header_works(secured) -> None:
    r = secured.get("/api/documents", headers=BEARER)
    assert r.status_code == 200
    assert upload_with(secured, BEARER).status_code == 201


def upload_with(client: TestClient, headers: dict[str, str]):
    return client.post(
        "/api/documents", files={"file": ("a.pdf", PDF, "application/pdf")}, headers=headers
    )


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Bearer "},
        {"Authorization": TOKEN},  # no scheme
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": "Bearer s3cret-tokeN"},
        {"Authorization": "Bearer café".encode("latin-1")},  # must not crash the comparison
    ],
)
def test_wrong_bearer_is_401(secured, headers) -> None:
    assert secured.get("/api/documents", headers=headers).status_code == 401


def test_health_and_static_stay_public(secured) -> None:
    assert secured.get("/api/health").status_code == 200
    index = secured.get("/")
    assert index.status_code == 200 and "<html" in index.text
    assert secured.get("/app.js").status_code == 200
    assert secured.get("/styles.css").status_code == 200


def test_wrong_session_token_is_401_and_sets_no_cookie(secured) -> None:
    r = secured.post("/api/session", json={"token": "nope"})
    assert r.status_code == 401
    assert r.json() == {"detail": "Wrong access token."}
    assert "set-cookie" not in r.headers
    assert secured.get("/api/documents").status_code == 401


def test_session_validation_errors_use_the_app_error_format(secured) -> None:
    r = secured.post("/api/session", json={})
    assert r.status_code == 422 and r.json()["detail"].startswith("Invalid request")


def test_session_cookie_flags_and_value(secured) -> None:
    r = secured.post("/api/session", json={"token": TOKEN})
    assert r.status_code == 204
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Path=/" in cookie
    assert "Max-Age" not in cookie and "expires" not in cookie.lower()  # session cookie
    assert TOKEN not in cookie  # the cookie holds a derived value, not the token
    assert "Secure" not in cookie  # plain http in tests


def test_cookie_is_secure_behind_https_proxy(secured) -> None:
    r = secured.post("/api/session", json={"token": TOKEN}, headers={"X-Forwarded-Proto": "https"})
    assert "Secure" in r.headers["set-cookie"]


def test_cookie_grants_access(signed_in) -> None:
    assert signed_in.get("/api/documents").status_code == 200
    assert signed_in.get("/api/documents/d1").json()["doc_id"] == "d1"


def test_forged_or_malformed_cookie_is_401(secured) -> None:
    for value in (TOKEN, "x", "", "é"):
        secured.cookies.clear()
        header = f"{SESSION_COOKIE}={value}".encode("latin-1")
        assert secured.get("/api/documents", headers={"Cookie": header}).status_code == 401
    r = secured.get("/api/documents", headers={"Cookie": "garbage;;=="})
    assert r.status_code == 401


def test_logout_clears_cookie(signed_in) -> None:
    r = signed_in.delete("/api/session")
    assert r.status_code == 204
    assert SESSION_COOKIE in r.headers["set-cookie"] and "Max-Age=0" in r.headers["set-cookie"]
    assert signed_in.get("/api/documents").status_code == 401


def test_sse_endpoints_and_page_images_work_with_cookie(signed_in) -> None:
    # exactly what EventSource and <img src> do: no Authorization header, cookie only
    ev = signed_in.get("/api/documents/d1/events")
    assert ev.status_code == 200 and ev.headers["content-type"].startswith("text/event-stream")
    assert [n for n, _ in parse_sse(ev.text)][-1] == "done"

    ask = signed_in.post("/api/ask/stream", json={"question": "revenue?", "doc_ids": ["d1"]})
    assert ask.status_code == 200
    assert [n for n, _ in parse_sse(ask.text)][-1] == "answer"

    img = signed_in.get("/api/documents/d1/pages/1.png")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_sse_endpoints_reject_anonymous_clients(secured) -> None:
    assert secured.get("/api/documents/d1/events").status_code == 401
    assert secured.post("/api/ask/stream", json={"question": "q"}).status_code == 401


def test_token_flag_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_RAG_TOKEN", raising=False)
    assert _parse_args([]).token is None
    monkeypatch.setenv("LOCAL_RAG_TOKEN", "from-env")
    assert _parse_args([]).token == "from-env"
    assert _parse_args(["--token", "from-flag"]).token == "from-flag"  # flag wins
    monkeypatch.setenv("LOCAL_RAG_TOKEN", "")
    assert _parse_args([]).token is None


def test_loopback_detection() -> None:
    assert all(_is_loopback(h) for h in ("127.0.0.1", "127.1.2.3", "localhost", "::1"))
    assert not any(_is_loopback(h) for h in ("0.0.0.0", "192.168.1.5", "::", "example.org"))


# ------------------------------------------------------------------ upload limits


def test_second_concurrent_upload_from_same_ip_is_429(lib, tmp_path) -> None:
    entered, release = threading.Event(), threading.Event()
    real_submit = lib.submit

    def slow_submit(src_path: str, filename: str):
        entered.set()
        assert release.wait(10)
        return real_submit(src_path, filename)

    lib.submit = slow_submit  # type: ignore[method-assign]
    with make_client(lib, tmp_path) as c:
        first: dict[str, int] = {}
        t = threading.Thread(target=lambda: first.update(status=upload(c).status_code))
        t.start()
        try:
            assert entered.wait(10)
            r = upload(c, data=PDF + b"other", name="second.pdf")
            assert r.status_code == 429
            assert "still in progress" in r.json()["detail"]
            assert r.headers["retry-after"]
        finally:
            release.set()
            t.join(10)
        assert first["status"] == 201
        # the slot is free again afterwards
        assert upload(c, data=PDF + b"third", name="third.pdf").status_code == 201


def test_failed_upload_frees_the_slot(lib, tmp_path) -> None:
    with make_client(lib, tmp_path) as c:
        assert upload(c, data=b"not a pdf", name="x.pdf").status_code == 415
        assert upload(c).status_code == 201


def test_daily_quota_returns_413(lib, tmp_path) -> None:
    # room for two uploads; the declared size includes multipart framing
    quota = len(PDF) * 2 + 1000
    with make_client(lib, tmp_path, upload_quota_bytes=quota) as c:
        assert upload(c, data=PDF + b"1", name="a.pdf").status_code == 201
        assert upload(c, data=PDF + b"2", name="b.pdf").status_code == 201
        r = upload(c, data=PDF + b"3", name="c.pdf")
        assert r.status_code == 413 and "quota" in r.json()["detail"]
        assert list((tmp_path / "incoming").iterdir()) == []
        # reading is unaffected
        assert c.get("/api/documents").status_code == 200


def test_quota_off_by_default(lib, tmp_path) -> None:
    with make_client(lib, tmp_path) as c:
        for i in range(5):
            assert upload(c, data=PDF + str(i).encode(), name=f"{i}.pdf").status_code == 201


def test_quota_rejects_before_reading_when_declared_size_does_not_fit(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, upload_quota_bytes=len(PDF) - 1) as c:
        assert upload(c).status_code == 413


def test_quota_applies_per_address_and_resets_each_day(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = UploadLimiter(1000)
    limiter.begin("a", 400)
    limiter.finish("a", 400)
    limiter.begin("a", 400)
    limiter.finish("a", 400)
    with pytest.raises(Exception) as exc:
        limiter.begin("a", 400)
    assert exc.value.status_code == 413  # type: ignore[attr-defined]
    limiter.begin("b", 400)  # another address is unaffected
    limiter.finish("b")
    monkeypatch.setattr(UploadLimiter, "_today", staticmethod(lambda: "2099-01-01"))
    limiter.begin("a", 400)  # new UTC day
    limiter.finish("a")


def test_quota_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_RAG_UPLOAD_QUOTA_BYTES", raising=False)
    assert _upload_quota_from_env() is None
    monkeypatch.setenv("LOCAL_RAG_UPLOAD_QUOTA_BYTES", "1048576")
    assert _upload_quota_from_env() == 1048576
    monkeypatch.setenv("LOCAL_RAG_UPLOAD_QUOTA_BYTES", "0")
    assert _upload_quota_from_env() is None
    monkeypatch.setenv("LOCAL_RAG_UPLOAD_QUOTA_BYTES", "ten")
    with pytest.raises(SystemExit):
        _upload_quota_from_env()


@pytest.mark.parametrize(
    "data", [b"PK\x03\x04 zip" + b"z" * 2000, b"<html>" + b"h" * 3000, b"\x89PNG" + b"p" * 2000]
)
def test_non_pdf_content_is_rejected_by_magic_bytes(lib, tmp_path, data) -> None:
    with make_client(lib, tmp_path) as c:
        # the .pdf name and the content type are not trusted
        r = upload(c, data=data, name="looks-like.pdf")
        assert r.status_code == 415 and "not a PDF" in r.json()["detail"]
        assert list((tmp_path / "incoming").iterdir()) == []


# ------------------------------------------------------------------ review fixes


def _chunked_upload(client: TestClient, data: bytes, name: str = "big.pdf"):
    """A multipart upload sent with Transfer-Encoding: chunked, so there is no Content-Length."""
    boundary = "xBOUNDARYx"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()

    def body():
        yield head
        for i in range(0, len(data), 500):
            yield data[i : i + 500]
        yield tail

    return client.post(
        "/api/documents",
        content=body(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def test_quota_caps_an_upload_that_declares_no_size(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, upload_quota_bytes=len(PDF) // 2) as c:
        r = _chunked_upload(c, PDF)
        assert r.status_code == 413 and "quota" in r.json()["detail"]
        assert list((tmp_path / "incoming").iterdir()) == []
        assert not [d for d in lib.docs if d != "d1"]


def test_chunked_upload_within_the_quota_is_accepted(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, upload_quota_bytes=len(PDF) + 10) as c:
        assert _chunked_upload(c, PDF).status_code == 201


def test_duplicate_upload_is_not_charged_to_the_quota(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, upload_quota_bytes=len(PDF) * 3 + 1000) as c:
        assert upload(c, data=PDF + b"A", name="a.pdf").status_code == 201
        for _ in range(3):  # the same file again: one stored copy, nothing new to charge
            assert upload(c, data=PDF + b"A", name="a.pdf").status_code == 201
        assert upload(c, data=PDF + b"B", name="b.pdf").status_code == 201
        assert upload(c, data=PDF + b"C", name="c.pdf").status_code == 201


def test_rejected_upload_is_not_charged_to_the_quota(lib, tmp_path, monkeypatch) -> None:
    with make_client(lib, tmp_path, upload_quota_bytes=len(PDF) * 2 + 1000) as c:
        real = lib.submit

        def refuse(src_path: str, filename: str):
            Path(src_path).unlink(missing_ok=True)
            raise ValueError("The PDF could not be added.")

        monkeypatch.setattr(lib, "submit", refuse)
        for _ in range(3):
            assert upload(c, data=PDF + b"X", name="x.pdf").status_code == 400
        monkeypatch.setattr(lib, "submit", real)
        assert upload(c, data=PDF + b"Y", name="y.pdf").status_code == 201


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_hidden_when_a_token_is_set(lib, tmp_path, path) -> None:
    with make_client(lib, tmp_path, token=TOKEN) as c:
        for headers in ({}, BEARER):
            r = c.get(path, headers=headers)
            assert r.status_code == 404 and "openapi" not in r.text.lower()
    with make_client(lib, tmp_path) as c:  # no token: unchanged
        assert c.get(path).status_code == 200
