"""What the user is told must match what happened: upload pre-check, honest wording."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tests.test_server import PDF, FakeLibrary, make_client, upload

TOKEN = "s3cret-token"


@pytest.fixture
def lib(tmp_path: Path):
    store = tmp_path / "store"
    store.mkdir()
    fake = FakeLibrary(store)
    fake.add_ready("d1", "Annual report")
    yield fake
    fake.release.set()


# ------------------------------------------------------------------ /api/upload-check


def test_upload_check_ok_reports_limits(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, max_upload_mb=2, upload_quota_bytes=5_000_000) as c:
        r = c.get("/api/upload-check", params={"size": 1000})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["max_bytes"] == 2 * 1024 * 1024
        assert body["quota_left_bytes"] == 5_000_000


def test_upload_check_without_quota_has_no_quota_left(lib, tmp_path) -> None:
    with make_client(lib, tmp_path) as c:
        assert c.get("/api/upload-check").json()["quota_left_bytes"] is None


def test_upload_check_too_large_gives_the_real_reason(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, max_upload_mb=1) as c:
        r = c.get("/api/upload-check", params={"size": 3 * 1024 * 1024})
        assert r.status_code == 413
        assert r.json()["detail"] == "This file is larger than the 1 MB upload limit."


def test_upload_check_over_quota_names_the_numbers(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, upload_quota_bytes=3_000_000) as c:
        r = c.get("/api/upload-check", params={"size": 40_000_000})
        assert r.status_code == 413
        detail = r.json()["detail"]
        assert "Daily upload quota exceeded" in detail
        assert "2.9 MB per address" in detail  # 3,000,000 bytes, not "2 MB"
        assert "38 MB" in detail and "left today" in detail


def test_upload_check_after_quota_is_used_up(lib, tmp_path) -> None:
    quota = len(PDF) * 2 + 1000
    with make_client(lib, tmp_path, upload_quota_bytes=quota) as c:
        assert upload(c, data=PDF + b"1", name="a.pdf").status_code == 201
        assert upload(c, data=PDF + b"2", name="b.pdf").status_code == 201
        r = c.get("/api/upload-check", params={"size": len(PDF)})
        assert r.status_code == 413 and "quota" in r.json()["detail"]


def test_upload_check_reports_another_upload_in_progress(lib, tmp_path) -> None:
    entered, release = threading.Event(), threading.Event()
    real_submit = lib.submit

    def slow_submit(src_path: str, filename: str):
        entered.set()
        assert release.wait(10)
        return real_submit(src_path, filename)

    lib.submit = slow_submit  # type: ignore[method-assign]
    with make_client(lib, tmp_path) as c:
        t = threading.Thread(target=lambda: upload(c))
        t.start()
        try:
            assert entered.wait(10)
            r = c.get("/api/upload-check", params={"size": 10})
            assert r.status_code == 429 and "still in progress" in r.json()["detail"]
        finally:
            release.set()
            t.join(10)
        assert c.get("/api/upload-check", params={"size": 10}).status_code == 200


def test_upload_check_reserves_nothing(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, upload_quota_bytes=len(PDF) + 1000) as c:
        for _ in range(5):
            assert c.get("/api/upload-check", params={"size": 10}).status_code == 200
        assert upload(c).status_code == 201


def test_upload_check_needs_the_token_and_says_401(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, token=TOKEN) as c:
        r = c.get("/api/upload-check", params={"size": 10})
        assert r.status_code == 401 and "Authentication required" in r.json()["detail"]
        ok = c.get("/api/upload-check", headers={"Authorization": f"Bearer {TOKEN}"})
        assert ok.status_code == 200


def test_upload_check_rejects_a_negative_size(lib, tmp_path) -> None:
    with make_client(lib, tmp_path) as c:
        assert c.get("/api/upload-check", params={"size": -5}).status_code == 422


def test_upload_check_is_not_a_document_id(lib, tmp_path) -> None:
    """The route must not be swallowed by /api/documents/{doc_id}."""
    with make_client(lib, tmp_path) as c:
        assert c.get("/api/upload-check").json()["ok"] is True
        assert c.get("/api/documents/upload-check").status_code == 404


# ------------------------------------------------------------------ duplicates


def test_upload_response_says_when_the_file_was_already_there(lib, tmp_path) -> None:
    with make_client(lib, tmp_path) as c:
        first = upload(c, data=PDF + b"same", name="a.pdf")
        again = upload(c, data=PDF + b"same", name="b.pdf")
        assert first.status_code == again.status_code == 201
        assert first.json()["already_in_library"] is False
        assert again.json()["already_in_library"] is True
        assert again.json()["doc_id"] == first.json()["doc_id"]


# ------------------------------------------------------------------ wording in the page


def test_page_wording_does_not_overclaim(lib, tmp_path) -> None:
    """Static guards: the page script must keep the honest wording (no browser here)."""
    with make_client(lib, tmp_path) as c:
        js = c.get("/app.js").text
    assert "verify: 'Verified'" not in js  # a finished check step is not a pass
    assert "api/upload-check" in js  # the size/quota pre-check before sending a body
    assert "Not checked" in js  # grounded=None must not look verified
    assert "Draft — " in js  # streamed text is labelled until the checked answer arrives
    assert "matched by wording" in js  # inferred citations are labelled as such
    assert "already_in_library" in js and "could not be processed" in js
