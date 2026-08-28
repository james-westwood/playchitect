"""
Unit tests for model-download integrity verification (TASK-31, rail 3).

essentia.upf.edu publishes NO checksums for its model .pb files (verified
2026-08-27: the model .json metadata only carries name/type/link/version/
description/author/release_date -- no digest field). So sha256 verification
here is necessarily trust-on-first-use: it does NOT protect the very first
download of a model. What it DOES protect is the model file silently
changing under us on a *later* download -- altering embedding semantics
while the code pin and the embedding cache's model_version column both
still claim nothing moved. That is the actual threat this rail defends
against, and this file's docstrings/messages should stay honest about that
distinction rather than overselling this as a supply-chain guarantee.

The three digests below were computed by the project owner from the
currently-published model files on 2026-08-27, and verified stable across
repeated downloads; see prd.json TASK-31 for provenance. They MUST appear
verbatim as module constants in embedding_extractor.py once implemented.

Interface under test (PROPOSED by the test writer -- see report):

    EmbeddingExtractor._download_model(
        self, target: Path, pb_url: str, meta_url: str, *, expected_sha256: str,
    ) -> None
        Downloads target from pb_url (and meta from meta_url) exactly as
        before, then verifies sha256(target's bytes) == expected_sha256,
        raising ModelIntegrityError on mismatch and removing the
        mismatched file from disk so a later _ensure_model() call does not
        mistake it for a valid cached download.

        NOTE: this is a signature change from the current 3-positional-arg
        _download_model(target, pb_url, meta_url). The three pre-existing
        calls to _download_model(...) with 3 args in
        tests/unit/test_embedding_extractor.py
        (TestEmbeddingExtractorDownload, TestDownloadFailure) will need
        updating by whoever implements this rail, since expected_sha256 has
        deliberately been given NO default -- an optional/defaulted digest
        would make it trivial to accidentally skip verification at a call
        site. See this task's Test Writer report for the explicit flag on
        this friction point.

    playchitect.core.embedding_extractor.ModelIntegrityError(RuntimeError)
        Raised on sha256 mismatch. Message must name both the expected and
        actual digest (for diagnosability) and explain how to proceed if
        the mismatch is a legitimate upstream re-release rather than
        tampering/corruption (i.e. how to update the pinned constant).

    Module constants pairing each existing `*_URL` constant with a
    `*_SHA256` sibling:
        _MSD_MUSICNN_URL     / _MSD_MUSICNN_SHA256
        _MIREX_MOODS_URL     / _MIREX_MOODS_SHA256
        _DISCOGS_EFFNET_URL  / _DISCOGS_EFFNET_SHA256

No real network downloads happen in this file: urllib.request.urlretrieve
is always monkeypatched to write locally-controlled bytes. The REAL
hashlib.sha256 computation is never mocked -- that is the logic under test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import playchitect.core.embedding_extractor as emb_mod
from playchitect.core.embedding_extractor import EmbeddingExtractor

# Digests supplied by the project owner (2026-08-27) from files they
# downloaded and hashed themselves; see this module's docstring. Kept here
# as independent literals (not imported from the module under test) so this
# file still asserts the real, owner-supplied values even if the eventual
# implementation's constants are wrong or transcribed incorrectly.
_KNOWN_MSD_MUSICNN_SHA256 = "cdea0722bcee7f731286843f2233e3aa69887bb5c3e2dce011eff55f38d04f3e"
_KNOWN_MIREX_MOODS_SHA256 = "d90d3020fab17ad641857a7272c94fd27a0d5c11aa92aa130dcc807c58ee6fab"
_KNOWN_DISCOGS_EFFNET_SHA256 = "3ed9af50d5367c0b9c795b294b00e7599e4943244f4cbd376869f3bfc87721b1"

_URL_TO_DIGEST_CONSTANT_NAME = {
    "_MSD_MUSICNN_URL": "_MSD_MUSICNN_SHA256",
    "_MIREX_MOODS_URL": "_MIREX_MOODS_SHA256",
    "_DISCOGS_EFFNET_URL": "_DISCOGS_EFFNET_SHA256",
}


def _fake_urlretrieve_writing(content: bytes):
    """
    Return a fake urlretrieve(url, dest) that writes `content` to dest,
    ignoring the URL -- stands in for the network fetch only.

    The real sha256 verification logic under test still runs against these
    locally-written bytes; only the network layer is faked, per the task's
    explicit instruction not to mock the hashing away.
    """

    def _fake(url: str, dest: object) -> None:
        Path(str(dest)).write_bytes(content)

    return _fake


# ── TestModelDigestPairing ───────────────────────────────────────────────────


class TestModelDigestPairing:
    """Every model URL constant must have a matching, correctly-valued sha256
    constant, so a future model addition can't accidentally ship without an
    integrity pin."""

    def test_msd_musicnn_pin_matches_owner_supplied_digest(self) -> None:
        assert emb_mod._MSD_MUSICNN_SHA256 == _KNOWN_MSD_MUSICNN_SHA256  # ty: ignore[unresolved-attribute]

    def test_mirex_moods_pin_matches_owner_supplied_digest(self) -> None:
        assert emb_mod._MIREX_MOODS_SHA256 == _KNOWN_MIREX_MOODS_SHA256  # ty: ignore[unresolved-attribute]

    def test_discogs_effnet_pin_matches_owner_supplied_digest(self) -> None:
        assert emb_mod._DISCOGS_EFFNET_SHA256 == _KNOWN_DISCOGS_EFFNET_SHA256  # ty: ignore[unresolved-attribute]

    def test_every_known_model_url_constant_has_a_sha256_sibling(self) -> None:
        for url_name, sha_name in _URL_TO_DIGEST_CONSTANT_NAME.items():
            assert hasattr(emb_mod, url_name), f"expected existing constant {url_name} missing"
            assert hasattr(emb_mod, sha_name), (
                f"{url_name} has no matching {sha_name} constant -- every "
                "downloadable model must pin an integrity digest"
            )

    def test_every_url_constant_in_the_module_has_a_sha256_sibling(self) -> None:
        """
        Structural guardrail, independent of the specific model list above:
        for every module-level '<NAME>_URL' string constant there must be a
        matching '<NAME>_SHA256' constant, each a 64-char lowercase hex
        string -- so a future model addition can't silently skip the
        integrity check by forgetting to pair a digest with its URL.
        """
        module_vars = vars(emb_mod)
        url_constants = {
            name: value
            for name, value in module_vars.items()
            if name.endswith("_URL") and isinstance(value, str)
        }
        assert len(url_constants) >= 3, "expected at least the 3 known model URLs"

        for url_name in url_constants:
            base = url_name[: -len("_URL")]
            sha_name = f"{base}_SHA256"
            assert sha_name in module_vars, (
                f"{url_name} has no matching {sha_name} constant -- every "
                "downloadable model must pin an integrity digest"
            )
            digest = module_vars[sha_name]
            assert isinstance(digest, str)
            assert len(digest) == 64, f"{sha_name} is not a 64-char sha256 hex digest"
            assert digest == digest.lower(), f"{sha_name} must be lowercase hex"
            assert all(c in "0123456789abcdef" for c in digest), (
                f"{sha_name} contains non-hex characters"
            )


# ── TestDownloadModelIntegrityVerification ───────────────────────────────────


class TestDownloadModelIntegrityVerification:
    """_download_model must verify sha256 of the downloaded .pb against an
    expected digest, raising a clearly-named exception on mismatch."""

    def test_succeeds_when_digest_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        content = b"a totally fake but internally-consistent model payload"
        expected_digest = hashlib.sha256(content).hexdigest()

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            _fake_urlretrieve_writing(content),
        )

        target = tmp_path / "models" / "msd-musicnn-1.pb"
        extractor = EmbeddingExtractor(model_path=target, cache_enabled=False)

        # Must not raise.
        extractor._download_model(
            target,
            emb_mod._MSD_MUSICNN_URL,
            "https://fake.json",
            expected_sha256=expected_digest,  # ty: ignore[unknown-argument]
        )
        assert target.read_bytes() == content

    def test_raises_named_exception_on_digest_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Write a deliberately corrupted file (bytes that do NOT hash to the
        expected digest) and let the real hashing logic detect the
        mismatch -- the hashing itself is what's under test here, not
        urllib, so only the network fetch is faked.
        """
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        corrupted_content = b"corrupted / truncated download, not the real model bytes"
        # Sanity check on the test fixture itself: must not coincidentally
        # match the pinned digest, or this test would prove nothing.
        assert hashlib.sha256(corrupted_content).hexdigest() != _KNOWN_MSD_MUSICNN_SHA256

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            _fake_urlretrieve_writing(corrupted_content),
        )

        target = tmp_path / "models" / "msd-musicnn-1.pb"
        extractor = EmbeddingExtractor(model_path=target, cache_enabled=False)

        with pytest.raises(emb_mod.ModelIntegrityError) as exc_info:  # ty: ignore[unresolved-attribute]
            extractor._download_model(
                target,
                emb_mod._MSD_MUSICNN_URL,
                "https://fake.json",
                expected_sha256=_KNOWN_MSD_MUSICNN_SHA256,  # ty: ignore[unknown-argument]
            )

        message = str(exc_info.value)
        # Message must be diagnosable: name both the expected and the actual
        # digest so an operator can tell what happened.
        assert _KNOWN_MSD_MUSICNN_SHA256 in message
        assert hashlib.sha256(corrupted_content).hexdigest() in message

    def test_mismatch_error_message_explains_how_to_update_a_legitimate_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Because essentia publishes no checksums, a mismatch is not always an
        attack -- it could be a legitimate upstream re-release. The error
        must tell the operator how to proceed if that's the case (i.e. how
        to update the pinned constant), not just that verification failed.
        """
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            _fake_urlretrieve_writing(b"some other content entirely"),
        )

        target = tmp_path / "models" / "msd-musicnn-1.pb"
        extractor = EmbeddingExtractor(model_path=target, cache_enabled=False)

        with pytest.raises(emb_mod.ModelIntegrityError) as exc_info:  # ty: ignore[unresolved-attribute]
            extractor._download_model(
                target,
                emb_mod._MSD_MUSICNN_URL,
                "https://fake.json",
                expected_sha256=_KNOWN_MSD_MUSICNN_SHA256,  # ty: ignore[unknown-argument]
            )

        message = str(exc_info.value).lower()
        assert "update" in message or "re-pin" in message or "regenerate" in message

    def test_mismatch_leaves_no_misleadingly_valid_file_on_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        On a digest mismatch, the corrupted/tampered file must not be left
        sitting at `target` looking like a successfully-downloaded model --
        otherwise the next _ensure_model() call would see target.exists()
        and skip re-downloading a good copy.
        """
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            _fake_urlretrieve_writing(b"corrupted bytes"),
        )

        target = tmp_path / "models" / "msd-musicnn-1.pb"
        extractor = EmbeddingExtractor(model_path=target, cache_enabled=False)

        with pytest.raises(emb_mod.ModelIntegrityError):  # ty: ignore[unresolved-attribute]
            extractor._download_model(
                target,
                emb_mod._MSD_MUSICNN_URL,
                "https://fake.json",
                expected_sha256=_KNOWN_MSD_MUSICNN_SHA256,  # ty: ignore[unknown-argument]
            )

        assert not target.exists()


# ── TestExistingCallSitesPassExpectedDigest ──────────────────────────────────


class TestExistingCallSitesPassExpectedDigest:
    """_ensure_model() / _ensure_discogs_effnet_model() must pass the correct
    pinned digest through to _download_model for each model they manage --
    otherwise rail 3 is wired up but never actually invoked in practice."""

    def test_ensure_model_passes_correct_digest_for_each_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        calls: list[tuple[Path, str, str, str]] = []

        def fake_download(
            target: Path, pb_url: str, meta_url: str, *, expected_sha256: str
        ) -> None:
            calls.append((target, pb_url, meta_url, expected_sha256))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")

        class DummyModel:
            def __init__(self, **kwargs: object) -> None:
                pass

        monkeypatch.setattr(emb_mod, "_EssentiaModel", DummyModel)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", DummyModel)

        model_file = tmp_path / "missing.pb"
        mood_file = tmp_path / "missing_mood.pb"
        extractor = EmbeddingExtractor(
            model_path=model_file, mood_model_path=mood_file, cache_enabled=False
        )
        extractor._download_model = fake_download  # ty: ignore[invalid-assignment]
        extractor._ensure_model()

        msd_calls = [c for c in calls if c[0] == model_file]
        assert len(msd_calls) == 1
        assert msd_calls[0][3] == emb_mod._MSD_MUSICNN_SHA256  # ty: ignore[unresolved-attribute]

        mood_calls = [c for c in calls if c[0] == mood_file]
        assert len(mood_calls) == 1
        assert mood_calls[0][3] == emb_mod._MIREX_MOODS_SHA256  # ty: ignore[unresolved-attribute]

    def test_ensure_discogs_effnet_model_passes_correct_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        calls: list[tuple[Path, str, str, str]] = []

        def fake_download(
            target: Path, pb_url: str, meta_url: str, *, expected_sha256: str
        ) -> None:
            calls.append((target, pb_url, meta_url, expected_sha256))

        discogs_target = tmp_path / "models" / "discogs-effnet-bs64-1.pb"
        extractor = EmbeddingExtractor(
            model_path=tmp_path / "fake.pb",
            mood_model_path=tmp_path / "fake_mood.pb",
            discogs_effnet_model_path=discogs_target,
            cache_enabled=False,
        )
        extractor._download_model = fake_download  # ty: ignore[invalid-assignment]
        extractor._ensure_discogs_effnet_model()

        assert len(calls) == 1
        assert calls[0][0] == discogs_target
        assert calls[0][3] == emb_mod._DISCOGS_EFFNET_SHA256  # ty: ignore[unresolved-attribute]
