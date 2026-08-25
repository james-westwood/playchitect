"""TDD red tests for TASK-18: degenerate K warning.

After a clustering run, detect degenerate separation:
  - largest cluster >= 70% of clustered tracks, AND
  - auto-selected K <= 2, AND
  - total clustered tracks >= 50.
When detected, log a WARNING and surface it via ``clusterer.last_degenerate_warning``.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

import pytest

from playchitect.core.clustering import PlaylistClusterer
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata


def _make_metadata(path: Path, bpm: float, duration: float = 360.0) -> TrackMetadata:
    return TrackMetadata(filepath=path, bpm=bpm, duration=duration)


def _make_intensity(
    path: Path,
    rms: float = 0.5,
    brightness: float = 0.5,
    sub_bass: float = 0.3,
    kick: float = 0.6,
    harmonics: float = 0.4,
    perc: float = 0.5,
    onset: float = 0.5,
) -> IntensityFeatures:
    return IntensityFeatures(
        file_path=path,
        file_hash="deadbeef",
        rms_energy=rms,
        brightness=brightness,
        sub_bass_energy=sub_bass,
        kick_energy=kick,
        bass_harmonics=harmonics,
        percussiveness=perc,
        onset_strength=onset,
        camelot_key="8B",
        key_index=0.0,
    )


def _degenerate_features_190() -> tuple[dict[Path, TrackMetadata], dict[Path, IntensityFeatures]]:
    """190 tracks: 184 near-identical, 6 clearly different.

    The 184-row blob is intentionally tight so auto-K lands at <= 2.
    """
    rng = random.Random(42)
    meta: dict[Path, TrackMetadata] = {}
    intensity: dict[Path, IntensityFeatures] = {}
    for i in range(190):
        p = Path(f"track_{i:03d}.mp3")
        if i < 184:
            bpm = 128.0 + rng.uniform(-0.05, 0.05)
            jitter = 0.005
            rms = 0.5 + rng.uniform(-jitter, jitter)
            brightness = 0.5 + rng.uniform(-jitter, jitter)
            sub_bass = 0.3 + rng.uniform(-jitter, jitter)
            kick = 0.6 + rng.uniform(-jitter, jitter)
            harmonics = 0.4 + rng.uniform(-jitter, jitter)
            perc = 0.5 + rng.uniform(-jitter, jitter)
            onset = 0.5 + rng.uniform(-jitter, jitter)
        else:
            bpm = 140.0 + rng.uniform(-0.5, 0.5)
            rms = 0.85 + rng.uniform(-0.02, 0.02)
            brightness = 0.85 + rng.uniform(-0.02, 0.02)
            sub_bass = 0.8 + rng.uniform(-0.02, 0.02)
            kick = 0.9 + rng.uniform(-0.02, 0.02)
            harmonics = 0.8 + rng.uniform(-0.02, 0.02)
            perc = 0.9 + rng.uniform(-0.02, 0.02)
            onset = 0.9 + rng.uniform(-0.02, 0.02)
        meta[p] = _make_metadata(p, bpm=bpm)
        intensity[p] = _make_intensity(
            p,
            rms=rms,
            brightness=brightness,
            sub_bass=sub_bass,
            kick=kick,
            harmonics=harmonics,
            perc=perc,
            onset=onset,
        )
    return meta, intensity


def _degenerate_bpm_190() -> dict[Path, TrackMetadata]:
    """190 tracks: 184 near-identical BPM, 6 clearly different BPM."""
    rng = random.Random(42)
    meta: dict[Path, TrackMetadata] = {}
    for i in range(190):
        p = Path(f"track_{i:03d}.mp3")
        if i < 184:
            bpm = 128.0 + rng.uniform(-0.05, 0.05)
        else:
            bpm = 140.0 + rng.uniform(-0.5, 0.5)
        meta[p] = _make_metadata(p, bpm=bpm)
    return meta


def _well_separated_200() -> tuple[dict[Path, TrackMetadata], dict[Path, IntensityFeatures]]:
    """200 tracks in 4 clean, well-separated feature blobs."""
    rng = random.Random(42)
    meta: dict[Path, TrackMetadata] = {}
    intensity: dict[Path, IntensityFeatures] = {}
    for i in range(200):
        blob = i // 50
        p = Path(f"track_{i:03d}.mp3")
        bpm = 120.0 + blob * 10.0 + rng.uniform(-1.0, 1.0)
        rms = 0.2 + blob * 0.2 + rng.uniform(-0.02, 0.02)
        brightness = 0.2 + blob * 0.2 + rng.uniform(-0.02, 0.02)
        meta[p] = _make_metadata(p, bpm=bpm)
        intensity[p] = _make_intensity(
            p,
            rms=rms,
            brightness=brightness,
            sub_bass=0.3,
            kick=0.6,
            harmonics=0.4,
            perc=0.5,
            onset=0.5,
        )
    return meta, intensity


def _dominant_40() -> tuple[dict[Path, TrackMetadata], dict[Path, IntensityFeatures]]:
    """40 tracks with the same 96% / 4% dominance ratio as the 190 case."""
    rng = random.Random(42)
    meta: dict[Path, TrackMetadata] = {}
    intensity: dict[Path, IntensityFeatures] = {}
    for i in range(40):
        p = Path(f"track_{i:03d}.mp3")
        if i < 38:
            bpm = 128.0 + rng.uniform(-0.05, 0.05)
            jitter = 0.005
            rms = 0.5 + rng.uniform(-jitter, jitter)
            brightness = 0.5 + rng.uniform(-jitter, jitter)
        else:
            bpm = 140.0 + rng.uniform(-0.5, 0.5)
            rms = 0.85 + rng.uniform(-0.02, 0.02)
            brightness = 0.85 + rng.uniform(-0.02, 0.02)
        meta[p] = _make_metadata(p, bpm=bpm)
        intensity[p] = _make_intensity(
            p,
            rms=rms,
            brightness=brightness,
        )
    return meta, intensity


@pytest.fixture
def caplog_warning(caplog):
    """Capture WARNING level logs from the clustering module."""
    with caplog.at_level(logging.WARNING, logger="playchitect.core.clustering"):
        yield caplog


class TestDegenerateKWarning:
    def test_degenerate_features_path_logs_warning(self, caplog_warning) -> None:
        meta, intensity = _degenerate_features_190()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25, random_state=42)
        clusterer.cluster_by_features(meta, intensity, use_ewkm=False)

        warning = getattr(clusterer, "last_degenerate_warning", None)
        assert warning is not None, "expected last_degenerate_warning to be set"
        assert isinstance(warning, str) and warning
        assert "Clusters did not separate" in warning
        assert "--use-embeddings" in warning

        assert "Clusters did not separate" in caplog_warning.text
        assert "--use-embeddings" in caplog_warning.text

    def test_well_separated_data_no_warning(self, caplog_warning) -> None:
        meta, intensity = _well_separated_200()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25, random_state=42)
        clusterer.cluster_by_features(meta, intensity, use_ewkm=False)

        warning = getattr(clusterer, "last_degenerate_warning", None)
        assert warning is None
        assert "Clusters did not separate" not in caplog_warning.text

    def test_small_library_no_warning(self, caplog_warning) -> None:
        meta, intensity = _dominant_40()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25, random_state=42)
        clusterer.cluster_by_features(meta, intensity, use_ewkm=False)

        warning = getattr(clusterer, "last_degenerate_warning", None)
        assert warning is None
        assert "Clusters did not separate" not in caplog_warning.text

    def test_forced_k_no_warning(self, caplog_warning) -> None:
        meta, intensity = _degenerate_features_190()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25, random_state=42)
        clusterer.cluster_by_features(meta, intensity, use_ewkm=False, n_playlists=2)

        warning = getattr(clusterer, "last_degenerate_warning", None)
        assert warning is None
        assert "Clusters did not separate" not in caplog_warning.text

    def test_bpm_only_path_warns(self, caplog_warning) -> None:
        meta = _degenerate_bpm_190()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25, random_state=42)
        clusterer.cluster_by_bpm(meta)

        warning = getattr(clusterer, "last_degenerate_warning", None)
        assert warning is not None, "expected last_degenerate_warning to be set"
        assert isinstance(warning, str) and warning
        assert "Clusters did not separate" in warning
        assert "--use-embeddings" in warning

        assert "Clusters did not separate" in caplog_warning.text
        assert "--use-embeddings" in caplog_warning.text

    def test_warning_message_contains_percentage_and_count(self, caplog_warning) -> None:
        meta, intensity = _degenerate_features_190()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25, random_state=42)
        clusterer.cluster_by_features(meta, intensity, use_ewkm=False)

        warning = getattr(clusterer, "last_degenerate_warning", None)
        assert warning is not None
        assert re.search(r"\d+(\.\d+)?%", warning) is not None
        assert "190" in warning

        assert re.search(r"\d+(\.\d+)?%", caplog_warning.text) is not None
        assert "190" in caplog_warning.text
