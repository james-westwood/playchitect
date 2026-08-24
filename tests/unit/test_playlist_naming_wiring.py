"""Failing-first unit tests for TASK-17: wiring playlist naming into exporters/CLI.

These tests define the contract for two new helpers:

1. playchitect.core.naming.assign_cluster_names
   Given clusters, features, metadata, and a base playlist name, returns a
   mapping of cluster_id to a display name.

2. playchitect.core.export.sanitize_filename
   Given a display name, returns a filesystem-safe filename string.
"""

from __future__ import annotations

import re
from pathlib import Path

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

PLAYLIST_NAME = "TestSet"

# Structural vocabulary check: names must contain at least one descriptor or noun
# drawn from the naming package's TAG_TO_DESCRIPTORS / MOOD_TO_NOUN tables.
_KNOWN_DESCRIPTOR_OR_NOUN = {
    "Dark",
    "Journey",
    "Ethereal",
    "Wave",
    "Aggressive",
    "Set",
    "Bright",
    "Luminous",
    "Loud",
    "Powerful",
    "Driving",
    "Hypnotic",
    "Rapid",
    "Swift",
    "Subtle",
    "Groove",
    "Intense",
    "Extreme",
    "Shadowed",
    "Nocturnal",
    "Celestial",
    "Propulsive",
    "Mesmerizing",
    "Trance",
    "Jagged",
    "Geometric",
    "Minimal",
    "Deep",
    "Raw",
    "Polished",
    "Gritty",
    "Smooth",
    "Rough",
    "Clean",
    "Pure",
    "Harsh",
    "Vast",
    "Dense",
    "Massive",
    "Compact",
    "Expansive",
    "Intimate",
    "Wide",
    "Narrow",
    "Late",
    "Ancient",
    "New",
    "Fresh",
    "Vintage",
    "Modern",
    "Classic",
    "Flowing",
    "Angular",
    "Curved",
    "Round",
    "Linear",
    "Midnight",
    "Amber",
    "Azure",
    "Crimson",
    "Golden",
    "Silver",
    "Obsidian",
    "Nordic",
    "Urban",
    "Tribal",
    "Industrial",
    "Tropical",
    "Cosmic",
    "Digital",
    "Analog",
    "Metallic",
    "Organic",
    "Synthetic",
    "Wooden",
    "Crystal",
    "Languid",
    "Unhurried",
    "Leisurely",
    "Relaxed",
    "Mellow",
    "Steady",
    "Measured",
    "Moderate",
    "Balanced",
    "Even",
    "Pulsing",
    "Thrusting",
    "Pushing",
    "Brisk",
    "Accelerated",
    "Turbo",
    "Delicate",
    "Nuanced",
    "Understated",
    "Refined",
    "Rhythmic",
    "Catchy",
    "Infectious",
    "Vigorous",
    "Dynamic",
    "Animated",
    "Lively",
    "Radiant",
    "Brilliant",
    "Vivid",
    "Punchy",
    "Staccato",
    "Vocal",
    "Lyrical",
    "Sung",
    "Harmonic",
    "Melodic",
    "Quick",
    "Fast",
    "Thunderous",
    "Resonant",
    "Potent",
    "Percussive",
    "Driving",
    "Staccato",
    "Rhythmic",
}


def _legacy_name(cluster: ClusterResult, index: int, playlist_name: str) -> str:
    """Reproduce the legacy exporter filename label."""
    bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
    genre_label = f" {cluster.genre}" if cluster.genre else ""
    return f"{playlist_name} {index + 1} [{bpm_label}{genre_label}]"


def _bpm_range_suffix(cluster: ClusterResult) -> str:
    """Return the BPM range suffix in the existing exporter style."""
    return f"[{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm]"


def _make_metadata(path: Path, bpm: float, duration: float = 300.0) -> TrackMetadata:
    return TrackMetadata(filepath=path, bpm=bpm, duration=duration)


def _make_features(
    path: Path,
    rms: float,
    brightness: float,
    mood: str = "Ethereal",
) -> IntensityFeatures:
    return IntensityFeatures(
        file_path=path,
        file_hash="deadbeef",
        rms_energy=rms,
        brightness=brightness,
        percussiveness=0.5,
        vocal_presence=0.3,
        mood_label=mood,
    )


def _make_cluster(
    cluster_id: int | str,
    tracks: list[Path],
    bpm_mean: float,
    bpm_std: float,
    genre: str | None = None,
) -> ClusterResult:
    return ClusterResult(
        cluster_id=cluster_id,
        tracks=tracks,
        bpm_mean=bpm_mean,
        bpm_std=bpm_std,
        track_count=len(tracks),
        total_duration=len(tracks) * 300.0,
        genre=genre,
    )


def _build_cluster_inputs(
    cluster_id: int | str,
    num_tracks: int,
    bpm_mean: float,
    bpm_std: float,
    rms: float,
    brightness: float,
    mood: str = "Ethereal",
    genre: str | None = None,
) -> tuple[ClusterResult, dict[Path, IntensityFeatures], dict[Path, TrackMetadata]]:
    base = f"cluster_{cluster_id}"
    tracks = [Path(f"/music/{base}_track{i}.mp3") for i in range(num_tracks)]
    cluster = _make_cluster(cluster_id, tracks, bpm_mean, bpm_std, genre)
    features = {t: _make_features(t, rms, brightness, mood) for t in tracks}
    metadata = {t: _make_metadata(t, bpm_mean) for t in tracks}
    return cluster, features, metadata


class TestAssignClusterNames:
    """Contract tests for playchitect.core.naming.assign_cluster_names."""

    def setup_method(self) -> None:
        """Import the helpers under test; imports fail until the implementation is added."""
        from playchitect.core.export import sanitize_filename
        from playchitect.core.naming import assign_cluster_names

        self._assign_cluster_names = assign_cluster_names
        self._sanitize_filename = sanitize_filename

    def test_distinct_feature_profiles_get_distinct_names(self) -> None:
        """Two clusters with very different rms/brightness get different, non-legacy names."""
        cluster_a, features_a, metadata_a = _build_cluster_inputs(
            cluster_id=0,
            num_tracks=3,
            bpm_mean=124.0,
            bpm_std=2.0,
            rms=0.9,
            brightness=0.9,
            mood="Aggressive",
        )
        cluster_b, features_b, metadata_b = _build_cluster_inputs(
            cluster_id=1,
            num_tracks=3,
            bpm_mean=136.0,
            bpm_std=3.0,
            rms=0.1,
            brightness=0.1,
            mood="Ethereal",
        )

        features = {**features_a, **features_b}
        metadata = {**metadata_a, **metadata_b}
        clusters = [cluster_a, cluster_b]

        names = self._assign_cluster_names(clusters, features, metadata, PLAYLIST_NAME)

        assert len(names) == 2
        name_a = names[0]
        name_b = names[1]
        assert name_a != name_b
        assert _legacy_name(cluster_a, 0, PLAYLIST_NAME) != name_a
        assert _legacy_name(cluster_b, 1, PLAYLIST_NAME) != name_b
        for name in (name_a, name_b):
            assert any(word in name for word in _KNOWN_DESCRIPTOR_OR_NOUN)

    def test_same_character_clusters_get_bpm_range_suffix(self) -> None:
        """Two clusters with near-identical features but different BPMs are disambiguated."""
        cluster_a, features_a, metadata_a = _build_cluster_inputs(
            cluster_id=0,
            num_tracks=3,
            bpm_mean=124.0,
            bpm_std=2.0,
            rms=0.5,
            brightness=0.5,
            mood="Ethereal",
        )
        cluster_b, features_b, metadata_b = _build_cluster_inputs(
            cluster_id=1,
            num_tracks=3,
            bpm_mean=129.0,
            bpm_std=2.0,
            rms=0.5,
            brightness=0.5,
            mood="Ethereal",
        )

        features = {**features_a, **features_b}
        metadata = {**metadata_a, **metadata_b}
        clusters = [cluster_a, cluster_b]

        names = self._assign_cluster_names(clusters, features, metadata, PLAYLIST_NAME)

        name_a = names[0]
        name_b = names[1]
        assert _bpm_range_suffix(cluster_a) in name_a
        assert _bpm_range_suffix(cluster_b) in name_b
        assert name_a != name_b

    def test_single_cluster_gets_clean_character_name(self) -> None:
        """A single cluster with features gets a clean character name with no BPM suffix."""
        cluster, features, metadata = _build_cluster_inputs(
            cluster_id=0,
            num_tracks=4,
            bpm_mean=128.0,
            bpm_std=2.0,
            rms=0.5,
            brightness=0.8,
            mood="Ethereal",
        )

        names = self._assign_cluster_names([cluster], features, metadata, PLAYLIST_NAME)

        assert len(names) == 1
        name = names[0]
        assert _bpm_range_suffix(cluster) not in name
        assert _legacy_name(cluster, 0, PLAYLIST_NAME) != name
        assert any(word in name for word in _KNOWN_DESCRIPTOR_OR_NOUN)

    def test_no_features_falls_back_to_legacy_naming(self) -> None:
        """Empty features dict forces the exact legacy BPM-range label."""
        cluster_a = _make_cluster(
            cluster_id=0,
            tracks=[Path(f"/music/legacy_a{i}.mp3") for i in range(3)],
            bpm_mean=124.0,
            bpm_std=2.0,
            genre=None,
        )
        cluster_b = _make_cluster(
            cluster_id=1,
            tracks=[Path(f"/music/legacy_b{i}.mp3") for i in range(3)],
            bpm_mean=136.0,
            bpm_std=3.0,
            genre="techno",
        )
        metadata = {
            t: _make_metadata(t, bpm=124.0 if "legacy_a" in str(t) else 136.0)
            for cluster in (cluster_a, cluster_b)
            for t in cluster.tracks
        }

        names = self._assign_cluster_names([cluster_a, cluster_b], {}, metadata, PLAYLIST_NAME)

        assert names[0] == _legacy_name(cluster_a, 0, PLAYLIST_NAME)
        assert names[1] == _legacy_name(cluster_b, 1, PLAYLIST_NAME)

    def test_names_are_filesystem_safe_after_sanitise(self) -> None:
        """Every assigned name can be sanitised into a safe filename."""
        cluster_a, features_a, metadata_a = _build_cluster_inputs(
            cluster_id=0,
            num_tracks=3,
            bpm_mean=124.0,
            bpm_std=2.0,
            rms=0.1,
            brightness=0.2,
            mood="Dark",
        )
        cluster_b, features_b, metadata_b = _build_cluster_inputs(
            cluster_id=1,
            num_tracks=3,
            bpm_mean=136.0,
            bpm_std=3.0,
            rms=0.9,
            brightness=0.8,
            mood="Aggressive",
        )
        cluster_c, features_c, metadata_c = _build_cluster_inputs(
            cluster_id=2,
            num_tracks=3,
            bpm_mean=129.0,
            bpm_std=2.0,
            rms=0.5,
            brightness=0.5,
            mood="Ethereal",
        )

        features = {**features_a, **features_b, **features_c}
        metadata = {**metadata_a, **metadata_b, **metadata_c}
        clusters = [cluster_a, cluster_b, cluster_c]

        names = self._assign_cluster_names(clusters, features, metadata, PLAYLIST_NAME)

        for name in names.values():
            sanitized = self._sanitize_filename(name)
            assert sanitized == sanitized.lower()
            assert " " not in sanitized
            assert re.fullmatch(r"[a-z0-9._-]+", sanitized)

    def test_sanitise_examples(self) -> None:
        """Sanitise helper produces the documented filename transformations."""
        assert self._sanitize_filename("Dark Hypnotic [128-131bpm]") == "dark_hypnotic_128-131bpm"
        assert self._sanitize_filename("Peak-Time Driving") == "peak-time_driving"
        weird = self._sanitize_filename("Weird/Name: *Test?*")
        assert "/" not in weird
        assert ":" not in weird
        assert "*" not in weird
        assert "?" not in weird

    def test_names_cover_all_clusters(self) -> None:
        """Every cluster_id appears in the returned name mapping, even without features."""
        cluster_present = _make_cluster(
            cluster_id=7,
            tracks=[Path(f"/music/present{i}.mp3") for i in range(3)],
            bpm_mean=124.0,
            bpm_std=2.0,
        )
        cluster_missing = _make_cluster(
            cluster_id=8,
            tracks=[Path(f"/music/missing{i}.mp3") for i in range(3)],
            bpm_mean=136.0,
            bpm_std=3.0,
        )

        features = {
            t: _make_features(t, rms=0.5, brightness=0.5, mood="Ethereal")
            for t in cluster_present.tracks
        }
        metadata = {
            t: _make_metadata(t, bpm=124.0 if "present" in str(t) else 136.0)
            for cluster in (cluster_present, cluster_missing)
            for t in cluster.tracks
        }

        names = self._assign_cluster_names(
            [cluster_present, cluster_missing],
            features,
            metadata,
            PLAYLIST_NAME,
        )

        assert 7 in names
        assert 8 in names
        assert isinstance(names[7], str)
        assert isinstance(names[8], str)
