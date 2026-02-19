"""
Unit tests for playchitect.core.track_selector.
"""

from pathlib import Path

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.track_selector import TrackScore, TrackSelection, TrackSelector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intensity(
    path: Path,
    *,
    onset_strength: float = 0.5,
    percussiveness: float = 0.5,
    kick_energy: float = 0.5,
    rms_energy: float = 0.5,
    brightness: float = 0.5,
    sub_bass_energy: float = 0.5,
    bass_harmonics: float = 0.5,
) -> IntensityFeatures:
    return IntensityFeatures(
        filepath=path,
        file_hash="abc123",
        rms_energy=rms_energy,
        brightness=brightness,
        sub_bass_energy=sub_bass_energy,
        kick_energy=kick_energy,
        bass_harmonics=bass_harmonics,
        percussiveness=percussiveness,
        onset_strength=onset_strength,
    )


def _make_metadata(path: Path, *, duration: float = 200.0, bpm: float = 128.0) -> TrackMetadata:
    return TrackMetadata(filepath=path, bpm=bpm, duration=duration)


def _make_cluster(tracks: list[Path], cluster_id: int = 0) -> ClusterResult:
    return ClusterResult(
        cluster_id=cluster_id,
        tracks=tracks,
        bpm_mean=128.0,
        bpm_std=2.0,
        track_count=len(tracks),
        total_duration=float(len(tracks) * 300),
    )


# ---------------------------------------------------------------------------
# TestTrackScore
# ---------------------------------------------------------------------------


class TestTrackScore:
    def test_construction(self, tmp_path: Path) -> None:
        p = tmp_path / "track.flac"
        ts = TrackScore(path=p, score=0.75, reason="low onset + quiet start")
        assert ts.path == p
        assert ts.score == 0.75
        assert ts.reason == "low onset + quiet start"


# ---------------------------------------------------------------------------
# TestTrackSelection
# ---------------------------------------------------------------------------


class TestTrackSelection:
    def _make_selection(
        self,
        tmp_path: Path,
        *,
        override_first: Path | None = None,
        override_last: Path | None = None,
    ) -> tuple["TrackSelection", Path, Path]:
        p1 = tmp_path / "a.flac"
        p2 = tmp_path / "b.flac"
        first_tracks = [TrackScore(path=p1, score=0.9, reason="low onset")]
        last_tracks = [TrackScore(path=p2, score=0.8, reason="energetic closer")]
        sel = TrackSelection(
            cluster_id=0,
            first_tracks=first_tracks,
            last_tracks=last_tracks,
            user_override_first=override_first,
            user_override_last=override_last,
        )
        return sel, p1, p2

    def test_selected_first_no_override(self, tmp_path: Path) -> None:
        sel, p1, _ = self._make_selection(tmp_path)
        assert sel.selected_first == p1

    def test_selected_last_no_override(self, tmp_path: Path) -> None:
        sel, _, p2 = self._make_selection(tmp_path)
        assert sel.selected_last == p2

    def test_selected_first_with_override(self, tmp_path: Path) -> None:
        override = tmp_path / "override.flac"
        sel, _, _ = self._make_selection(tmp_path, override_first=override)
        assert sel.selected_first == override

    def test_selected_last_with_override(self, tmp_path: Path) -> None:
        override = tmp_path / "override.flac"
        sel, _, _ = self._make_selection(tmp_path, override_last=override)
        assert sel.selected_last == override

    def test_override_takes_precedence_over_top_score(self, tmp_path: Path) -> None:
        override = tmp_path / "user_pick.flac"
        sel, p1, _ = self._make_selection(tmp_path, override_first=override)
        assert sel.selected_first != p1
        assert sel.selected_first == override


# ---------------------------------------------------------------------------
# TestTrackSelector
# ---------------------------------------------------------------------------


class TestTrackSelector:
    def test_low_onset_scores_higher_as_opener(self, tmp_path: Path) -> None:
        quiet = tmp_path / "quiet.flac"
        loud = tmp_path / "loud.flac"
        tracks = [quiet, loud]
        cluster = _make_cluster(tracks)
        metadata = {t: _make_metadata(t) for t in tracks}
        intensity = {
            quiet: _make_intensity(
                quiet, onset_strength=0.1, percussiveness=0.1, kick_energy=0.1, rms_energy=0.1
            ),
            loud: _make_intensity(
                loud, onset_strength=0.9, percussiveness=0.9, kick_energy=0.9, rms_energy=0.9
            ),
        }

        sel = TrackSelector().select(cluster, metadata, intensity)
        assert sel.first_tracks[0].path == quiet

    def test_long_track_gets_duration_bonus_for_opener(self, tmp_path: Path) -> None:
        long_track = tmp_path / "long.flac"
        short_track = tmp_path / "short.flac"
        tracks = [long_track, short_track]
        cluster = _make_cluster(tracks)
        # Both tracks have identical intensity
        avg = _make_intensity(
            long_track, onset_strength=0.5, percussiveness=0.5, kick_energy=0.5, rms_energy=0.5
        )
        avg2 = _make_intensity(
            short_track, onset_strength=0.5, percussiveness=0.5, kick_energy=0.5, rms_energy=0.5
        )
        metadata = {
            long_track: _make_metadata(long_track, duration=350.0),  # >= 300 s → bonus
            short_track: _make_metadata(short_track, duration=200.0),
        }
        intensity = {long_track: avg, short_track: avg2}

        sel = TrackSelector().select(cluster, metadata, intensity)
        # Long track should have a higher opener score due to duration bonus
        long_score = next(s for s in sel.first_tracks if s.path == long_track)
        short_score = next(s for s in sel.first_tracks if s.path == short_track)
        assert long_score.score > short_score.score

    def test_high_energy_track_scores_higher_as_energetic_closer(self, tmp_path: Path) -> None:
        energetic = tmp_path / "energetic.flac"
        quiet = tmp_path / "quiet.flac"
        tracks = [energetic, quiet]
        cluster = _make_cluster(tracks)
        metadata = {t: _make_metadata(t) for t in tracks}
        intensity = {
            energetic: _make_intensity(
                energetic, rms_energy=0.9, onset_strength=0.9, kick_energy=0.9
            ),
            quiet: _make_intensity(quiet, rms_energy=0.1, onset_strength=0.1, kick_energy=0.1),
        }

        sel = TrackSelector().select(cluster, metadata, intensity)
        assert sel.last_tracks[0].path == energetic

    def test_long_quiet_track_wins_as_smooth_fade_closer(self, tmp_path: Path) -> None:
        fade = tmp_path / "fade.flac"
        energetic = tmp_path / "energetic.flac"
        tracks = [fade, energetic]
        cluster = _make_cluster(tracks)
        metadata = {
            fade: _make_metadata(fade, duration=400.0),  # >= 360 s → fade bonus
            energetic: _make_metadata(energetic, duration=200.0),
        }
        intensity = {
            # fade: very low onset and rms — fade score wins with duration bonus
            fade: _make_intensity(fade, onset_strength=0.05, rms_energy=0.05, kick_energy=0.05),
            # energetic: middling values so it can't beat the fade track
            energetic: _make_intensity(
                energetic, rms_energy=0.5, onset_strength=0.5, kick_energy=0.5
            ),
        }

        sel = TrackSelector().select(cluster, metadata, intensity)
        # fade should win (smooth fade-out)
        assert sel.last_tracks[0].path == fade
        assert sel.last_tracks[0].reason == "smooth fade-out"

    def test_user_override_first_is_returned(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.flac"
        p2 = tmp_path / "b.flac"
        tracks = [p1, p2]
        cluster = _make_cluster(tracks)
        metadata = {t: _make_metadata(t) for t in tracks}
        intensity = {t: _make_intensity(t) for t in tracks}

        sel = TrackSelector().select(cluster, metadata, intensity, user_override_first=p2)
        assert sel.selected_first == p2

    def test_user_override_last_is_returned(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.flac"
        p2 = tmp_path / "b.flac"
        tracks = [p1, p2]
        cluster = _make_cluster(tracks)
        metadata = {t: _make_metadata(t) for t in tracks}
        intensity = {t: _make_intensity(t) for t in tracks}

        sel = TrackSelector().select(cluster, metadata, intensity, user_override_last=p1)
        assert sel.selected_last == p1

    def test_override_not_in_cluster_raises_value_error(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.flac"
        outsider = tmp_path / "outsider.flac"
        cluster = _make_cluster([p1])
        metadata = {p1: _make_metadata(p1)}
        intensity = {p1: _make_intensity(p1)}

        with pytest.raises(ValueError, match="not in cluster"):
            TrackSelector().select(cluster, metadata, intensity, user_override_first=outsider)

    def test_override_last_not_in_cluster_raises_value_error(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.flac"
        outsider = tmp_path / "outsider.flac"
        cluster = _make_cluster([p1])
        metadata = {p1: _make_metadata(p1)}
        intensity = {p1: _make_intensity(p1)}

        with pytest.raises(ValueError, match="not in cluster"):
            TrackSelector().select(cluster, metadata, intensity, user_override_last=outsider)

    def test_single_track_cluster_handled_gracefully(self, tmp_path: Path) -> None:
        p1 = tmp_path / "only.flac"
        cluster = _make_cluster([p1])
        metadata = {p1: _make_metadata(p1)}
        intensity = {p1: _make_intensity(p1)}

        sel = TrackSelector().select(cluster, metadata, intensity)
        assert sel.selected_first == p1
        assert sel.selected_last == p1
        assert len(sel.first_tracks) == 1
        assert sel.first_tracks[0].reason == "only track in cluster"

    def test_track_missing_from_intensity_dict_is_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        p1 = tmp_path / "a.flac"
        p2 = tmp_path / "b.flac"
        tracks = [p1, p2]
        cluster = _make_cluster(tracks)
        metadata = {t: _make_metadata(t) for t in tracks}
        # Only p1 has intensity data; p2 is missing
        intensity = {p1: _make_intensity(p1)}

        import logging

        with caplog.at_level(logging.WARNING):
            sel = TrackSelector().select(cluster, metadata, intensity)

        # Both tracks appear in results; missing track gets score=-1.0 and sorts last
        assert len(sel.first_tracks) == 2
        assert sel.first_tracks[0].path == p1
        assert sel.first_tracks[1].path == p2
        assert sel.first_tracks[1].score == -1.0
        assert "missing intensity data" in sel.first_tracks[1].reason
        assert any("missing from intensity_dict" in rec.message for rec in caplog.records)

    def test_top_n_3_returns_three_candidates(self, tmp_path: Path) -> None:
        tracks = [tmp_path / f"{i}.flac" for i in range(5)]
        cluster = _make_cluster(tracks)
        metadata = {t: _make_metadata(t) for t in tracks}
        intensity = {
            t: _make_intensity(
                t,
                onset_strength=float(i) * 0.15,
                percussiveness=float(i) * 0.1,
                kick_energy=float(i) * 0.1,
                rms_energy=float(i) * 0.1,
            )
            for i, t in enumerate(tracks)
        }

        sel = TrackSelector(top_n=3).select(cluster, metadata, intensity)
        assert len(sel.first_tracks) == 3
        assert len(sel.last_tracks) == 3

    def test_opener_score_in_range(self, tmp_path: Path) -> None:
        p = tmp_path / "track.flac"
        cluster = _make_cluster([p, tmp_path / "other.flac"])
        other = tmp_path / "other.flac"
        metadata = {t: _make_metadata(t) for t in [p, other]}
        intensity = {
            p: _make_intensity(
                p, onset_strength=0.0, percussiveness=0.0, kick_energy=0.0, rms_energy=0.0
            ),
            other: _make_intensity(other),
        }

        sel = TrackSelector().select(cluster, metadata, intensity)
        for ts in sel.first_tracks:
            assert 0.0 <= ts.score <= 1.0

    def test_closer_score_in_range(self, tmp_path: Path) -> None:
        p = tmp_path / "track.flac"
        other = tmp_path / "other.flac"
        cluster = _make_cluster([p, other])
        metadata = {t: _make_metadata(t) for t in [p, other]}
        intensity = {
            p: _make_intensity(p, rms_energy=1.0, onset_strength=1.0, kick_energy=1.0),
            other: _make_intensity(other),
        }

        sel = TrackSelector().select(cluster, metadata, intensity)
        for ts in sel.last_tracks:
            assert 0.0 <= ts.score <= 1.0
