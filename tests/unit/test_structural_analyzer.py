"""
Unit tests for structural audio analysis.

Tests for intro/outro detection, drop detection, and cue point prediction
using synthetic audio fixtures.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

# Import after pytest fixtures are defined
from playchitect.core.mixxx_sync import MixxxSync
from playchitect.core.structural_analyzer import (
    StructuralAnalysis,
    StructuralAnalyzer,
    predict_cue_points,
)


@pytest.fixture
def synthetic_intro_outro_audio(tmp_path: Path) -> Path:
    """
    Create a synthetic audio file with distinct intro and outro sections.

    Structure: 5s silence, 10s loud, 5s silence (20s total at 22050 Hz)
    Intro end should be around 5s (5000ms)
    Outro start should be around 15s (15000ms)
    """
    import soundfile as sf

    sr = 22050
    duration = 20.0
    t = np.linspace(0, duration, int(sr * duration))

    # Build the signal: silence -> loud tone -> silence
    silence1_duration = 5.0
    loud_duration = 10.0

    # Create envelope
    envelope = np.zeros_like(t)

    # Silence section 1 (0-5s)
    silence1_samples = int(silence1_duration * sr)

    # Loud section (5-15s) - sine wave at 440Hz
    loud_start = silence1_samples
    loud_samples = int(loud_duration * sr)
    loud_end = loud_start + loud_samples

    # Generate loud tone
    tone = 0.5 * np.sin(2 * np.pi * 440 * t[loud_start:loud_end])
    envelope[loud_start:loud_end] = tone

    # Silence section 2 (15-20s) - already zeros

    filepath = tmp_path / "test_intro_outro.wav"
    sf.write(filepath, envelope, sr, subtype="PCM_16")

    return filepath


@pytest.fixture
def synthetic_drop_audio(tmp_path: Path) -> Path:
    """
    Create a synthetic audio file with energy peaks (drops).

    Structure: 2s baseline, 3s high energy drop, 2s baseline, 3s high energy drop
    Should detect 2 drops
    """
    import soundfile as sf

    sr = 22050
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create signal with two high-energy bursts
    envelope = np.ones_like(t) * 0.1  # Low baseline

    # First drop at 2-5s
    drop1_start = int(2.0 * sr)
    drop1_end = int(5.0 * sr)
    envelope[drop1_start:drop1_end] = 0.8

    # Second drop at 7-10s
    drop2_start = int(7.0 * sr)
    drop2_end = int(10.0 * sr)
    envelope[drop2_start:drop2_end] = 0.8

    # Add some high-frequency content
    signal = envelope * np.sin(2 * np.pi * 440 * t)

    filepath = tmp_path / "test_drops.wav"
    sf.write(filepath, signal, sr, subtype="PCM_16")

    return filepath


class TestStructuralAnalyzer:
    """Tests for the StructuralAnalyzer class."""

    def test_analyze_returns_structural_analysis(self, synthetic_intro_outro_audio: Path):
        """Test that analyze() returns a StructuralAnalysis object."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(synthetic_intro_outro_audio)

        assert isinstance(result, StructuralAnalysis)

    def test_analyze_detects_intro_end(self, synthetic_intro_outro_audio: Path):
        """Test that intro_end_ms is detected as a positive float."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(synthetic_intro_outro_audio)

        # Intro end should be around 5000ms (5 seconds)
        assert result.intro_end_ms > 0
        assert result.intro_end_ms > 4000  # At least 4 seconds
        assert result.intro_end_ms < 6000  # But less than 6 seconds

    def test_analyze_detects_outro_start(self, synthetic_intro_outro_audio: Path):
        """Test that outro_start_ms is detected as a positive float."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(synthetic_intro_outro_audio)

        # Outro start should be around 15000ms (15 seconds)
        assert result.outro_start_ms > 0
        assert result.outro_start_ms > 14000  # At least 14 seconds
        assert result.outro_start_ms < 16000  # But less than 16 seconds

    def test_analyze_intro_less_than_outro(self, synthetic_intro_outro_audio: Path):
        """Test that intro end is before outro start."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(synthetic_intro_outro_audio)

        assert result.intro_end_ms < result.outro_start_ms

    def test_analyze_detects_drops(self, synthetic_drop_audio: Path):
        """Test that drops are detected."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(synthetic_drop_audio)

        # Should detect at least one drop
        assert len(result.drops) >= 1

        # Drops should be positive timestamps
        for drop in result.drops:
            assert drop > 0

    def test_analyze_breakdowns_is_list(self, synthetic_intro_outro_audio: Path):
        """Test that breakdowns field is a list (even if empty)."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(synthetic_intro_outro_audio)

        assert isinstance(result.breakdowns, list)

    def test_analyze_raises_on_nonexistent_file(self):
        """Test that analyze() raises FileNotFoundError for missing file."""
        analyzer = StructuralAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.analyze(Path("/nonexistent/path/to/file.wav"))

    def test_analyze_handles_short_file(self, tmp_path: Path):
        """Test handling of very short audio files."""
        import soundfile as sf

        # Create a 0.5 second file
        sr = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        filepath = tmp_path / "short.wav"
        sf.write(filepath, signal, sr, subtype="PCM_16")

        analyzer = StructuralAnalyzer()

        # Should either return a valid result or raise ValueError
        try:
            result = analyzer.analyze(filepath)
            assert isinstance(result, StructuralAnalysis)
        except ValueError:
            pass  # Also acceptable for very short files


class TestPredictCuePoints:
    """Tests for the predict_cue_points function."""

    def test_predict_cue_points_returns_dict(self):
        """Test that predict_cue_points returns a dictionary."""
        analysis = StructuralAnalysis(
            intro_end_ms=5000.0,
            outro_start_ms=15000.0,
            breakdowns=[],
            drops=[],
        )

        result = predict_cue_points(analysis)

        assert isinstance(result, dict)

    def test_predict_cue_points_has_required_keys(self):
        """Test that result has cue_1_ms and cue_2_ms keys."""
        analysis = StructuralAnalysis(
            intro_end_ms=5000.0,
            outro_start_ms=15000.0,
            breakdowns=[],
            drops=[],
        )

        result = predict_cue_points(analysis)

        assert "cue_1_ms" in result
        assert "cue_2_ms" in result

    def test_predict_cue_points_values_are_floats(self):
        """Test that cue point values are floats."""
        analysis = StructuralAnalysis(
            intro_end_ms=5000.0,
            outro_start_ms=15000.0,
            breakdowns=[],
            drops=[],
        )

        result = predict_cue_points(analysis)

        assert isinstance(result["cue_1_ms"], float)
        assert isinstance(result["cue_2_ms"], float)

    def test_predict_cue_points_maps_correctly(self):
        """Test that cue_1_ms is intro_end and cue_2_ms is outro_start."""
        analysis = StructuralAnalysis(
            intro_end_ms=1234.5,
            outro_start_ms=9876.5,
            breakdowns=[],
            drops=[],
        )

        result = predict_cue_points(analysis)

        assert result["cue_1_ms"] == 1234.5
        assert result["cue_2_ms"] == 9876.5


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_end_to_end_analysis_and_prediction(self, synthetic_intro_outro_audio: Path):
        """Test the full workflow from analysis to cue prediction."""
        # Step 1: Analyze
        analyzer = StructuralAnalyzer()
        analysis = analyzer.analyze(synthetic_intro_outro_audio)

        # Step 2: Predict cues
        cue_points = predict_cue_points(analysis)

        # Step 3: Verify structure
        assert "cue_1_ms" in cue_points
        assert "cue_2_ms" in cue_points
        assert cue_points["cue_1_ms"] > 0
        assert cue_points["cue_2_ms"] > 0
        assert cue_points["cue_1_ms"] < cue_points["cue_2_ms"]


def _make_mixxx_db(db_path: Path, track_path: Path | None = None) -> None:
    """Create a minimal Mixxx-schema SQLite DB at db_path."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE track_locations (
            id INTEGER PRIMARY KEY,
            location TEXT NOT NULL,
            fs_deleted INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            type INTEGER NOT NULL,
            position INTEGER NOT NULL,
            length INTEGER NOT NULL DEFAULT 0,
            hotcue INTEGER,
            label TEXT
        );
    """)
    if track_path is not None:
        conn.execute(
            "INSERT INTO track_locations VALUES (1, ?, 0)",
            (str(track_path.resolve()),),
        )
    conn.commit()
    conn.close()


class TestWriteCuePoints:
    """Tests for MixxxSync.write_cue_points against an in-memory Mixxx DB."""

    def test_write_cue_points_inserts_rows(self, tmp_path: Path) -> None:
        """write_cue_points inserts cue rows with type=1 and hotcue offset 4."""
        db_path = tmp_path / "mixxxdb.sqlite"
        track_path = tmp_path / "fake.mp3"
        track_path.touch()
        _make_mixxx_db(db_path, track_path)

        sync = MixxxSync()
        count = sync.write_cue_points(
            db_path, track_path, {"cue_1_ms": 5000.0, "cue_2_ms": 175000.0}
        )

        assert count == 2
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT type, hotcue FROM cues ORDER BY hotcue").fetchall()
        conn.close()
        assert len(rows) == 2
        assert all(r[0] == 1 for r in rows)  # type == 1 (hot cue)
        assert rows[0][1] == 4  # first cue at offset 4
        assert rows[1][1] == 5  # second cue at offset 5

    def test_write_cue_points_file_not_found(self, tmp_path: Path) -> None:
        """write_cue_points raises FileNotFoundError for a missing DB."""
        sync = MixxxSync()
        with pytest.raises(FileNotFoundError):
            sync.write_cue_points(
                tmp_path / "nonexistent.sqlite",
                tmp_path / "fake.mp3",
                {"cue_1_ms": 1000.0},
            )

    def test_write_cue_points_track_not_in_library(self, tmp_path: Path) -> None:
        """write_cue_points raises ValueError when track absent from track_locations."""
        db_path = tmp_path / "empty.sqlite"
        _make_mixxx_db(db_path)  # no tracks inserted

        sync = MixxxSync()
        with pytest.raises(ValueError, match="not found in Mixxx library"):
            sync.write_cue_points(
                db_path,
                tmp_path / "missing_track.mp3",
                {"cue_1_ms": 1000.0},
            )
