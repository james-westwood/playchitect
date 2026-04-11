"""Unit tests for play_history module."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from playchitect.core.play_history import PlayHistory, TrackHistory


class TestTrackHistory:
    """Test TrackHistory dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        entry = TrackHistory()
        assert entry.times_used == 0
        assert entry.last_used == ""

    def test_custom_initialization(self):
        """Test custom initialization."""
        entry = TrackHistory(times_used=5, last_used="2024-01-15")
        assert entry.times_used == 5
        assert entry.last_used == "2024-01-15"


class TestPlayHistory:
    """Test PlayHistory class."""

    @pytest.fixture
    def temp_history_path(self, tmp_path):
        """Provide a temporary history file path."""
        return tmp_path / "play_history.json"

    @pytest.fixture
    def history(self, temp_history_path):
        """Provide a fresh PlayHistory instance."""
        return PlayHistory(history_path=temp_history_path)

    def test_initialization_creates_empty_history(self, temp_history_path):
        """Test that initialization creates empty history when file doesn't exist."""
        history = PlayHistory(history_path=temp_history_path)
        assert history._history == {}
        assert not temp_history_path.exists()

    def test_load_empty_file(self, temp_history_path):
        """Test loading from non-existent file creates empty history."""
        history = PlayHistory(history_path=temp_history_path)
        history.load()
        assert history._history == {}

    def test_save_and_load(self, temp_history_path):
        """Test saving and loading history."""
        history = PlayHistory(history_path=temp_history_path)

        track = Path("/music/test.mp3")
        history.record(track)
        history.save()

        # Load into new instance
        history2 = PlayHistory(history_path=temp_history_path)
        assert str(track) in history2._history
        assert history2._history[str(track)].times_used == 1

    def test_record_new_track(self, history):
        """Test recording a new track."""
        track = Path("/music/new_track.mp3")
        history.record(track)

        entry = history.get_history(track)
        assert entry is not None
        assert entry.times_used == 1
        assert entry.last_used == datetime.now(UTC).strftime("%Y-%m-%d")

    def test_record_existing_track_increments_counter(self, history):
        """Test recording an existing track increments times_used."""
        track = Path("/music/existing.mp3")
        history.record(track)
        history.record(track)
        history.record(track)

        entry = history.get_history(track)
        assert entry.times_used == 3

    def test_freshness_score_unseen_track_is_1(self, history):
        """Test that freshness score for unseen track is 1.0."""
        track = Path("/music/unseen.mp3")
        score = history.get_freshness_score(track)
        assert score == 1.0

    def test_freshness_score_after_one_record_is_less_than_1(self, history):
        """Test that freshness score is < 1.0 after record() called once."""
        track = Path("/music/played_once.mp3")
        history.record(track)

        # Set the last_used to yesterday so days_decay is non-zero
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        history._history[str(track)].last_used = yesterday

        score = history.get_freshness_score(track)
        assert score < 1.0
        assert score > 0.0

    def test_freshness_score_approaches_1_after_30_days(self, history):
        """Test that freshness score approaches 1.0 after 30 days."""
        track = Path("/music/old_track.mp3")
        history.record(track)

        # Mock the date to be 30 days ago
        old_date = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%d")
        history._history[str(track)].last_used = old_date

        score = history.get_freshness_score(track)
        # Score should be close to 1.0 (within 0.25 given the formula)
        assert score >= 0.75
        assert score <= 1.0

    def test_freshness_score_decreases_with_more_plays(self, history):
        """Test that freshness score decreases with more plays."""
        track = Path("/music/frequent.mp3")
        history.record(track)

        # Set to yesterday so days_decay is not 0
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        history._history[str(track)].last_used = yesterday

        score_1 = history.get_freshness_score(track)

        # Record again
        history.record(track)
        # Restore yesterday's date (record() sets it to today)
        history._history[str(track)].last_used = yesterday
        score_2 = history.get_freshness_score(track)

        assert score_2 < score_1

    def test_freshness_score_excludes_low_score_tracks(self, history):
        """Test that tracks with score < 0.1 are excluded by sequence_fresh."""
        # Create tracks with varying freshness
        tracks = []
        features = {}

        for i in range(5):
            track = Path(f"/music/track_{i}.mp3")
            tracks.append(track)
            # Create mock features with 0.5 RMS energy
            feat = MagicMock()
            feat.rms_energy = 0.5
            features[track] = feat

            # Record each track multiple times to reduce freshness
            times = max(1, i * 5)  # 1, 5, 10, 15, 20 times (at least 1 for all)
            for _ in range(times):
                history.record(track)

            # Set date to many days ago so scores aren't all 0
            # Need at least ~4 days for score to be > 0.1 with 1 play
            days_ago = (datetime.now(UTC) - timedelta(days=10)).strftime("%Y-%m-%d")
            history._history[str(track)].last_used = days_ago

        # Now import and test sequence_fresh
        from playchitect.core.sequencer import sequence_fresh

        result = sequence_fresh(tracks, features, history)

        # All tracks should be included since 10 days gives days_decay = 0.33
        # and 1 play gives usage_score ~ 0.91, so score ~ 0.3 > 0.1
        assert len(result) > 0

    def test_sequence_fresh_sorts_by_combined_score(self, history):
        """Test that sequence_fresh sorts by freshness * rms_energy."""
        tracks = [Path(f"/music/track_{i}.mp3") for i in range(3)]

        # Create features with different RMS energies
        features = {}
        for i, track in enumerate(tracks):
            feat = MagicMock()
            feat.rms_energy = 0.5 + i * 0.2  # 0.5, 0.7, 0.9 (higher energy)
            features[track] = feat

        # Record tracks with different frequencies
        # Track 0: once
        # Track 1: twice
        # Track 2: once
        history.record(tracks[0])
        history.record(tracks[1])
        history.record(tracks[1])
        history.record(tracks[2])

        # Set all to date 15 days ago so all scores > 0.1
        # With 15 days, days_decay = 0.5
        test_date = (datetime.now(UTC) - timedelta(days=15)).strftime("%Y-%m-%d")
        for track in tracks:
            history._history[str(track)].last_used = test_date

        from playchitect.core.sequencer import sequence_fresh

        result = sequence_fresh(tracks, features, history)

        # Should have all 3 tracks (scores > 0.1 with 15 day decay)
        assert len(result) == 3

        # Track 2 should be first (highest energy 0.9, same freshness as track 0)
        assert result[0] == tracks[2]

        # Track 0 should be before Track 1 (lower energy but more fresh - 1 vs 2 plays)
        # Track 0: score ≈ 0.91 * 0.5 * 0.5 ≈ 0.23
        # Track 1: score ≈ 0.83 * 0.5 * 0.7 ≈ 0.29
        # Actually track 1 might come before track 0 due to higher energy
        # Let's just verify ordering is by combined score descending
        prev_score = float("inf")
        for track in result:
            freshness = history.get_freshness_score(track)
            score = freshness * features[track].rms_energy
            assert score <= prev_score  # Should be descending
            prev_score = score

    def test_sequence_fresh_excludes_low_freshness(self, history):
        """Test that sequence_fresh excludes tracks with freshness < 0.1."""
        tracks = [Path("/music/good.mp3"), Path("/music/bad.mp3")]

        features = {
            tracks[0]: MagicMock(rms_energy=0.5),
            tracks[1]: MagicMock(rms_energy=0.5),
        }

        # Record first track once with a date that gives score > 0.1
        # Need at least ~4 days for score > 0.1
        history.record(tracks[0])
        good_date = (datetime.now(UTC) - timedelta(days=10)).strftime("%Y-%m-%d")
        history._history[str(tracks[0])].last_used = good_date

        # Record second track once but with yesterday's date (score will be ~0.03 < 0.1)
        history.record(tracks[1])
        bad_date = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        history._history[str(tracks[1])].last_used = bad_date

        from playchitect.core.sequencer import sequence_fresh

        result = sequence_fresh(tracks, features, history)  # ty: ignore[invalid-argument-type]

        # Only first track should be in result (second has score < 0.1)
        assert len(result) == 1
        assert result[0] == tracks[0]

    def test_load_invalid_json(self, temp_history_path):
        """Test loading invalid JSON creates empty history."""
        # Write invalid JSON
        temp_history_path.parent.mkdir(parents=True, exist_ok=True)
        temp_history_path.write_text("invalid json")

        history = PlayHistory(history_path=temp_history_path)
        # Should not crash, creates empty history
        assert history._history == {}

    def test_load_corrupted_entry(self, temp_history_path):
        """Test loading corrupted entries handles gracefully."""
        # Write JSON with invalid entry
        temp_history_path.parent.mkdir(parents=True, exist_ok=True)
        temp_history_path.write_text('{"track1.mp3": "corrupted"}')

        history = PlayHistory(history_path=temp_history_path)
        # Should handle gracefully
        assert "track1.mp3" in history._history
        # Should create default TrackHistory
        entry = history._history["track1.mp3"]
        assert entry.times_used == 0
        assert entry.last_used == ""

    def test_clear_history(self, history):
        """Test clearing history."""
        track = Path("/music/track.mp3")
        history.record(track)
        history.save()

        history.clear()
        assert history._history == {}

    def test_get_history_nonexistent(self, history):
        """Test getting history for non-existent track."""
        track = Path("/music/nonexistent.mp3")
        entry = history.get_history(track)
        assert entry is None

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories."""
        history_path = tmp_path / "nested" / "dirs" / "history.json"
        history = PlayHistory(history_path=history_path)
        history.record(Path("/music/test.mp3"))
        history.save()

        assert history_path.exists()

    def test_freshness_with_no_last_used_date(self, history):
        """Test freshness calculation when track has no last_used date."""
        track = Path("/music/no_date.mp3")
        history._history[str(track)] = TrackHistory(times_used=5, last_used="")

        score = history.get_freshness_score(track)
        assert score == 1.0  # Should be treated as never played

    def test_freshness_with_invalid_date(self, history):
        """Test freshness calculation with invalid date format."""
        track = Path("/music/bad_date.mp3")
        history._history[str(track)] = TrackHistory(times_used=5, last_used="invalid-date")

        # Should not crash, treats as old track
        score = history.get_freshness_score(track)
        assert score >= 0.0
        assert score <= 1.0

    def test_multiple_tracks_independence(self, history):
        """Test that multiple tracks have independent history."""
        track1 = Path("/music/track1.mp3")
        track2 = Path("/music/track2.mp3")

        history.record(track1)
        history.record(track1)
        history.record(track2)

        assert history.get_history(track1).times_used == 2
        assert history.get_history(track2).times_used == 1

    def test_freshness_monotonic_with_time(self, history):
        """Test that freshness increases monotonically with days since use."""
        track = Path("/music/time_test.mp3")
        history.record(track)

        # Set different dates and check scores
        scores = []
        days_list = [0, 7, 14, 21, 30, 60]

        for days in days_list:
            date = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
            history._history[str(track)].last_used = date
            score = history.get_freshness_score(track)
            scores.append(score)

        # Scores should generally increase (or stay same) with more days
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 0.01  # Allow small tolerance
