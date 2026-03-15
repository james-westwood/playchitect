"""Unit tests for compatibility scoring module.

Tests for track compatibility scoring and next-track suggestions,
verifying the weighted combination of BPM, key, energy, and timbre scores.
"""

from pathlib import Path

from playchitect.core.compatibility import compatibility_score, next_track_suggestions
from playchitect.core.intensity_analyzer import IntensityFeatures


class TestCompatibilityScore:
    """Tests for the compatibility_score function."""

    def test_high_compatibility_same_bpm_compatible_key_same_energy(self) -> None:
        """Tracks with same BPM, compatible key, and same energy score > 0.8."""
        features_a = IntensityFeatures(
            filepath=Path("track_a.mp3"),
            file_hash="hash_a",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="8B",
            key_index=0.0,
        )

        features_b = IntensityFeatures(
            filepath=Path("track_b.mp3"),
            file_hash="hash_b",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="8B",  # Same number = compatible
            key_index=0.0,
        )

        score = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=128.0)

        assert score > 0.8, f"Expected score > 0.8 for identical tracks, got {score}"
        assert score <= 1.0, f"Score should not exceed 1.0, got {score}"

    def test_high_compatibility_adjacent_camelot_key(self) -> None:
        """Tracks with adjacent Camelot keys (same letter) are compatible."""
        features_a = IntensityFeatures(
            filepath=Path("track_a.mp3"),
            file_hash="hash_a",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="8B",
            key_index=0.0,
        )

        features_b = IntensityFeatures(
            filepath=Path("track_b.mp3"),
            file_hash="hash_b",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="9B",  # Adjacent number, same letter = compatible
            key_index=2.0,
        )

        score = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=128.0)

        # Should have high key score (1.0), same BPM (1.0), same energy (1.0), same timbre (1.0)
        # Expected: 0.3*1.0 + 0.3*1.0 + 0.25*1.0 + 0.15*1.0 = 1.0
        assert score > 0.8, f"Expected score > 0.8 for adjacent Camelot keys, got {score}"

    def test_low_compatibility_incompatible_key_large_bpm_diff(self) -> None:
        """Tracks with incompatible key and large BPM difference score < 0.4."""
        features_a = IntensityFeatures(
            filepath=Path("track_a.mp3"),
            file_hash="hash_a",
            rms_energy=0.9,  # High energy
            brightness=0.8,
            sub_bass_energy=0.7,
            kick_energy=0.8,
            bass_harmonics=0.6,
            percussiveness=0.9,
            onset_strength=0.8,
            camelot_key="8B",
            key_index=0.0,
        )

        features_b = IntensityFeatures(
            filepath=Path("track_b.mp3"),
            file_hash="hash_b",
            rms_energy=0.1,  # Low energy - different
            brightness=0.2,  # Different timbre
            sub_bass_energy=0.1,
            kick_energy=0.2,
            bass_harmonics=0.1,
            percussiveness=0.2,
            onset_strength=0.1,
            camelot_key="3B",  # Incompatible key
            key_index=1.0,
        )

        # Large BPM difference (40 BPM)
        score = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=88.0)

        # BPM score: max(0, 1 - 40/20) = 0.0
        # Key score: 0.0 (incompatible)
        # Energy score: 1 - |0.9 - 0.1| = 0.2
        # Timbre score: 1 - |0.8 - 0.2| = 0.4
        # Final: 0.3*0 + 0.3*0 + 0.25*0.2 + 0.15*0.4 = 0.05 + 0.06 = 0.11
        assert score < 0.4, f"Expected score < 0.4 for incompatible tracks, got {score}"

    def test_bpm_score_calculation(self) -> None:
        """Test BPM score component calculation."""
        features_a = IntensityFeatures(
            filepath=Path("track_a.mp3"),
            file_hash="hash_a",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        features_b = IntensityFeatures(
            filepath=Path("track_b.mp3"),
            file_hash="hash_b",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",  # Same key for max key score
            key_index=0.0,
        )

        # Same BPM: score = 1.0
        score_same = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=128.0)
        assert score_same == 1.0, f"Expected 1.0 for identical tracks, got {score_same}"

        # 10 BPM diff: score = 1 - 10/20 = 0.5
        score_10 = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=138.0)
        # Key: 1.0, BPM: 0.5, Energy: 1.0, Timbre: 1.0
        # Expected: 0.3*0.5 + 0.3*1.0 + 0.25*1.0 + 0.15*1.0 = 0.15 + 0.3 + 0.25 + 0.15 = 0.85
        expected_10 = 0.3 * 0.5 + 0.3 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0
        assert abs(score_10 - expected_10) < 0.001, f"Expected {expected_10}, got {score_10}"

        # 20 BPM diff: score = 1 - 20/20 = 0.0
        score_20 = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=148.0)
        # Expected: 0.3*0 + 0.3*1.0 + 0.25*1.0 + 0.15*1.0 = 0.7
        expected_20 = 0.3 * 0.0 + 0.3 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0
        assert abs(score_20 - expected_20) < 0.001, f"Expected {expected_20}, got {score_20}"

        # 30 BPM diff: score = max(0, 1 - 30/20) = 0.0 (clamped)
        score_30 = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=158.0)
        assert score_30 == score_20, "BPM diff > 20 should be clamped to 0"

    def test_key_score_same_number(self) -> None:
        """Same Camelot number (different letter) is compatible."""
        features_a = IntensityFeatures(
            filepath=Path("track_a.mp3"),
            file_hash="hash_a",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8A",  # A version
            key_index=0.0,
        )

        features_b = IntensityFeatures(
            filepath=Path("track_b.mp3"),
            file_hash="hash_b",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",  # B version - same number, compatible
            key_index=0.0,
        )

        score = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=128.0)
        assert score == 1.0, f"Same number different letter should be compatible, got {score}"

    def test_energy_score_similarity(self) -> None:
        """Test that energy difference affects score."""
        features_a = IntensityFeatures(
            filepath=Path("track_a.mp3"),
            file_hash="hash_a",
            rms_energy=0.8,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        features_b = IntensityFeatures(
            filepath=Path("track_b.mp3"),
            file_hash="hash_b",
            rms_energy=0.2,  # Different energy
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        score = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=128.0)
        # Key: 1.0, BPM: 1.0, Energy: 1 - |0.8 - 0.2| = 0.4, Timbre: 1.0
        # Expected: 0.3*1.0 + 0.3*1.0 + 0.25*0.4 + 0.15*1.0 = 0.3 + 0.3 + 0.1 + 0.15 = 0.85
        expected = 0.3 * 1.0 + 0.3 * 1.0 + 0.25 * 0.4 + 0.15 * 1.0
        assert abs(score - expected) < 0.001, f"Expected {expected}, got {score}"

    def test_timbre_score_similarity(self) -> None:
        """Test that brightness difference affects score."""
        features_a = IntensityFeatures(
            filepath=Path("track_a.mp3"),
            file_hash="hash_a",
            rms_energy=0.5,
            brightness=0.9,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        features_b = IntensityFeatures(
            filepath=Path("track_b.mp3"),
            file_hash="hash_b",
            rms_energy=0.5,
            brightness=0.1,  # Different brightness
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        score = compatibility_score(features_a, features_b, bpm_a=128.0, bpm_b=128.0)
        # Key: 1.0, BPM: 1.0, Energy: 1.0, Timbre: 1 - |0.9 - 0.1| = 0.2
        # Expected: 0.3*1.0 + 0.3*1.0 + 0.25*1.0 + 0.15*0.2 = 0.3 + 0.3 + 0.25 + 0.03 = 0.88
        expected = 0.3 * 1.0 + 0.3 * 1.0 + 0.25 * 1.0 + 0.15 * 0.2
        assert abs(score - expected) < 0.001, f"Expected {expected}, got {score}"


class TestNextTrackSuggestions:
    """Tests for the next_track_suggestions function."""

    def test_returns_exactly_n_results(self) -> None:
        """next_track_suggestions returns exactly n results sorted descending."""
        current_path = Path("current.mp3")
        current_features = IntensityFeatures(
            filepath=current_path,
            file_hash="current_hash",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="8B",
            key_index=0.0,
        )

        # Create 10 candidate tracks
        candidates: list[tuple[Path, IntensityFeatures, float]] = []
        for i in range(10):
            path = Path(f"candidate_{i}.mp3")
            # Vary BPM and energy to get different scores
            bpm = 128.0 + i * 2  # Increasing BPM difference
            features = IntensityFeatures(
                filepath=path,
                file_hash=f"hash_{i}",
                rms_energy=0.5 + i * 0.05,  # Increasing energy difference
                brightness=0.6,
                sub_bass_energy=0.3,
                kick_energy=0.7,
                bass_harmonics=0.4,
                percussiveness=0.8,
                onset_strength=0.65,
                camelot_key="8B",
                key_index=0.0,
            )
            candidates.append((path, features, bpm))

        # Request top 5
        results = next_track_suggestions(
            current_path=current_path,
            current_features=current_features,
            current_bpm=128.0,
            candidates=candidates,
            n=5,
        )

        assert len(results) == 5, f"Expected exactly 5 results, got {len(results)}"

        # Verify sorted descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1], (
                f"Results not sorted descending: {results[i][1]} < {results[i + 1][1]}"
            )

    def test_excludes_current_track(self) -> None:
        """Current track should be excluded from suggestions."""
        current_path = Path("current.mp3")
        current_features = IntensityFeatures(
            filepath=current_path,
            file_hash="current_hash",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="8B",
            key_index=0.0,
        )

        # Include current track in candidates
        candidates: list[tuple[Path, IntensityFeatures, float]] = [
            (current_path, current_features, 128.0),  # Current track (identical)
            (Path("other.mp3"), current_features, 128.0),  # Another identical track
        ]

        results = next_track_suggestions(
            current_path=current_path,
            current_features=current_features,
            current_bpm=128.0,
            candidates=candidates,
            n=5,
        )

        # Should only return the other track, not the current track
        paths = [r[0] for r in results]
        assert current_path not in paths, "Current track should be excluded from suggestions"
        assert Path("other.mp3") in paths, "Other track should be included"

    def test_sorts_descending_by_score(self) -> None:
        """Results are sorted by score in descending order."""
        current_path = Path("current.mp3")
        current_features = IntensityFeatures(
            filepath=current_path,
            file_hash="current_hash",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        # Create candidates with known compatibility levels
        candidates: list[tuple[Path, IntensityFeatures, float]] = [
            # Best: same everything
            (
                Path("best.mp3"),
                IntensityFeatures(
                    filepath=Path("best.mp3"),
                    file_hash="hash1",
                    rms_energy=0.5,
                    brightness=0.5,
                    sub_bass_energy=0.5,
                    kick_energy=0.5,
                    bass_harmonics=0.5,
                    percussiveness=0.5,
                    onset_strength=0.5,
                    camelot_key="8B",
                    key_index=0.0,
                ),
                128.0,
            ),
            # Worst: different key, BPM, energy, brightness
            (
                Path("worst.mp3"),
                IntensityFeatures(
                    filepath=Path("worst.mp3"),
                    file_hash="hash2",
                    rms_energy=0.9,
                    brightness=0.9,
                    sub_bass_energy=0.5,
                    kick_energy=0.5,
                    bass_harmonics=0.5,
                    percussiveness=0.5,
                    onset_strength=0.5,
                    camelot_key="3B",  # Incompatible
                    key_index=0.0,
                ),
                150.0,  # Large BPM diff
            ),
            # Medium: adjacent key, same BPM, different energy/brightness
            (
                Path("medium.mp3"),
                IntensityFeatures(
                    filepath=Path("medium.mp3"),
                    file_hash="hash3",
                    rms_energy=0.6,
                    brightness=0.6,
                    sub_bass_energy=0.5,
                    kick_energy=0.5,
                    bass_harmonics=0.5,
                    percussiveness=0.5,
                    onset_strength=0.5,
                    camelot_key="9B",  # Adjacent, compatible
                    key_index=0.0,
                ),
                128.0,
            ),
        ]

        results = next_track_suggestions(
            current_path=current_path,
            current_features=current_features,
            current_bpm=128.0,
            candidates=candidates,
            n=5,
        )

        # Should be sorted: best, medium, worst
        assert len(results) == 3
        assert results[0][0] == Path("best.mp3"), "Best match should be first"
        assert results[1][0] == Path("medium.mp3"), "Medium match should be second"
        assert results[2][0] == Path("worst.mp3"), "Worst match should be third"

        # Verify descending scores
        assert results[0][1] > results[1][1], "Scores should be descending"
        assert results[1][1] > results[2][1], "Scores should be descending"

    def test_returns_empty_when_no_candidates(self) -> None:
        """Returns empty list when no candidates provided."""
        current_path = Path("current.mp3")
        current_features = IntensityFeatures(
            filepath=current_path,
            file_hash="current_hash",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        results = next_track_suggestions(
            current_path=current_path,
            current_features=current_features,
            current_bpm=128.0,
            candidates=[],
            n=5,
        )

        assert results == [], "Should return empty list when no candidates"

    def test_returns_fewer_than_n_when_not_enough_candidates(self) -> None:
        """Returns all available candidates when fewer than n."""
        current_path = Path("current.mp3")
        current_features = IntensityFeatures(
            filepath=current_path,
            file_hash="current_hash",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        # Include current track in candidates, plus one other
        candidates: list[tuple[Path, IntensityFeatures, float]] = [
            (current_path, current_features, 128.0),  # Current track (excluded)
            (Path("other.mp3"), current_features, 128.0),  # One other track
        ]

        results = next_track_suggestions(
            current_path=current_path,
            current_features=current_features,
            current_bpm=128.0,
            candidates=candidates,
            n=5,
        )

        assert len(results) == 1, "Should return 1 result (2 candidates - 1 current = 1)"

    def test_default_n_is_5(self) -> None:
        """Default n value is 5."""
        current_path = Path("current.mp3")
        current_features = IntensityFeatures(
            filepath=current_path,
            file_hash="current_hash",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

        candidates: list[tuple[Path, IntensityFeatures, float]] = []
        for i in range(10):
            candidates.append((Path(f"{i}.mp3"), current_features, 128.0))

        results = next_track_suggestions(
            current_path=current_path,
            current_features=current_features,
            current_bpm=128.0,
            candidates=candidates,
            # n not specified, should default to 5
        )

        assert len(results) == 5, "Default n should be 5"
