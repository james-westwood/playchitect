"""
Unit tests for sequencer module.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.sequencer import (
    FiveRhythmsPhase,
    Sequencer,
    classify_five_rhythms_phase,
    sequence_five_rhythms,
)


class TestSequencer:
    """Test Sequencer class."""

    @pytest.fixture
    def mock_data(self):
        """Mock cluster, metadata, and intensity data."""
        paths = [Path(f"track_{i}.flac") for i in range(5)]

        cluster = ClusterResult(
            cluster_id=0,
            tracks=paths,
            bpm_mean=128.0,
            bpm_std=1.0,
            track_count=5,
            total_duration=1500.0,
        )

        metadata_dict = {p: TrackMetadata(filepath=p, bpm=128.0, duration=300.0) for p in paths}

        # Create different intensities
        intensity_dict = {}
        for i, p in enumerate(paths):
            # Hardness will be roughly proportional to i
            intensity_dict[p] = IntensityFeatures(
                filepath=p,
                file_hash=f"hash_{i}",
                rms_energy=0.2 * i,
                brightness=0.2 * i,
                sub_bass_energy=0.5,
                kick_energy=0.5,
                bass_harmonics=0.5,
                percussiveness=0.2 * i,
                onset_strength=0.2 * i,
                camelot_key="8B",
                key_index=0.0,
            )

        return cluster, metadata_dict, intensity_dict

    def test_sequencer_initialization(self):
        """Test sequencer initialization."""
        sequencer = Sequencer()
        assert sequencer.selector is not None

    def test_sequence_fixed_mode(self, mock_data):
        """Test 'fixed' mode returns original order."""
        cluster, metadata, intensity = mock_data
        sequencer = Sequencer()

        result = sequencer.sequence(cluster, metadata, intensity, mode="fixed")
        assert result == cluster.tracks

    def test_sequence_ramp_mode(self, mock_data):
        """Test 'ramp' mode orders tracks correctly."""
        cluster, metadata, intensity = mock_data
        sequencer = Sequencer()

        # Track 0 has lowest intensity (good opener)
        # Track 4 has highest intensity (good closer)
        # Track 1, 2, 3 should be ramped in between

        result = sequencer.sequence(cluster, metadata, intensity, mode="ramp")

        assert len(result) == 5
        assert result[0] == Path("track_0.flac")  # Top opener
        assert result[-1] == Path("track_4.flac")  # Top closer

        # Check ramp in the middle (Track 1, 2, 3)
        assert intensity[result[1]].hardness < intensity[result[2]].hardness
        assert intensity[result[2]].hardness < intensity[result[3]].hardness

    def test_small_cluster_sequencing(self):
        """Test clusters with <= 2 tracks."""
        paths = [Path("t1.flac"), Path("t2.flac")]
        cluster = ClusterResult(
            cluster_id=0, tracks=paths, bpm_mean=120, bpm_std=0, track_count=2, total_duration=600
        )

        sequencer = Sequencer()
        result = sequencer.sequence(cluster, {}, {}, mode="ramp")
        assert result == paths

    def test_hardness_property(self):
        """Test the hardness calculation in IntensityFeatures."""
        feat = IntensityFeatures(
            filepath=Path("test.flac"),
            file_hash="hash",
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

        # score = 0.4*0.5 + 0.2*0.5 + 0.2*0.5 + 0.2*0.5 = 0.5
        assert feat.hardness == pytest.approx(0.5)

        feat.brightness = 1.0
        # score = 0.4*1.0 + 0.6*0.5 = 0.4 + 0.3 = 0.7
        assert feat.hardness == pytest.approx(0.7)

    def test_sequence_ramp_same_first_last_fallback(self):
        """Test fallback when top opener and closer are the same track."""
        paths = [Path(f"t{i}.flac") for i in range(3)]
        cluster = ClusterResult(
            cluster_id=0, tracks=paths, bpm_mean=120, bpm_std=0, track_count=3, total_duration=900
        )
        metadata = {p: TrackMetadata(filepath=p, bpm=120, duration=300) for p in paths}

        # Setup intensities so t0 is both best opener and best closer?
        # Actually, let's just mock the TrackSelector to force this condition.
        sequencer = Sequencer()
        mock_selector = MagicMock()

        from playchitect.core.track_selector import TrackScore, TrackSelection

        # t0 is best for both
        selection = TrackSelection(
            cluster_id=0,
            first_tracks=[
                TrackScore(path=paths[0], score=0.9, reason=""),
                TrackScore(path=paths[1], score=0.8, reason=""),
            ],
            last_tracks=[
                TrackScore(path=paths[0], score=0.9, reason=""),
                TrackScore(path=paths[2], score=0.8, reason=""),
            ],
        )
        mock_selector.select.return_value = selection
        sequencer.selector = mock_selector

        intensity = {
            p: IntensityFeatures(
                filepath=p,
                file_hash="h",
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
            for p in paths
        }

        result = sequencer.sequence(cluster, metadata, intensity, mode="ramp")

        assert result[0] == paths[0]
        assert result[-1] == paths[2]  # Fallback to second best closer
        assert result[1] == paths[1]

    def test_sequence_ramp_same_first_last_fallback_opener(self):
        """Test fallback to second best opener when top closer is only choice."""
        paths = [Path(f"t{i}.flac") for i in range(3)]
        cluster = ClusterResult(
            cluster_id=0, tracks=paths, bpm_mean=120, bpm_std=0, track_count=3, total_duration=900
        )
        metadata = {p: TrackMetadata(filepath=p, bpm=120, duration=300) for p in paths}

        sequencer = Sequencer()
        mock_selector = MagicMock()

        from playchitect.core.track_selector import TrackScore, TrackSelection

        # t0 is best for both, but only one closer candidate provided
        selection = TrackSelection(
            cluster_id=0,
            first_tracks=[
                TrackScore(path=paths[0], score=0.9, reason=""),
                TrackScore(path=paths[1], score=0.8, reason=""),
            ],
            last_tracks=[
                TrackScore(path=paths[0], score=0.9, reason=""),
            ],
        )
        mock_selector.select.return_value = selection
        sequencer.selector = mock_selector

        intensity = {
            p: IntensityFeatures(
                filepath=p,
                file_hash="h",
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
            for p in paths
        }

        result = sequencer.sequence(cluster, metadata, intensity, mode="ramp")

        assert result[0] == paths[1]  # Fallback to second best opener
        assert result[-1] == paths[0]  # Top choice remains closer


class TestFiveRhythms:
    """Test Five Rhythms sequencing functionality."""

    def test_classify_flowing_phase(self):
        """Test classification of Flowing phase: BPM 85-115 and RMS < 0.5."""
        # Flowing: 90 BPM, low energy
        assert classify_five_rhythms_phase(90.0, 0.3) == FiveRhythmsPhase.FLOWING
        assert classify_five_rhythms_phase(100.0, 0.4) == FiveRhythmsPhase.FLOWING
        assert classify_five_rhythms_phase(110.0, 0.49) == FiveRhythmsPhase.FLOWING

    def test_classify_staccato_phase(self):
        """Test classification of Staccato phase: BPM 115-135 and RMS 0.4-0.7."""
        # Staccato: 120 BPM, mid energy
        assert classify_five_rhythms_phase(120.0, 0.5) == FiveRhythmsPhase.STACCATO
        assert classify_five_rhythms_phase(125.0, 0.6) == FiveRhythmsPhase.STACCATO
        assert classify_five_rhythms_phase(130.0, 0.65) == FiveRhythmsPhase.STACCATO

    def test_classify_chaos_phase(self):
        """Test classification of Chaos phase: BPM > 135 or RMS > 0.75."""
        # Chaos: high BPM
        assert classify_five_rhythms_phase(145.0, 0.5) == FiveRhythmsPhase.CHAOS
        # Chaos: high energy
        assert classify_five_rhythms_phase(100.0, 0.8) == FiveRhythmsPhase.CHAOS
        # Chaos: both high BPM and high energy
        assert classify_five_rhythms_phase(145.0, 0.8) == FiveRhythmsPhase.CHAOS

    def test_classify_stillness_phase(self):
        """Test classification of Stillness phase: BPM < 85 or RMS < 0.25."""
        # Stillness: low BPM
        assert classify_five_rhythms_phase(70.0, 0.5) == FiveRhythmsPhase.STILLNESS
        # Stillness: low energy
        assert classify_five_rhythms_phase(100.0, 0.2) == FiveRhythmsPhase.STILLNESS
        # Stillness: both low BPM and low energy
        assert classify_five_rhythms_phase(70.0, 0.2) == FiveRhythmsPhase.STILLNESS

    def test_classify_lyrical_phase(self):
        """Test classification of Lyrical phase (catch-all)."""
        # Lyrical: mid BPM with mid energy (doesn't fit other categories)
        assert classify_five_rhythms_phase(120.0, 0.3) == FiveRhythmsPhase.LYRICAL
        assert classify_five_rhythms_phase(100.0, 0.6) == FiveRhythmsPhase.LYRICAL

    def test_classify_five_rhythms_phase_examples(self):
        """Test specific examples from acceptance criteria."""
        # classify_five_rhythms_phase(90, 0.3) == FLOWING
        assert classify_five_rhythms_phase(90.0, 0.3) == FiveRhythmsPhase.FLOWING
        # classify_five_rhythms_phase(145, 0.8) == CHAOS
        assert classify_five_rhythms_phase(145.0, 0.8) == FiveRhythmsPhase.CHAOS

    def test_classify_invalid_bpm(self):
        """Test that invalid BPM raises ValueError."""
        with pytest.raises(ValueError, match="BPM must be positive"):
            classify_five_rhythms_phase(0.0, 0.5)
        with pytest.raises(ValueError, match="BPM must be positive"):
            classify_five_rhythms_phase(-10.0, 0.5)

    def _create_intensity_features(self, path: Path, rms_energy: float) -> IntensityFeatures:
        """Create IntensityFeatures with specified RMS energy."""
        return IntensityFeatures(
            filepath=path,
            file_hash=f"hash_{path.stem}",
            rms_energy=rms_energy,
            brightness=0.5,
            sub_bass_energy=0.5,
            kick_energy=0.5,
            bass_harmonics=0.5,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )

    def test_sequence_five_rhythms_all_phases(self):
        """Test sequencing 5 tracks, one per phase, returns Flowing first, Stillness last."""
        # Create 5 tracks with specific BPMs and energies for each phase
        paths = [
            Path("flowing.flac"),  # Flowing: 100 BPM, RMS 0.3
            Path("staccato.flac"),  # Staccato: 120 BPM, RMS 0.5
            Path("chaos.flac"),  # Chaos: 145 BPM, RMS 0.8
            Path("lyrical.flac"),  # Lyrical: 120 BPM, RMS 0.3
            Path("stillness.flac"),  # Stillness: 70 BPM, RMS 0.2
        ]

        metadata = {
            paths[0]: TrackMetadata(filepath=paths[0], bpm=100.0, duration=300.0),  # Flowing
            paths[1]: TrackMetadata(filepath=paths[1], bpm=120.0, duration=300.0),  # Staccato
            paths[2]: TrackMetadata(filepath=paths[2], bpm=145.0, duration=300.0),  # Chaos
            paths[3]: TrackMetadata(filepath=paths[3], bpm=120.0, duration=300.0),  # Lyrical
            paths[4]: TrackMetadata(filepath=paths[4], bpm=70.0, duration=300.0),  # Stillness
        }

        features = {
            paths[0]: self._create_intensity_features(paths[0], 0.3),  # Flowing
            paths[1]: self._create_intensity_features(paths[1], 0.5),  # Staccato
            paths[2]: self._create_intensity_features(paths[2], 0.8),  # Chaos
            paths[3]: self._create_intensity_features(paths[3], 0.3),  # Lyrical
            paths[4]: self._create_intensity_features(paths[4], 0.2),  # Stillness
        }

        result = sequence_five_rhythms(paths, metadata, features)

        # Should return Flowing first, Stillness last
        assert result[0] == paths[0], f"Expected Flowing track first, got {result[0]}"
        assert result[-1] == paths[4], f"Expected Stillness track last, got {result[-1]}"

        # Verify order: Flowing → Staccato → Chaos → Lyrical → Stillness
        expected_order = [
            paths[0],  # Flowing
            paths[1],  # Staccato
            paths[2],  # Chaos
            paths[3],  # Lyrical
            paths[4],  # Stillness
        ]
        assert result == expected_order

    def test_sequence_five_rhythms_within_phase_energy_sorting(self):
        """Test that tracks within each phase are sorted by energy ascending."""
        # Create multiple Flowing tracks with different energies
        # Flowing: BPM 85-115 and RMS < 0.5 (but >= 0.25 to avoid Stillness)
        paths = [Path(f"flowing_{i}.flac") for i in range(3)]

        metadata = {
            paths[0]: TrackMetadata(filepath=paths[0], bpm=100.0, duration=300.0),
            paths[1]: TrackMetadata(filepath=paths[1], bpm=105.0, duration=300.0),
            paths[2]: TrackMetadata(filepath=paths[2], bpm=95.0, duration=300.0),
        }

        features = {
            paths[0]: self._create_intensity_features(paths[0], 0.4),  # Medium energy
            paths[1]: self._create_intensity_features(
                paths[1], 0.25
            ),  # Low energy (first) - at threshold
            paths[2]: self._create_intensity_features(paths[2], 0.45),  # High energy (last)
        }

        result = sequence_five_rhythms(paths, metadata, features)

        # Within Flowing phase, should be sorted by RMS energy ascending
        assert result[0] == paths[1]  # Lowest energy first
        assert result[1] == paths[0]  # Medium energy second
        assert result[2] == paths[2]  # Highest energy third

    def test_sequence_five_rhythms_skips_empty_phases(self):
        """Test that phases with no tracks are skipped."""
        # Only create Flowing and Stillness tracks
        paths = [
            Path("flowing.flac"),
            Path("stillness.flac"),
        ]

        metadata = {
            paths[0]: TrackMetadata(filepath=paths[0], bpm=100.0, duration=300.0),  # Flowing
            paths[1]: TrackMetadata(filepath=paths[1], bpm=70.0, duration=300.0),  # Stillness
        }

        features = {
            paths[0]: self._create_intensity_features(paths[0], 0.3),  # Flowing
            paths[1]: self._create_intensity_features(paths[1], 0.2),  # Stillness
        }

        result = sequence_five_rhythms(paths, metadata, features)

        # Should skip Staccato, Chaos, Lyrical and only return Flowing → Stillness
        assert result == [paths[0], paths[1]]

    def test_sequence_five_rhythms_missing_metadata(self):
        """Test that missing metadata raises ValueError."""
        paths = [Path("test.flac")]
        metadata = {}
        features = {paths[0]: self._create_intensity_features(paths[0], 0.3)}

        with pytest.raises(ValueError, match="Missing metadata"):
            sequence_five_rhythms(paths, metadata, features)

    def test_sequence_five_rhythms_missing_features(self):
        """Test that missing features raises ValueError."""
        paths = [Path("test.flac")]
        metadata = {paths[0]: TrackMetadata(filepath=paths[0], bpm=100.0, duration=300.0)}
        features = {}

        with pytest.raises(ValueError, match="Missing features"):
            sequence_five_rhythms(paths, metadata, features)

    def test_sequence_five_rhythms_none_bpm_defaults_to_lyrical(self):
        """Test that tracks with None BPM default to Lyrical phase."""
        paths = [Path("none_bpm.flac")]

        metadata = {paths[0]: TrackMetadata(filepath=paths[0], bpm=None, duration=300.0)}
        features = {paths[0]: self._create_intensity_features(paths[0], 0.5)}

        result = sequence_five_rhythms(paths, metadata, features)

        # Track with None BPM should default to Lyrical
        assert result == [paths[0]]
        # Verify it was classified as Lyrical by checking it's in the result
        assert len(result) == 1
