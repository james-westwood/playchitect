"""
Unit tests for sequencer module.
"""

from pathlib import Path

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.sequencer import Sequencer


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
        )

        # score = 0.4*0.5 + 0.2*0.5 + 0.2*0.5 + 0.2*0.5 = 0.5
        assert feat.hardness == pytest.approx(0.5)

        feat.brightness = 1.0
        # score = 0.4*1.0 + 0.6*0.5 = 0.4 + 0.3 = 0.7
        assert feat.hardness == pytest.approx(0.7)
