"""Unit tests for DJ software export format plugins.

Tests Rekordbox XML and Traktor NML exporters.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Generator
from pathlib import Path

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.exporters import RekordboxXMLExporter, TraktorNMLExporter
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata


@pytest.fixture
def temp_output_dir() -> Generator[Path]:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_cluster() -> ClusterResult:
    """Create a sample cluster with 2 tracks."""
    return ClusterResult(
        cluster_id=0,
        tracks=[
            Path("/music/track1.mp3"),
            Path("/music/track2.wav"),
        ],
        bpm_mean=128.0,
        bpm_std=2.0,
        track_count=2,
        total_duration=480.0,  # 8 minutes
        feature_means=None,
        feature_importance=None,
        weight_source=None,
    )


@pytest.fixture
def sample_cluster_with_genre() -> ClusterResult:
    """Create a sample cluster with genre."""
    return ClusterResult(
        cluster_id=1,
        tracks=[
            Path("/music/techno1.mp3"),
            Path("/music/techno2.mp3"),
        ],
        bpm_mean=135.0,
        bpm_std=1.5,
        track_count=2,
        total_duration=420.0,
        genre="techno",
        feature_means=None,
        feature_importance=None,
        weight_source=None,
    )


@pytest.fixture
def metadata_dict(sample_cluster: ClusterResult) -> dict[Path, TrackMetadata]:
    """Create a sample metadata dictionary."""
    return {
        sample_cluster.tracks[0]: TrackMetadata(
            filepath=sample_cluster.tracks[0],
            bpm=128.0,
            artist="Test Artist 1",
            title="Test Track 1",
            duration=240.0,
            genre="House",
        ),
        sample_cluster.tracks[1]: TrackMetadata(
            filepath=sample_cluster.tracks[1],
            bpm=130.0,
            artist="Test Artist 2",
            title="Test Track 2",
            duration=240.0,
            genre="Techno",
        ),
    }


@pytest.fixture
def features_dict(sample_cluster: ClusterResult) -> dict[Path, IntensityFeatures]:
    """Create a sample intensity features dictionary."""
    features_list = [
        IntensityFeatures(
            filepath=sample_cluster.tracks[0],
            file_hash="hash1",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.4,
            kick_energy=0.5,
            bass_harmonics=0.3,
            percussiveness=0.7,
            onset_strength=0.6,
            camelot_key="8B",
            key_index=0.0,
        ),
        IntensityFeatures(
            filepath=sample_cluster.tracks[1],
            file_hash="hash2",
            rms_energy=0.6,
            brightness=0.7,
            sub_bass_energy=0.5,
            kick_energy=0.6,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.7,
            camelot_key="9B",
            key_index=1.0,
        ),
    ]
    return {f.filepath: f for f in features_list}


class TestRekordboxXMLExporter:
    """Tests for RekordboxXMLExporter."""

    def test_exporter_initialization(self, temp_output_dir: Path) -> None:
        """Test that exporter initializes correctly."""
        exporter = RekordboxXMLExporter(temp_output_dir)
        assert exporter.output_dir == temp_output_dir
        assert exporter.playlist_prefix == "Playlist"

    def test_exporter_initialization_custom_prefix(self, temp_output_dir: Path) -> None:
        """Test that exporter accepts custom prefix."""
        exporter = RekordboxXMLExporter(temp_output_dir, playlist_prefix="Set")
        assert exporter.playlist_prefix == "Set"

    def test_export_single_cluster(
        self,
        temp_output_dir: Path,
        sample_cluster: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
        features_dict: dict[Path, IntensityFeatures],
    ) -> None:
        """Test exporting a single cluster to Rekordbox XML."""
        exporter = RekordboxXMLExporter(temp_output_dir)
        output_path = exporter.export_cluster(sample_cluster, 0, metadata_dict, features_dict)

        assert output_path.exists()
        assert output_path.suffix == ".xml"

        # Parse XML and verify structure
        tree = ET.parse(output_path)
        root = tree.getroot()

        assert root.tag == "DJ_PLAYLISTS"
        assert root.get("Version") == "1.0.0"

        # Check COLLECTION exists
        collection = root.find("COLLECTION")
        assert collection is not None

        # Check TRACK elements
        tracks = collection.findall("TRACK")
        assert len(tracks) == 2

        # Verify first track attributes
        track1 = tracks[0]
        assert track1.get("TrackID") == "1"
        assert track1.get("Name") == "Test Track 1"
        assert track1.get("Artist") == "Test Artist 1"
        assert track1.get("TotalTime") == "240"
        assert track1.get("AverageBpm") == "128.00"
        assert track1.get("Tonality") == "8B"
        location = track1.get("Location")
        assert location is not None
        assert location.startswith("file://localhost")
        assert "/music/track1.mp3" in location

        # Verify second track
        track2 = tracks[1]
        assert track2.get("TrackID") == "2"
        assert track2.get("Tonality") == "9B"

        # Check PLAYLISTS section
        playlists = root.find("PLAYLISTS")
        assert playlists is not None

        root_node = playlists.find("NODE")
        assert root_node is not None
        assert root_node.get("Type") == "0"
        assert root_node.get("Name") == "ROOT"

        playlist_node = root_node.find("NODE")
        assert playlist_node is not None
        assert playlist_node.get("Type") == "1"

        # Check playlist tracks
        playlist_tracks = playlist_node.findall("TRACK")
        assert len(playlist_tracks) == 2
        assert playlist_tracks[0].get("Key") == "1"
        assert playlist_tracks[1].get("Key") == "2"

    def test_export_cluster_no_metadata(
        self, temp_output_dir: Path, sample_cluster: ClusterResult
    ) -> None:
        """Test exporting without metadata uses defaults."""
        exporter = RekordboxXMLExporter(temp_output_dir)
        output_path = exporter.export_cluster(sample_cluster, 0, None, None)

        assert output_path.exists()

        tree = ET.parse(output_path)
        root = tree.getroot()

        collection = root.find("COLLECTION")
        assert collection is not None
        tracks = collection.findall("TRACK")

        # Should use filepath stem as title when no metadata
        track_name = tracks[0].get("Name")
        assert track_name is not None
        assert "track1" in track_name
        assert tracks[0].get("Artist") == ""
        assert tracks[0].get("TotalTime") == "0"
        assert tracks[0].get("Tonality") == ""

    def test_export_multiple_clusters(
        self,
        temp_output_dir: Path,
        sample_cluster: ClusterResult,
        sample_cluster_with_genre: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
    ) -> None:
        """Test exporting multiple clusters."""
        # Create metadata for second cluster
        meta2 = {
            sample_cluster_with_genre.tracks[0]: TrackMetadata(
                filepath=sample_cluster_with_genre.tracks[0],
                bpm=135.0,
                artist="Techno Artist",
                title="Techno Track",
                duration=210.0,
            ),
            sample_cluster_with_genre.tracks[1]: TrackMetadata(
                filepath=sample_cluster_with_genre.tracks[1],
                bpm=136.0,
                title="Another Techno",
                duration=210.0,
            ),
        }

        exporter = RekordboxXMLExporter(temp_output_dir)
        paths = exporter.export_clusters(
            [sample_cluster, sample_cluster_with_genre],
            {**metadata_dict, **meta2},
        )

        assert len(paths) == 2
        for path in paths:
            assert path.exists()

        # Check first file name contains BPM range
        assert "128-130bpm" in paths[0].name
        # Check second file name contains genre
        assert "techno" in paths[1].name.lower()


class TestTraktorNMLExporter:
    """Tests for TraktorNMLExporter."""

    def test_exporter_initialization(self, temp_output_dir: Path) -> None:
        """Test that exporter initializes correctly."""
        exporter = TraktorNMLExporter(temp_output_dir)
        assert exporter.output_dir == temp_output_dir
        assert exporter.playlist_prefix == "Playlist"

    def test_export_single_cluster(
        self,
        temp_output_dir: Path,
        sample_cluster: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
        features_dict: dict[Path, IntensityFeatures],
    ) -> None:
        """Test exporting a single cluster to Traktor NML."""
        exporter = TraktorNMLExporter(temp_output_dir)
        output_path = exporter.export_cluster(sample_cluster, 0, metadata_dict, features_dict)

        assert output_path.exists()
        assert output_path.suffix == ".nml"

        # Parse XML and verify structure
        tree = ET.parse(output_path)
        root = tree.getroot()

        assert root.tag == "NML"
        assert root.get("Version") == "19"

        # Check COLLECTION exists with correct entry count
        collection = root.find("COLLECTION")
        assert collection is not None
        assert collection.get("ENTRIES") == "2"

        # Check ENTRY elements
        entries = collection.findall("ENTRY")
        assert len(entries) == 2

        # Verify first entry
        entry1 = entries[0]
        assert entry1.get("PRIMARYKEY") == str(sample_cluster.tracks[0].resolve())

        title1 = entry1.find("TITLE")
        assert title1 is not None
        assert title1.text == "Test Track 1"

        artist1 = entry1.find("ARTIST")
        assert artist1 is not None
        assert artist1.text == "Test Artist 1"

        location1 = entry1.find("LOCATION")
        assert location1 is not None
        assert location1.get("FILE") == "track1.mp3"
        dir_attr = location1.get("DIR")
        assert dir_attr is not None
        assert "/music/" in dir_attr

        tempo1 = entry1.find("TEMPO")
        assert tempo1 is not None
        assert tempo1.get("BPM") == "128.00"

        key1 = entry1.find("KEY")
        assert key1 is not None
        assert key1.text == "8B"

        # Verify second entry has key
        entry2 = entries[1]
        key2 = entry2.find("KEY")
        assert key2 is not None
        assert key2.text == "9B"

        # Check PLAYLISTS section
        playlists = root.find("PLAYLISTS")
        assert playlists is not None

        root_node = playlists.find("NODE")
        assert root_node is not None
        assert root_node.get("TYPE") == "FOLDER"
        assert root_node.get("NAME") == "$ROOT"

        subnodes = root_node.find("SUBNODES")
        assert subnodes is not None
        assert subnodes.get("COUNT") == "1"

        playlist_node = subnodes.find("NODE")
        assert playlist_node is not None
        assert playlist_node.get("TYPE") == "PLAYLIST"

        playlist = playlist_node.find("PLAYLIST")
        assert playlist is not None
        assert playlist.get("ENTRIES") == "2"
        assert playlist.get("TYPE") == "LIST"

        # Check playlist entries
        playlist_entries = playlist.findall("ENTRY")
        assert len(playlist_entries) == 2

        # Verify primary key in playlist entry
        pk = playlist_entries[0].find("PRIMARYKEY")
        assert pk is not None
        assert pk.get("TYPE") == "TRACK"
        assert pk.get("KEY") == str(sample_cluster.tracks[0].resolve())

    def test_export_cluster_no_metadata(
        self, temp_output_dir: Path, sample_cluster: ClusterResult
    ) -> None:
        """Test exporting without metadata."""
        exporter = TraktorNMLExporter(temp_output_dir)
        output_path = exporter.export_cluster(sample_cluster, 0, None, None)

        assert output_path.exists()

        tree = ET.parse(output_path)
        root = tree.getroot()

        collection = root.find("COLLECTION")
        assert collection is not None
        entries = collection.findall("ENTRY")

        # Entry should still exist with PRIMARYKEY
        assert len(entries) == 2
        assert entries[0].get("PRIMARYKEY") == str(sample_cluster.tracks[0].resolve())

        # No title/artist elements when no metadata
        assert entries[0].find("TITLE") is None
        assert entries[0].find("ARTIST") is None

    def test_export_multiple_clusters(
        self,
        temp_output_dir: Path,
        sample_cluster: ClusterResult,
        sample_cluster_with_genre: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
    ) -> None:
        """Test exporting multiple clusters."""
        # Create metadata for second cluster
        meta2 = {
            sample_cluster_with_genre.tracks[0]: TrackMetadata(
                filepath=sample_cluster_with_genre.tracks[0],
                bpm=135.0,
                artist="Techno Artist",
                title="Techno Track",
                duration=210.0,
            ),
            sample_cluster_with_genre.tracks[1]: TrackMetadata(
                filepath=sample_cluster_with_genre.tracks[1],
                bpm=136.0,
                title="Another Techno",
                duration=210.0,
            ),
        }

        exporter = TraktorNMLExporter(temp_output_dir)
        paths = exporter.export_clusters(
            [sample_cluster, sample_cluster_with_genre],
            {**metadata_dict, **meta2},
        )

        assert len(paths) == 2
        for path in paths:
            assert path.exists()

        # Check file names
        assert ".nml" in paths[0].name
        assert ".nml" in paths[1].name


class TestExportViewIntegration:
    """Tests verifying ExportView has the new exporters enabled."""

    def test_exporters_module_available(self) -> None:
        """Test that the exporters module is properly exposed."""
        # This test verifies the exports are available from the package
        from playchitect.core.exporters import (
            RekordboxXMLExporter,
            TraktorNMLExporter,
        )

        assert RekordboxXMLExporter is not None
        assert TraktorNMLExporter is not None

    def test_format_constants_exist(self) -> None:
        """Test that format constants exist in export_view module."""
        # Import the constants directly without triggering gi import
        import sys
        from unittest.mock import MagicMock

        # Mock gi module to avoid import error
        sys.modules["gi"] = MagicMock()
        sys.modules["gi.repository"] = MagicMock()

        from playchitect.gui.views.export_view import FORMAT_REKORDBOX, FORMAT_TRAKTOR

        assert FORMAT_REKORDBOX == "rekordbox"
        assert FORMAT_TRAKTOR == "traktor"
