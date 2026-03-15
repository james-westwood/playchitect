"""
Unit tests for rekordbox_import module.
"""

from pathlib import Path

import pytest

from playchitect.core.importers.rekordbox_import import (
    parse_rekordbox_xml,
    rekordbox_key_to_camelot,
)


class TestRekordboxKeyToCamelot:
    """Test key conversion from Rekordbox to Camelot notation."""

    def test_numeric_minor_keys(self):
        """Test numeric minor key conversion (1A-12A)."""
        assert rekordbox_key_to_camelot("6A") == "6A"
        assert rekordbox_key_to_camelot("1A") == "1A"
        assert rekordbox_key_to_camelot("12A") == "12A"

    def test_numeric_major_keys(self):
        """Test numeric major key conversion (1B-12B)."""
        assert rekordbox_key_to_camelot("4B") == "4B"
        assert rekordbox_key_to_camelot("1B") == "1B"
        assert rekordbox_key_to_camelot("12B") == "12B"

    def test_traditional_minor_keys(self):
        """Test traditional minor key notation conversion."""
        assert rekordbox_key_to_camelot("Cm") == "5A"
        assert rekordbox_key_to_camelot("G#m") == "1A"
        assert rekordbox_key_to_camelot("Dm") == "7A"
        assert rekordbox_key_to_camelot("F#m") == "11A"

    def test_traditional_major_keys(self):
        """Test traditional major key notation conversion."""
        assert rekordbox_key_to_camelot("C") == "8B"
        assert rekordbox_key_to_camelot("G") == "9B"
        assert rekordbox_key_to_camelot("F") == "7B"

    def test_case_insensitive(self):
        """Test that key conversion is case-insensitive."""
        assert rekordbox_key_to_camelot("6a") == "6A"
        assert rekordbox_key_to_camelot("4b") == "4B"
        assert rekordbox_key_to_camelot("cm") == "5A"

    def test_whitespace_handling(self):
        """Test that whitespace is stripped."""
        assert rekordbox_key_to_camelot(" 6A ") == "6A"
        assert rekordbox_key_to_camelot("  Cm  ") == "5A"

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert rekordbox_key_to_camelot("") == ""

    def test_unknown_key(self):
        """Test unknown key returns empty string."""
        assert rekordbox_key_to_camelot("XYZ") == ""
        assert rekordbox_key_to_camelot("99A") == ""


class TestParseRekordboxXml:
    """Test parsing Rekordbox XML files."""

    @pytest.fixture
    def fixture_path(self):
        """Return path to test fixtures."""
        return Path(__file__).parent.parent / "fixtures"

    def test_parse_returns_five_tracks(self, fixture_path):
        """Test that parsing mini fixture returns 5 tracks."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        assert len(tracks) == 5

    def test_parse_track_keys_exist(self, fixture_path):
        """Test that all expected keys exist in parsed tracks."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        for track in tracks:
            assert "location" in track
            assert "bpm" in track
            assert "key_rekordbox" in track
            assert "cue_points" in track

    def test_parse_location_stripped_prefix(self, fixture_path):
        """Test that file://localhost prefix is stripped from locations."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        assert tracks[0]["location"] == "/home/user/music/test1.mp3"
        assert tracks[1]["location"] == "/home/user/music/test2.wav"
        assert not tracks[0]["location"].startswith("file://")

    def test_parse_bpm_values(self, fixture_path):
        """Test that BPM values are parsed correctly as floats."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        assert tracks[0]["bpm"] == 128.5
        assert tracks[1]["bpm"] == 124.0
        assert tracks[2]["bpm"] == 135.0
        assert tracks[3]["bpm"] == 140.0
        assert tracks[4]["bpm"] == 130.0

    def test_parse_key_values(self, fixture_path):
        """Test that key values are extracted correctly."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        assert tracks[0]["key_rekordbox"] == "6A"
        assert tracks[1]["key_rekordbox"] == "4B"
        assert tracks[2]["key_rekordbox"] == "Cm"
        assert tracks[3]["key_rekordbox"] == "G#m"
        assert tracks[4]["key_rekordbox"] == "8B"

    def test_parse_cue_points(self, fixture_path):
        """Test that cue points are parsed correctly."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        # Track 1 has 3 cue points (2 hot cues, 1 memory)
        assert len(tracks[0]["cue_points"]) == 3
        assert tracks[0]["cue_points"][0]["name"] == "Intro"
        assert tracks[0]["cue_points"][0]["position_ms"] == 0.0
        assert tracks[0]["cue_points"][0]["type"] == "hot_cue"
        assert tracks[0]["cue_points"][1]["name"] == "Drop"
        assert tracks[0]["cue_points"][1]["position_ms"] == 15000.0
        assert tracks[0]["cue_points"][2]["name"] == "Break"
        assert tracks[0]["cue_points"][2]["type"] == "memory_cue"

    def test_parse_track_without_cues(self, fixture_path):
        """Test that tracks without cue points return empty list."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        # Track 2 has no cue points
        assert tracks[1]["cue_points"] == []

    def test_parse_track_with_unnamed_cue(self, fixture_path):
        """Test that unnamed cue points are parsed with empty name."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        # Track 3 has one unnamed cue point
        assert len(tracks[2]["cue_points"]) == 1
        assert tracks[2]["cue_points"][0]["name"] == ""
        assert tracks[2]["cue_points"][0]["position_ms"] == 5000.0

    def test_parse_multiple_hot_cues(self, fixture_path):
        """Test parsing track with all 8 hot cues plus memory cue."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        # Track 5 has 8 hot cues (0-7) and 1 memory cue (8)
        # After sorting by position: hot cues at 0, 30s, 60s, 90s, 120s, 150s, 180s,
        # memory cue at 200s, hot cue at 210s
        assert len(tracks[4]["cue_points"]) == 9

        # All should be hot_cue type except the memory cue
        cue_types = [cue["type"] for cue in tracks[4]["cue_points"]]
        assert cue_types.count("hot_cue") == 8
        assert cue_types.count("memory_cue") == 1

        # Find the memory cue (should be the one at position 200000.0)
        memory_cue = next(cue for cue in tracks[4]["cue_points"] if cue["position_ms"] == 200000.0)
        assert memory_cue["type"] == "memory_cue"
        assert memory_cue["name"] == "Memory"

    def test_parse_cue_points_sorted(self, fixture_path):
        """Test that cue points are sorted by position."""
        xml_path = fixture_path / "rekordbox_mini.xml"
        tracks = parse_rekordbox_xml(xml_path)

        # Track 4 has cue points that should be sorted
        positions = [cue["position_ms"] for cue in tracks[3]["cue_points"]]
        assert positions == sorted(positions)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_rekordbox_xml(Path("/nonexistent/path.xml"))

    def test_empty_collection(self, tmp_path):
        """Test parsing XML with empty collection."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <DJ_PLAYLISTS Version="1.0.0">
          <COLLECTION Entries="0">
          </COLLECTION>
        </DJ_PLAYLISTS>
        """
        xml_path = tmp_path / "empty.xml"
        xml_path.write_text(xml_content)

        tracks = parse_rekordbox_xml(xml_path)
        assert tracks == []

    def test_malformed_xml(self, tmp_path):
        """Test that ET.ParseError is raised for malformed XML."""
        xml_path = tmp_path / "malformed.xml"
        xml_path.write_text("<invalid>unclosed tag")

        import xml.etree.ElementTree as ET

        with pytest.raises(ET.ParseError):
            parse_rekordbox_xml(xml_path)
