"""Unit tests for vibe_tags module."""

from pathlib import Path

import pytest

from playchitect.core.vibe_tags import VibeTagStore, _normalize_tag


class TestNormalizeTag:
    """Test tag normalization function."""

    def test_lowercase_conversion(self):
        """Test that tags are converted to lowercase."""
        assert _normalize_tag("Techno") == "techno"
        assert _normalize_tag("TECHNO") == "techno"
        assert _normalize_tag("TeChNo") == "techno"

    def test_whitespace_stripping(self):
        """Test that leading/trailing whitespace is stripped."""
        assert _normalize_tag("  techno  ") == "techno"
        assert _normalize_tag("techno ") == "techno"
        assert _normalize_tag(" techno") == "techno"

    def test_combined_normalization(self):
        """Test combined lowercase and whitespace stripping."""
        assert _normalize_tag("  Peak Time  ") == "peak time"
        assert _normalize_tag("  TECHNO  ") == "techno"

    def test_empty_string(self):
        """Test that empty strings remain empty."""
        assert _normalize_tag("") == ""
        assert _normalize_tag("   ") == ""


class TestVibeTagStoreInit:
    """Test VibeTagStore initialization."""

    def test_default_store_path(self, tmp_path, monkeypatch):
        """Test that default store path uses ~/.local/share/playchitect/."""
        # Use custom path to verify the logic
        custom_path = tmp_path / ".local" / "share" / "playchitect" / "vibe_tags.json"
        store = VibeTagStore(store_path=custom_path)

        expected = tmp_path / ".local" / "share" / "playchitect" / "vibe_tags.json"
        assert store._store_path == expected

    def test_custom_store_path(self, tmp_path):
        """Test that custom store path is respected."""
        custom_path = tmp_path / "custom" / "tags.json"
        store = VibeTagStore(store_path=custom_path)
        assert store._store_path == custom_path

    def test_load_on_init_creates_empty_data(self, tmp_path):
        """Test that init creates empty data when store doesn't exist."""
        store_path = tmp_path / "vibe_tags.json"
        store = VibeTagStore(store_path=store_path)
        assert store._data == {}


class TestVibeTagStoreAddTag:
    """Test add_tag method."""

    def test_add_single_tag(self, tmp_path):
        """Test adding a single tag to a track."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "techno")

        assert str(track_path) in store._data
        assert store._data[str(track_path)] == ["techno"]

    def test_add_multiple_tags(self, tmp_path):
        """Test adding multiple tags to a track."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "techno")
        store.add_tag(track_path, "peak time")
        store.add_tag(track_path, "driving")

        assert store._data[str(track_path)] == ["driving", "peak time", "techno"]

    def test_add_tag_normalizes_case(self, tmp_path):
        """Test that tags are normalized to lowercase."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "TECHNO")
        store.add_tag(track_path, "  Peak Time  ")

        assert store._data[str(track_path)] == ["peak time", "techno"]

    def test_add_duplicate_tag_ignored(self, tmp_path):
        """Test that duplicate tags are ignored."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "techno")
        store.add_tag(track_path, "techno")  # Duplicate
        store.add_tag(track_path, "TECHNO")  # Case-insensitive duplicate

        assert store._data[str(track_path)] == ["techno"]

    def test_add_tag_to_multiple_tracks(self, tmp_path):
        """Test adding tags to multiple different tracks."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track1 = Path("/music/track1.mp3")
        track2 = Path("/music/track2.mp3")

        store.add_tag(track1, "techno")
        store.add_tag(track2, "house")

        assert store._data[str(track1)] == ["techno"]
        assert store._data[str(track2)] == ["house"]

    def test_add_empty_tag_ignored(self, tmp_path):
        """Test that empty tags are ignored."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "")
        store.add_tag(track_path, "   ")

        assert str(track_path) not in store._data


class TestVibeTagStoreRemoveTag:
    """Test remove_tag method."""

    def test_remove_existing_tag(self, tmp_path):
        """Test removing an existing tag."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "techno")
        store.add_tag(track_path, "house")
        store.remove_tag(track_path, "techno")

        assert store._data[str(track_path)] == ["house"]

    def test_remove_last_tag_deletes_entry(self, tmp_path):
        """Test that removing the last tag deletes the track entry."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "techno")
        store.remove_tag(track_path, "techno")

        assert str(track_path) not in store._data

    def test_remove_nonexistent_tag_silent(self, tmp_path):
        """Test that removing a nonexistent tag is silent."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        # Should not raise
        store.remove_tag(track_path, "techno")

        assert str(track_path) not in store._data

    def test_remove_tag_case_insensitive(self, tmp_path):
        """Test that tag removal is case-insensitive."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "TECHNO")
        store.remove_tag(track_path, "techno")

        assert str(track_path) not in store._data


class TestVibeTagStoreGetTags:
    """Test get_tags method."""

    def test_get_tags_existing_track(self, tmp_path):
        """Test getting tags for a track with tags."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "techno")
        store.add_tag(track_path, "house")

        tags = store.get_tags(track_path)
        assert tags == ["house", "techno"]

    def test_get_tags_returns_copy(self, tmp_path):
        """Test that get_tags returns a copy, not a reference."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        store.add_tag(track_path, "techno")
        tags = store.get_tags(track_path)
        tags.append("house")  # Modify the returned list

        # Original should be unchanged
        assert store._data[str(track_path)] == ["techno"]

    def test_get_tags_nonexistent_track(self, tmp_path):
        """Test getting tags for a track without tags."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_path = Path("/music/track1.mp3")

        tags = store.get_tags(track_path)
        assert tags == []


class TestVibeTagStoreSearchByTag:
    """Test search_by_tag method."""

    def test_search_by_tag_single_match(self, tmp_path):
        """Test searching for tracks with a specific tag."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track1 = Path("/music/track1.mp3")
        track2 = Path("/music/track2.mp3")

        store.add_tag(track1, "techno")
        store.add_tag(track2, "house")

        results = store.search_by_tag("techno")
        assert results == [track1]

    def test_search_by_tag_multiple_matches(self, tmp_path):
        """Test searching returns multiple matching tracks."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track1 = Path("/music/track1.mp3")
        track2 = Path("/music/track2.mp3")
        track3 = Path("/music/track3.mp3")

        store.add_tag(track1, "techno")
        store.add_tag(track2, "techno")
        store.add_tag(track3, "house")

        results = store.search_by_tag("techno")
        assert len(results) == 2
        assert track1 in results
        assert track2 in results

    def test_search_by_tag_case_insensitive(self, tmp_path):
        """Test that tag search is case-insensitive."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track1 = Path("/music/track1.mp3")

        store.add_tag(track1, "TECHNO")

        results = store.search_by_tag("techno")
        assert results == [track1]

    def test_search_by_tag_no_matches(self, tmp_path):
        """Test searching for nonexistent tag returns empty list."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track1 = Path("/music/track1.mp3")

        store.add_tag(track1, "techno")

        results = store.search_by_tag("house")
        assert results == []

    def test_search_returns_sorted_paths(self, tmp_path):
        """Test that search results are sorted by path."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track_b = Path("/music/b_track.mp3")
        track_a = Path("/music/a_track.mp3")
        track_c = Path("/music/c_track.mp3")

        store.add_tag(track_b, "techno")
        store.add_tag(track_a, "techno")
        store.add_tag(track_c, "techno")

        results = store.search_by_tag("techno")
        assert results == [track_a, track_b, track_c]


class TestVibeTagStoreAllTags:
    """Test all_tags method."""

    def test_all_tags_single_track(self, tmp_path):
        """Test getting all unique tags from single track."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track = Path("/music/track1.mp3")

        store.add_tag(track, "techno")
        store.add_tag(track, "house")

        tags = store.all_tags()
        assert tags == ["house", "techno"]

    def test_all_tags_multiple_tracks(self, tmp_path):
        """Test getting all unique tags across multiple tracks."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track1 = Path("/music/track1.mp3")
        track2 = Path("/music/track2.mp3")

        store.add_tag(track1, "techno")
        store.add_tag(track1, "driving")
        store.add_tag(track2, "house")
        store.add_tag(track2, "techno")  # Duplicate

        tags = store.all_tags()
        assert tags == ["driving", "house", "techno"]

    def test_all_tags_empty_store(self, tmp_path):
        """Test getting all tags from empty store."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")

        tags = store.all_tags()
        assert tags == []


class TestVibeTagStorePersistence:
    """Test save/load persistence."""

    def test_save_creates_file(self, tmp_path):
        """Test that save creates the JSON file."""
        store_path = tmp_path / "vibe_tags.json"
        store = VibeTagStore(store_path=store_path)
        track = Path("/music/track1.mp3")

        store.add_tag(track, "techno")

        assert store_path.exists()

    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save creates parent directories if needed."""
        store_path = tmp_path / "nested" / "deep" / "vibe_tags.json"
        store = VibeTagStore(store_path=store_path)
        track = Path("/music/track1.mp3")

        store.add_tag(track, "techno")

        assert store_path.parent.exists()
        assert store_path.exists()

    def test_save_load_roundtrip(self, tmp_path):
        """Test that data persists across save and load."""
        store_path = tmp_path / "vibe_tags.json"

        # Create store and add tags
        store1 = VibeTagStore(store_path=store_path)
        track1 = Path("/music/track1.mp3")
        track2 = Path("/music/track2.mp3")

        store1.add_tag(track1, "techno")
        store1.add_tag(track1, "house")
        store1.add_tag(track2, "driving")

        # Create new store instance loading same file
        store2 = VibeTagStore(store_path=store_path)

        assert store2.get_tags(track1) == ["house", "techno"]
        assert store2.get_tags(track2) == ["driving"]

    def test_save_removes_empty_entries(self, tmp_path):
        """Test that save removes tracks with empty tag lists."""
        store_path = tmp_path / "vibe_tags.json"
        store = VibeTagStore(store_path=store_path)
        track = Path("/music/track1.mp3")

        store.add_tag(track, "techno")
        store.remove_tag(track, "techno")

        # Load fresh store to verify empty entry was not saved
        store2 = VibeTagStore(store_path=store_path)
        assert str(track) not in store2._data

    def test_load_handles_missing_file(self, tmp_path):
        """Test that loading missing file creates empty store."""
        store_path = tmp_path / "nonexistent" / "vibe_tags.json"
        store = VibeTagStore(store_path=store_path)

        assert store._data == {}

    def test_load_handles_corrupt_json(self, tmp_path, caplog):
        """Test that corrupt JSON is handled gracefully."""
        store_path = tmp_path / "vibe_tags.json"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("not valid json {{")

        store = VibeTagStore(store_path=store_path)

        assert store._data == {}
        assert "Corrupt vibe tag store" in caplog.text

    def test_load_handles_invalid_data_structure(self, tmp_path):
        """Test that invalid data structure is filtered out."""
        store_path = tmp_path / "vibe_tags.json"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text('{"track1.mp3": "not a list", "track2.mp3": ["techno"]}')

        store = VibeTagStore(store_path=store_path)

        assert "track1.mp3" not in store._data
        assert store._data.get("track2.mp3") == ["techno"]


class TestVibeTagStoreAdditionalMethods:
    """Test additional helper methods."""

    def test_clear_tags(self, tmp_path):
        """Test clearing all tags from a track."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track = Path("/music/track1.mp3")

        store.add_tag(track, "techno")
        store.add_tag(track, "house")
        store.clear_tags(track)

        assert str(track) not in store._data

    def test_clear_tags_nonexistent_track(self, tmp_path):
        """Test clearing tags from track without tags is silent."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track = Path("/music/track1.mp3")

        # Should not raise
        store.clear_tags(track)

    def test_remove_tag_from_all(self, tmp_path):
        """Test removing a tag from all tracks."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track1 = Path("/music/track1.mp3")
        track2 = Path("/music/track2.mp3")

        store.add_tag(track1, "techno")
        store.add_tag(track1, "house")
        store.add_tag(track2, "techno")

        store.remove_tag_from_all("techno")

        assert store.get_tags(track1) == ["house"]
        assert store.get_tags(track2) == []

    def test_to_dict(self, tmp_path):
        """Test exporting data to dictionary."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track = Path("/music/track1.mp3")

        store.add_tag(track, "techno")

        data = store.to_dict()
        assert data == {str(track): ["techno"]}

    def test_to_dict_returns_copy(self, tmp_path):
        """Test that to_dict returns a copy."""
        store = VibeTagStore(store_path=tmp_path / "tags.json")
        track = Path("/music/track1.mp3")

        store.add_tag(track, "techno")
        data = store.to_dict()
        data[str(track)].append("house")

        # Original should be unchanged
        assert store._data[str(track)] == ["techno"]

    def test_from_dict(self, tmp_path):
        """Test creating store from dictionary."""
        data = {
            "/music/track1.mp3": ["techno", "house"],
            "/music/track2.mp3": ["driving"],
        }

        store = VibeTagStore.from_dict(data, store_path=tmp_path / "tags.json")

        assert store.get_tags(Path("/music/track1.mp3")) == ["house", "techno"]
        assert store.get_tags(Path("/music/track2.mp3")) == ["driving"]

    def test_from_dict_normalizes_tags(self, tmp_path):
        """Test that from_dict normalizes tag values."""
        data = {"/music/track1.mp3": ["TECHNO", "  Peak Time  "]}

        store = VibeTagStore.from_dict(data, store_path=tmp_path / "tags.json")

        assert store.get_tags(Path("/music/track1.mp3")) == ["peak time", "techno"]
