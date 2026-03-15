"""Smoke tests for SetBuilderView.

gi mocks are installed by tests/gui/conftest.py before this module is collected.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.energy_blocks import EnergyBlock, create_custom_block
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

# Import after conftest.py mocks are installed
from playchitect.gui.views.set_builder_view import EnergyBlockCard, SetBuilderView  # noqa: E402


class TestEnergyBlockCard:
    """Tests for EnergyBlockCard widget."""

    def test_card_initializes(self) -> None:
        """Test that EnergyBlockCard can be initialized."""
        block = EnergyBlock(
            id="test",
            name="Test Block",
            target_duration_min=60.0,
            energy_min=0.2,
            energy_max=0.5,
            cluster_ids=[1, 2, 3],
        )
        card = EnergyBlockCard(block)
        assert card is not None
        assert card.block == block

    def test_card_selection_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test card selection state management."""
        block = create_custom_block("test-block")
        card = EnergyBlockCard(block)

        # Mock add/remove_css_class to track calls
        added_classes: list[str] = []
        removed_classes: list[str] = []

        monkeypatch.setattr(card, "add_css_class", lambda x: added_classes.append(x))
        monkeypatch.setattr(card, "remove_css_class", lambda x: removed_classes.append(x))

        # Select the card
        card.set_selected(True)
        assert "card-selected" in added_classes

        # Deselect the card
        added_classes.clear()
        card.set_selected(False)
        assert "card-selected" in removed_classes


class TestSetBuilderView:
    """Tests for SetBuilderView."""

    def test_view_initializes(self) -> None:
        """Test that SetBuilderView can be instantiated without crashing.

        This is the primary smoke test from the acceptance criteria.
        """
        view = SetBuilderView()
        assert view is not None

    def test_view_loads_library_data(self) -> None:
        """Test loading library metadata and features."""
        view = SetBuilderView()

        # Create some test data
        path1 = Path("/music/track1.mp3")
        path2 = Path("/music/track2.mp3")

        metadata = {
            path1: TrackMetadata(filepath=path1, title="Track 1", bpm=128.0),
            path2: TrackMetadata(filepath=path2, title="Track 2", bpm=130.0),
        }

        features = {
            path1: IntensityFeatures(
                filepath=path1,
                file_hash="hash1",
                rms_energy=0.5,
                brightness=0.4,
                sub_bass_energy=0.3,
                kick_energy=0.6,
                bass_harmonics=0.4,
                percussiveness=0.5,
                onset_strength=0.5,
                camelot_key="8B",
                key_index=0.0,
            ),
            path2: IntensityFeatures(
                filepath=path2,
                file_hash="hash2",
                rms_energy=0.7,
                brightness=0.6,
                sub_bass_energy=0.4,
                kick_energy=0.7,
                bass_harmonics=0.5,
                percussiveness=0.6,
                onset_strength=0.7,
                camelot_key="9B",
                key_index=1.0,
            ),
        }

        view.load_library(metadata, features)
        assert view._metadata_map == metadata
        assert view._features_map == features

    def test_view_loads_clusters(self) -> None:
        """Test loading clusters and generating energy blocks."""
        view = SetBuilderView()

        # Create test clusters
        clusters = [
            ClusterResult(
                cluster_id=i,
                tracks=[Path(f"track_{i}_{j}.mp3") for j in range(5)],
                bpm_mean=120.0 + i * 2,
                bpm_std=1.5,
                track_count=5,
                total_duration=1800.0,
            )
            for i in range(10)
        ]

        # Create corresponding features
        features: dict[Path, IntensityFeatures] = {}
        for cluster in clusters:
            for i, path in enumerate(cluster.tracks):
                # Assign increasing rms energy per cluster
                rms = 0.1 + (int(str(cluster.cluster_id)) * 0.09)  # type: ignore[arg-type]
                features[path] = IntensityFeatures(
                    filepath=path,
                    file_hash=f"hash_{path.name}",
                    rms_energy=rms,
                    brightness=0.5,
                    sub_bass_energy=0.3,
                    kick_energy=0.6,
                    bass_harmonics=0.4,
                    percussiveness=0.5,
                    onset_strength=0.5,
                    camelot_key="8B",
                    key_index=0.0,
                )

        metadata = {
            path: TrackMetadata(filepath=path, title=f"Track {i}", bpm=120.0 + i)
            for i, path in enumerate(features.keys())
        }

        view.load_clusters(clusters, metadata, features)

        # Verify that energy blocks were generated
        assert len(view._energy_blocks) >= 4
        assert len(view._energy_blocks) <= 5

        # Verify first block is selected by default
        assert view._selected_block is not None

    def test_add_block_button_creates_custom_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Add Block button creates a custom block."""
        view = SetBuilderView()

        # Track blocks_box.append calls
        appended_cards: list = []

        def original_append(x):  # noqa: ANN001, ANN202
            appended_cards.append(x)

        monkeypatch.setattr(view._blocks_box, "append", original_append)

        # Simulate button click
        view._on_add_block_clicked(MagicMock())

        # Verify a custom block was created
        assert len(view._energy_blocks) == 1
        assert view._energy_blocks[0].name == "Custom"
        assert view._energy_blocks[0].id == "custom-1"

    def test_block_selection_triggers_track_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that selecting a block loads tracks into the view."""
        view = SetBuilderView()

        # Create test cluster and block
        cluster = ClusterResult(
            cluster_id=0,
            tracks=[Path("track1.mp3"), Path("track2.mp3")],
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=2,
            total_duration=720.0,
        )

        features = {
            path: IntensityFeatures(
                filepath=path,
                file_hash=f"hash_{path.name}",
                rms_energy=0.5,
                brightness=0.4,
                sub_bass_energy=0.3,
                kick_energy=0.6,
                bass_harmonics=0.4,
                percussiveness=0.5,
                onset_strength=0.5,
                camelot_key="8B",
                key_index=0.0,
            )
            for path in cluster.tracks
        }

        metadata = {
            path: TrackMetadata(filepath=path, title=f"Track {i}", bpm=128.0)
            for i, path in enumerate(cluster.tracks)
        }

        view.load_clusters([cluster], metadata, features)

        # Mock the tracks_store to verify it's populated
        store_data: list = []

        def original_append(x):  # noqa: ANN001, ANN202
            store_data.append(x)

        def mock_clear():
            store_data.clear()

        monkeypatch.setattr(view._tracks_store, "append", original_append)
        monkeypatch.setattr(view._tracks_store, "clear", mock_clear)

        # Select a block (should trigger track loading)
        block = view._energy_blocks[0]
        view._on_block_selected(MagicMock(), block)

        # Verify tracks were loaded (2 tracks should be in store)
        # Note: actual loading happens in _load_block_tracks, which calls clear() then append()
        # Since we mocked clear to clear store_data, we should see the new data
        assert len(store_data) == 2
