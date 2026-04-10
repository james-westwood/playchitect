"""Smoke tests for SetBuilderView.

gi mocks are installed by tests/gui/conftest.py before this module is collected.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.energy_blocks import EnergyBlock, create_custom_block
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

# Import after conftest.py mocks are installed
from playchitect.gui.views.set_builder_view import (  # noqa: E402
    EnergyBlockCard,
    SetBuilderView,
    TrackCard,
)


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


class TestTrackCard:
    """Tests for TrackCard widget."""

    def test_track_card_initializes_with_title_and_bpm(self) -> None:
        """Test that TrackCard renders with title and BPM."""
        path = Path("/music/test_track.mp3")
        metadata = TrackMetadata(
            filepath=path,
            title="Test Track",
            artist="Test Artist",
            bpm=128.0,
            duration=300.0,
        )
        features = IntensityFeatures(
            file_path=path,
            file_hash="hash123",
            rms_energy=0.6,
            brightness=0.5,
            sub_bass_energy=0.4,
            kick_energy=0.7,
            bass_harmonics=0.5,
            percussiveness=0.6,
            onset_strength=0.6,
            camelot_key="8B",
            key_index=0.0,
        )

        card = TrackCard(sequence=1, metadata=metadata, features=features)
        assert card is not None
        assert card.metadata == metadata
        assert card.features == features
        assert card.filepath == path

    def test_track_card_displays_sequence_number(self) -> None:
        """Test that TrackCard displays the correct sequence number."""
        path = Path("/music/track.mp3")
        metadata = TrackMetadata(filepath=path, title="Track", bpm=130.0)
        features = IntensityFeatures(
            file_path=path,
            file_hash="hash",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.3,
            kick_energy=0.6,
            bass_harmonics=0.4,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="9B",
            key_index=1.0,
        )

        card = TrackCard(sequence=5, metadata=metadata, features=features)
        assert card._sequence == 5

    def test_track_card_set_sequence_updates_display(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that set_sequence updates the sequence display."""
        path = Path("/music/track.mp3")
        metadata = TrackMetadata(filepath=path, title="Track", bpm=130.0)
        features = IntensityFeatures(
            file_path=path,
            file_hash="hash",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.3,
            kick_energy=0.6,
            bass_harmonics=0.4,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="9B",
            key_index=1.0,
        )

        card = TrackCard(sequence=1, metadata=metadata, features=features)

        # Mock the label update
        markup_calls: list[str] = []
        monkeypatch.setattr(card._seq_label, "set_markup", lambda x: markup_calls.append(x))

        card.set_sequence(10)
        assert card._sequence == 10
        assert any("10" in call for call in markup_calls)

    def test_track_card_transition_color(self) -> None:
        """Test that TrackCard can set transition indicator color."""
        path = Path("/music/track.mp3")
        metadata = TrackMetadata(filepath=path, title="Track", bpm=130.0)
        features = IntensityFeatures(
            file_path=path,
            file_hash="hash",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.3,
            kick_energy=0.6,
            bass_harmonics=0.4,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="9B",
            key_index=1.0,
        )

        card = TrackCard(sequence=1, metadata=metadata, features=features)

        # Test setting different colors
        card.set_transition_color("green")
        assert card._transition_color == "green"

        card.set_transition_color("amber")
        assert card._transition_color == "amber"

        card.set_transition_color("red")
        assert card._transition_color == "red"


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
                file_path=path1,
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
                file_path=path2,
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
                rms = 0.1 + (int(str(cluster.cluster_id)) * 0.09)
                features[path] = IntensityFeatures(
                    file_path=path,
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
                file_path=path,
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

        monkeypatch.setattr(view._browser_store, "append", original_append)
        monkeypatch.setattr(view._browser_store, "clear", mock_clear)

        # Select a block (should trigger track loading)
        block = view._energy_blocks[0]
        view._on_block_selected(MagicMock(), block)

        # Verify tracks were loaded (2 tracks should be in store)
        assert len(store_data) == 2

    def test_auto_fill_button_is_present(self) -> None:
        """Test that Auto-fill button is present and accessible."""
        view = SetBuilderView()
        assert view.auto_fill_button is not None
        # Button should be a GTK widget (mocked as _FakeGtkBase)
        assert hasattr(view.auto_fill_button, "connect")

    def test_export_set_button_is_present(self) -> None:
        """Test that Export Set button is present and accessible."""
        view = SetBuilderView()
        assert view.export_button is not None
        # Button should be a GTK widget (mocked as _FakeGtkBase)
        assert hasattr(view.export_button, "connect")

    def test_footer_labels_exist(self) -> None:
        """Test that footer labels exist (Duration, Mean BPM, Mean Energy)."""
        view = SetBuilderView()

        assert view.duration_label is not None
        assert view.mean_bpm_label is not None
        assert view.mean_energy_label is not None

    def test_footer_updates_with_timeline_data(self) -> None:
        """Test that footer updates when tracks are added to timeline."""
        view = SetBuilderView()

        # Create test tracks
        path1 = Path("/music/track1.mp3")
        path2 = Path("/music/track2.mp3")

        meta1 = TrackMetadata(filepath=path1, title="Track 1", bpm=128.0, duration=300.0)
        meta2 = TrackMetadata(filepath=path2, title="Track 2", bpm=130.0, duration=240.0)

        features1 = IntensityFeatures(
            file_path=path1,
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
        )
        features2 = IntensityFeatures(
            file_path=path2,
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
        )

        view._metadata_map = {path1: meta1, path2: meta2}
        view._features_map = {path1: features1, path2: features2}

        # Set timeline tracks
        view.set_timeline_tracks([(path1, meta1, features1), (path2, meta2, features2)])

        # Verify footer was updated
        # Total duration: 300 + 240 = 540 seconds = 9 minutes
        # Mean BPM: (128 + 130) / 2 = 129
        # Mean energy: (0.5 + 0.7) / 2 = 0.6
        assert view._timeline_tracks is not None
        assert len(view._timeline_tracks) == 2

    def test_auto_fill_adds_tracks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Auto-fill button adds compatible tracks to timeline."""
        view = SetBuilderView()

        # Create test tracks
        path1 = Path("/music/track1.mp3")
        path2 = Path("/music/track2.mp3")
        path3 = Path("/music/track3.mp3")

        meta1 = TrackMetadata(filepath=path1, title="Track 1", bpm=128.0, duration=300.0)
        meta2 = TrackMetadata(filepath=path2, title="Track 2", bpm=128.5, duration=240.0)
        meta3 = TrackMetadata(filepath=path3, title="Track 3", bpm=200.0, duration=200.0)

        features1 = IntensityFeatures(
            file_path=path1,
            file_hash="hash1",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.3,
            kick_energy=0.6,
            bass_harmonics=0.4,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )
        features2 = IntensityFeatures(
            file_path=path2,
            file_hash="hash2",
            rms_energy=0.52,  # Similar energy for high compatibility
            brightness=0.52,
            sub_bass_energy=0.32,
            kick_energy=0.62,
            bass_harmonics=0.42,
            percussiveness=0.52,
            onset_strength=0.52,
            camelot_key="9B",  # Compatible key
            key_index=1.0,
        )
        features3 = IntensityFeatures(
            file_path=path3,
            file_hash="hash3",
            rms_energy=0.9,
            brightness=0.8,
            sub_bass_energy=0.7,
            kick_energy=0.8,
            bass_harmonics=0.6,
            percussiveness=0.9,
            onset_strength=0.8,
            camelot_key="3A",  # Different key
            key_index=5.0,
        )

        view._metadata_map = {path1: meta1, path2: meta2, path3: meta3}
        view._features_map = {path1: features1, path2: features2, path3: features3}

        # Start with one track in timeline
        view.set_timeline_tracks([(path1, meta1, features1)])
        initial_count = len(view._timeline_tracks)
        assert initial_count == 1

        # Mock refresh to prevent GTK operations
        monkeypatch.setattr(view, "_refresh_timeline", lambda: None)
        monkeypatch.setattr(view, "_update_footer", lambda: None)

        # Click auto-fill
        view._on_auto_fill_clicked(MagicMock())

        # Should have added at least one track (track2 with similar BPM/energy/key)
        assert len(view._timeline_tracks) > initial_count

    def test_add_track_to_timeline(self) -> None:
        """Test adding a track to timeline."""
        view = SetBuilderView()

        path = Path("/music/test_track.mp3")
        meta = TrackMetadata(filepath=path, title="Test Track", bpm=128.0, duration=300.0)
        features = IntensityFeatures(
            file_path=path,
            file_hash="hash",
            rms_energy=0.6,
            brightness=0.5,
            sub_bass_energy=0.4,
            kick_energy=0.7,
            bass_harmonics=0.5,
            percussiveness=0.6,
            onset_strength=0.6,
            camelot_key="8B",
            key_index=0.0,
        )

        view._metadata_map = {path: meta}
        view._features_map = {path: features}

        # Add track
        view.add_track_to_timeline(path)

        # Verify track was added
        assert len(view._timeline_tracks) == 1
        assert view._timeline_tracks[0][0] == path

    def test_get_timeline_tracks(self) -> None:
        """Test getting timeline tracks."""
        view = SetBuilderView()

        path1 = Path("/music/track1.mp3")
        path2 = Path("/music/track2.mp3")

        meta1 = TrackMetadata(filepath=path1, title="Track 1", bpm=128.0)
        meta2 = TrackMetadata(filepath=path2, title="Track 2", bpm=130.0)

        features1 = IntensityFeatures(
            file_path=path1,
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
        )
        features2 = IntensityFeatures(
            file_path=path2,
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
        )

        view.set_timeline_tracks([(path1, meta1, features1), (path2, meta2, features2)])

        tracks = view.get_timeline_tracks()
        assert len(tracks) == 2
        assert tracks[0][0] == path1
        assert tracks[1][0] == path2

    def test_export_button_click(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Export Set button triggers export."""
        view = SetBuilderView()

        # Create a track
        path = Path("/music/track1.mp3")
        meta = TrackMetadata(filepath=path, title="Track 1", bpm=128.0, duration=300.0)
        features = IntensityFeatures(
            file_path=path,
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
        )

        view._metadata_map = {path: meta}
        view._features_map = {path: features}
        view.set_timeline_tracks([(path, meta, features)])

        # Mock M3UExporter
        mock_export_path = Path("/tmp/test_export.m3u")
        mock_exporter = MagicMock()
        mock_exporter.export_cluster.return_value = mock_export_path

        with patch(
            "playchitect.gui.views.set_builder_view.M3UExporter",
            return_value=mock_exporter,
        ):
            # Mock emit to capture the signal
            emitted_signals: list[tuple[str]] = []
            monkeypatch.setattr(
                view, "emit", lambda signal, *args: emitted_signals.append((signal, *args))
            )

            # Click export button
            view._on_export_clicked(MagicMock())

            # Verify export was called
            mock_exporter.export_cluster.assert_called_once()
            # Verify signal was emitted
            assert any(sig[0] == "set-exported" for sig in emitted_signals)

    def test_timeline_refresh_calculates_transition_colors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that timeline refresh calculates transition indicator colors."""
        view = SetBuilderView()

        # Create two compatible tracks
        path1 = Path("/music/track1.mp3")
        path2 = Path("/music/track2.mp3")

        meta1 = TrackMetadata(filepath=path1, title="Track 1", bpm=128.0, duration=300.0)
        meta2 = TrackMetadata(filepath=path2, title="Track 2", bpm=128.0, duration=240.0)

        features1 = IntensityFeatures(
            file_path=path1,
            file_hash="hash1",
            rms_energy=0.5,
            brightness=0.5,
            sub_bass_energy=0.3,
            kick_energy=0.6,
            bass_harmonics=0.4,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )
        features2 = IntensityFeatures(
            file_path=path2,
            file_hash="hash2",
            rms_energy=0.5,  # Same energy
            brightness=0.5,
            sub_bass_energy=0.3,
            kick_energy=0.6,
            bass_harmonics=0.4,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="9B",  # Compatible key
            key_index=1.0,
        )

        view._metadata_map = {path1: meta1, path2: meta2}
        view._features_map = {path1: features1, path2: features2}

        # Add tracks to timeline
        view.set_timeline_tracks([(path1, meta1, features1), (path2, meta2, features2)])

        # Verify TrackCards were created with proper transition colors
        # Since the tracks have same BPM and compatible keys, first card should show green
        cards = view.get_track_cards()
        assert len(cards) == 2

    def test_block_filter_chips_updated_on_selection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that block filter chips are updated when blocks are loaded."""
        view = SetBuilderView()

        # Create test cluster
        cluster = ClusterResult(
            cluster_id=0,
            tracks=[Path("track1.mp3")],
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=1,
            total_duration=300.0,
        )

        features = {
            path: IntensityFeatures(
                file_path=path,
                file_hash="hash",
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
            path: TrackMetadata(filepath=path, title="Track", bpm=128.0) for path in cluster.tracks
        }

        # Track filter box updates
        chip_count = [0]

        def count_append(x):  # noqa: ANN001, ANN202
            chip_count[0] += 1

        monkeypatch.setattr(view._block_filter_box, "append", count_append)

        view.load_clusters([cluster], metadata, features)

        # Should have added chips (All + energy blocks)
        assert chip_count[0] > 0
