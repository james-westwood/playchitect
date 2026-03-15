"""Set Builder view for interactive playlist construction.

Provides a workspace for building DJ sets with energy block visualization,
timeline management, and next-track suggestions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gi

from playchitect.core.compatibility import compatibility_score
from playchitect.core.energy_blocks import EnergyBlock, create_custom_block, suggest_blocks
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    GObject,
    Gtk,
    Pango,
)


class EnergyBlockCard(Gtk.Frame):
    """Card widget representing an energy block.

    Displays block name, duration badge, and energy range badge.
    """

    __gsignals__ = {
        "selected": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self, energy_block: EnergyBlock) -> None:
        super().__init__()
        self._block = energy_block

        self.set_margin_start(4)
        self.set_margin_end(4)
        self.set_margin_top(4)
        self.set_margin_bottom(4)

        # Main container
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        # Block name label
        self._name_label = Gtk.Label()
        self._name_label.set_markup(f"<b>{energy_block.name}</b>")
        self._name_label.set_xalign(0.0)
        box.append(self._name_label)

        # Badges row
        badges_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        # Duration badge
        duration_text = f"{energy_block.target_duration_min:.0f} min"
        self._duration_badge = Gtk.Label(label=duration_text)
        self._duration_badge.add_css_class("badge")
        self._duration_badge.set_tooltip_text("Target duration")
        badges_box.append(self._duration_badge)

        # Energy range badge
        energy_text = f"{energy_block.energy_min:.0%}-{energy_block.energy_max:.0%}"
        self._energy_badge = Gtk.Label(label=energy_text)
        self._energy_badge.add_css_class("badge")
        self._energy_badge.set_tooltip_text("Energy range")
        badges_box.append(self._energy_badge)

        box.append(badges_box)
        self.set_child(box)

        # Make clickable
        self.add_css_class("card")
        click_controller = Gtk.GestureClick()
        click_controller.connect("pressed", self._on_clicked)
        self.add_controller(click_controller)

        self._is_selected = False

    def _on_clicked(
        self, _controller: Gtk.GestureClick, _n_press: int, _x: float, _y: float
    ) -> None:
        """Handle click to select this block."""
        self.emit("selected")

    def set_selected(self, selected: bool) -> None:
        """Update visual selection state."""
        self._is_selected = selected
        if selected:
            self.add_css_class("card-selected")
        else:
            self.remove_css_class("card-selected")

    @property
    def block(self) -> EnergyBlock:
        """Get the underlying energy block."""
        return self._block


class SetBuilderView(Gtk.Box):
    """Main Set Builder workspace widget.

    Features:
    - Energy block strip at top for visualizing set structure
    - ColumnView for tracks in the selected block
    - Add Block button for creating custom blocks
    - Next Track suggestions expander with compatibility scoring
    """

    __gsignals__ = {
        "track-selected": (GObject.SignalFlags.RUN_FIRST, None, (str,)),  # filepath
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)

        # State
        self._metadata_map: dict[Path, TrackMetadata] = {}
        self._features_map: dict[Path, IntensityFeatures] = {}
        self._clusters: list[ClusterResult] = []
        self._energy_blocks: list[EnergyBlock] = []
        self._selected_block: EnergyBlock | None = None
        self._selected_track_path: Path | None = None
        self._custom_block_counter = 0

        # Energy block strip (top)
        self._block_strip = self._build_block_strip()
        self.append(self._block_strip)

        # Add Block button
        self._add_block_button = self._build_add_block_button()
        self.append(self._add_block_button)

        # Tracks view (ColumnView)
        self._tracks_view = self._build_tracks_view()
        self.append(self._tracks_view)

        # Next Track suggestions expander
        self._next_track_expander = self._build_next_track_expander()
        self.append(self._next_track_expander)

    def _build_block_strip(self) -> Gtk.ScrolledWindow:
        """Build the energy block strip at the top."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
        scroll.set_propagate_natural_width(True)

        self._blocks_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._blocks_box.set_margin_start(8)
        self._blocks_box.set_margin_end(8)
        self._blocks_box.set_margin_top(8)
        self._blocks_box.set_margin_bottom(8)

        scroll.set_child(self._blocks_box)
        return scroll

    def _build_add_block_button(self) -> Gtk.Button:
        """Build the Add Block button."""
        button = Gtk.Button(label="+ Add Block")
        button.set_margin_start(8)
        button.set_margin_end(8)
        button.set_tooltip_text("Add a custom energy block")
        button.connect("clicked", self._on_add_block_clicked)
        return button

    def _build_tracks_view(self) -> Gtk.ScrolledWindow:
        """Build the ColumnView for tracks in the selected block."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        # Create ListStore model
        self._tracks_store = Gtk.ListStore(str, str, str, str)  # title, artist, bpm, filepath

        # ColumnView setup
        self._column_view = Gtk.ColumnView()
        self._column_view.set_model(Gtk.SingleSelection(model=self._tracks_store))

        # Title column
        title_col = Gtk.ColumnViewColumn(
            title="Title",
            factory=self._create_factory("Title", 0),
        )
        title_col.set_expand(True)
        self._column_view.append_column(title_col)

        # Artist column
        artist_col = Gtk.ColumnViewColumn(
            title="Artist",
            factory=self._create_factory("Artist", 1),
        )
        artist_col.set_expand(True)
        self._column_view.append_column(artist_col)

        # BPM column
        bpm_col = Gtk.ColumnViewColumn(
            title="BPM",
            factory=self._create_factory("BPM", 2),
        )
        bpm_col.set_fixed_width(80)
        self._column_view.append_column(bpm_col)

        scroll.set_child(self._column_view)
        return scroll

    def _create_factory(self, title: str, column_index: int) -> Gtk.SignalListItemFactory:
        """Create a list item factory for a column."""
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_factory_setup, column_index)
        return factory

    def _on_factory_setup(
        self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem, column_index: int
    ) -> None:
        """Setup a list item with a label."""
        label = Gtk.Label()
        label.set_xalign(0.0 if column_index < 2 else 1.0)
        label.set_ellipsize(Pango.EllipsizeMode.END)
        item.set_child(label)

        # Bind data
        row = item.get_item()
        if row:
            label.set_text(str(row[column_index]))

    def _build_next_track_expander(self) -> Gtk.Expander:
        """Build the 'Next Track' expander with candidate suggestions."""
        expander = Gtk.Expander(label="Next Track Suggestions")
        expander.set_expanded(True)
        expander.set_margin_top(8)

        # Container for candidate rows
        self._candidates_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._candidates_box.set_margin_top(8)
        self._candidates_box.set_margin_start(8)
        self._candidates_box.set_margin_end(8)
        self._candidates_box.set_margin_bottom(8)

        # Default message when no track selected
        self._no_selection_label = Gtk.Label(label="Select a track to see suggestions")
        self._no_selection_label.set_opacity(0.5)
        self._candidates_box.append(self._no_selection_label)

        expander.set_child(self._candidates_box)
        return expander

    def _on_add_block_clicked(self, _button: Gtk.Button) -> None:
        """Handle Add Block button click."""
        self._custom_block_counter += 1
        custom_block = create_custom_block(
            block_id=f"custom-{self._custom_block_counter}",
            target_duration_min=60.0,
        )
        self._energy_blocks.append(custom_block)
        self._add_block_card(custom_block)

    def _add_block_card(self, block: EnergyBlock) -> None:
        """Add an EnergyBlockCard to the strip."""
        card = EnergyBlockCard(block)
        card.connect("selected", self._on_block_selected, block)
        self._blocks_box.append(card)

    def _on_block_selected(self, _card: EnergyBlockCard, block: EnergyBlock) -> None:
        """Handle energy block selection."""
        # Update selection state
        child = self._blocks_box.get_first_child()
        while child:
            if isinstance(child, EnergyBlockCard):
                child.set_selected(child.block.id == block.id)
            child = child.get_next_sibling()

        self._selected_block = block
        self._load_block_tracks(block)

    def _load_block_tracks(self, block: EnergyBlock) -> None:
        """Load tracks for the selected block into the ColumnView."""
        self._tracks_store.clear()

        # Get tracks from clusters assigned to this block
        track_paths: list[Path] = []
        for cluster_id in block.cluster_ids:
            for cluster in self._clusters:
                if cluster.cluster_id == cluster_id:
                    track_paths.extend(cluster.tracks)
                    break

        # Populate store
        for path in track_paths:
            meta = self._metadata_map.get(path)
            if meta:
                title = meta.title or path.name
                artist = meta.artist or "Unknown Artist"
                bpm = f"{meta.bpm:.1f}" if meta.bpm else "-"
                self._tracks_store.append([title, artist, bpm, str(path)])

    def load_clusters(
        self,
        clusters: list[ClusterResult],
        metadata_map: dict[Path, TrackMetadata],
        features_map: dict[Path, IntensityFeatures],
    ) -> None:
        """Load clusters and generate energy blocks.

        Args:
            clusters: List of ClusterResult objects
            metadata_map: Mapping of paths to track metadata
            features_map: Mapping of paths to intensity features
        """
        self._clusters = clusters
        self._metadata_map = metadata_map
        self._features_map = features_map

        # Clear existing blocks
        child = self._blocks_box.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._blocks_box.remove(child)
            child = next_child

        # Generate suggested blocks
        self._energy_blocks = suggest_blocks(clusters, features_map)

        # Create cards for each block
        for block in self._energy_blocks:
            self._add_block_card(block)

        # Select first block if available
        if self._energy_blocks:
            self._selected_block = self._energy_blocks[0]
            self._load_block_tracks(self._energy_blocks[0])

    def _build_candidate_row(
        self,
        title: str,
        artist: str,
        score: float,
    ) -> Gtk.Box:
        """Build a single candidate row with title, artist, and score bar.

        Args:
            title: Track title
            artist: Track artist
            score: Compatibility score (0.0 to 1.0)

        Returns:
            Horizontal box containing the row widgets
        """
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.set_margin_top(4)
        row.set_margin_bottom(4)

        # Info section (title and artist)
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        info_box.set_hexpand(True)

        title_label = Gtk.Label()
        title_label.set_xalign(0.0)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.set_markup(f"<b>{title or 'Unknown Title'}</b>")
        info_box.append(title_label)

        artist_label = Gtk.Label()
        artist_label.set_xalign(0.0)
        artist_label.set_ellipsize(Pango.EllipsizeMode.END)
        artist_label.set_markup(f"<small>{artist or 'Unknown Artist'}</small>")
        info_box.append(artist_label)

        row.append(info_box)

        # Score bar
        score_box = self._build_score_bar(score)
        row.append(score_box)

        return row

    def _build_score_bar(self, score: float) -> Gtk.Box:
        """Build a visual score bar showing compatibility.

        Args:
            score: Compatibility score (0.0 to 1.0)

        Returns:
            Box containing the score bar and percentage label
        """
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        box.set_size_request(120, -1)

        # Progress bar
        bar = Gtk.ProgressBar()
        bar.set_fraction(score)
        bar.set_size_request(80, 12)

        # Color based on score
        if score >= 0.8:
            bar.add_css_class("success")  # Green
        elif score >= 0.5:
            bar.add_css_class("warning")  # Amber
        else:
            bar.add_css_class("error")  # Red

        box.append(bar)

        # Percentage label
        pct_label = Gtk.Label(label=f"{score:.0%}")
        pct_label.set_xalign(1.0)
        pct_label.set_size_request(40, -1)
        box.append(pct_label)

        return box

    def update_suggestions(
        self,
        selected_path: Path,
        candidates: list[tuple[Path, TrackMetadata, IntensityFeatures]],
    ) -> None:
        """Update the next track suggestions for the selected track.

        Args:
            selected_path: Path of the currently selected track
            candidates: List of (path, metadata, features) for candidate tracks
        """
        self._selected_track_path = selected_path

        # Clear existing candidates
        while self._candidates_box.get_first_child():
            self._candidates_box.remove(self._candidates_box.get_first_child())

        current_meta = self._metadata_map.get(selected_path)
        current_features = self._features_map.get(selected_path)

        if current_meta is None or current_features is None or not candidates:
            # Show no selection message
            self._no_selection_label = Gtk.Label(label="No suggestions available")
            self._no_selection_label.set_opacity(0.5)
            self._candidates_box.append(self._no_selection_label)
            return

        current_bpm = current_meta.bpm or 0.0
        if current_bpm == 0.0:
            self._no_selection_label = Gtk.Label(label="Cannot score: current track has no BPM")
            self._no_selection_label.set_opacity(0.5)
            self._candidates_box.append(self._no_selection_label)
            return

        # Score and rank candidates
        scored: list[tuple[Path, TrackMetadata, IntensityFeatures, float]] = []
        for path, meta, features in candidates:
            if path == selected_path:
                continue

            candidate_bpm = meta.bpm or 0.0
            if candidate_bpm == 0.0:
                continue

            score = compatibility_score(
                current_features,
                features,
                current_bpm,
                candidate_bpm,
            )
            scored.append((path, meta, features, score))

        # Sort by score descending and take top 5
        scored.sort(key=lambda x: x[3], reverse=True)
        top_5 = scored[:5]

        if not top_5:
            self._no_selection_label = Gtk.Label(label="No compatible tracks found")
            self._no_selection_label.set_opacity(0.5)
            self._candidates_box.append(self._no_selection_label)
            return

        # Add rows for each candidate
        for _path, meta, _features, score in top_5:
            row = self._build_candidate_row(
                title=meta.title or "",
                artist=meta.artist or "",
                score=score,
            )
            self._candidates_box.append(row)

    def load_library(
        self,
        metadata_map: dict[Path, TrackMetadata],
        features_map: dict[Path, IntensityFeatures],
    ) -> None:
        """Load library data for suggestions.

        Args:
            metadata_map: Mapping of paths to track metadata
            features_map: Mapping of paths to intensity features
        """
        self._metadata_map = metadata_map
        self._features_map = features_map

    def on_track_selected(self, filepath: Path) -> None:
        """Handle track selection from timeline.

        Args:
            filepath: Path to the selected track
        """
        # Build candidates list from all loaded tracks
        candidates: list[tuple[Path, TrackMetadata, IntensityFeatures]] = []
        for path, meta in self._metadata_map.items():
            features = self._features_map.get(path)
            if features is not None:
                candidates.append((path, meta, features))

        self.update_suggestions(filepath, candidates)
        self.emit("track-selected", str(filepath))
