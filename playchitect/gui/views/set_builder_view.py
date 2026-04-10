"""Set Builder view for interactive playlist construction.

Provides a workspace for building DJ sets with:
- Timeline lane with TrackCard widgets showing sequence, title, BPM, energy, transitions
- Left browser panel with library tracks (ColumnView)
- Drag-and-drop reordering
- Auto-fill suggestions
- Export to M3U
- Footer with set statistics
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gi

from playchitect.core.compatibility import compatibility_score, next_track_suggestions
from playchitect.core.energy_blocks import EnergyBlock, create_custom_block, suggest_blocks
from playchitect.core.export import M3UExporter
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    Gdk,
    Gio,
    GObject,
    Gtk,
    Pango,
)


class BrowserTrackItem(GObject.Object):
    """GObject backing a single row in the set-builder browser ColumnView."""

    title = GObject.Property(type=str, default="")
    artist = GObject.Property(type=str, default="")
    bpm = GObject.Property(type=str, default="")
    filepath = GObject.Property(type=str, default="")

    def __init__(self, title: str, artist: str, bpm: str, filepath: str) -> None:
        super().__init__()
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.filepath = filepath

    def __getitem__(self, index: int) -> str:
        return (self.title, self.artist, self.bpm, self.filepath, "")[index]


# Constants for styling
_MAX_SET_SIZE = 20
_SEQUENCE_WIDTH = 30
_TITLE_WIDTH = 150
_BPM_WIDTH = 50
_ENERGY_WIDTH = 60
_TRANSITION_SIZE = 12


def _format_duration(seconds: float) -> str:
    """Format duration seconds as H:MM:SS or M:SS."""
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


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


class TrackCard(Gtk.Frame):
    """Card widget representing a track in the timeline.

    Displays:
    - 2-char sequence number
    - Title (truncated)
    - BPM
    - Colored energy mini-bar
    - Transition indicator dot (green/amber/red)
    """

    __gsignals__ = {
        "removed": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self, sequence: int, metadata: TrackMetadata, features: IntensityFeatures) -> None:
        super().__init__()
        self._sequence = sequence
        self._metadata = metadata
        self._features = features
        self._transition_color = "gray"

        self.set_margin_start(2)
        self.set_margin_end(2)
        self.set_margin_top(2)
        self.set_margin_bottom(2)
        self.set_size_request(180, -1)

        # Main container
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        # Top row: sequence + title
        top_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Sequence number
        self._seq_label = Gtk.Label()
        self._seq_label.set_markup(f"<b>{sequence:02d}</b>")
        self._seq_label.set_xalign(0.5)
        self._seq_label.set_size_request(_SEQUENCE_WIDTH, -1)
        top_box.append(self._seq_label)

        # Title (truncated)
        title_text = metadata.title or metadata.filepath.name
        self._title_label = Gtk.Label()
        self._title_label.set_text(title_text)
        self._title_label.set_xalign(0.0)
        self._title_label.set_ellipsize(Pango.EllipsizeMode.END)
        self._title_label.set_max_width_chars(15)
        self._title_label.set_tooltip_text(title_text)
        top_box.append(self._title_label)

        box.append(top_box)

        # BPM row
        bpm_text = f"{metadata.bpm:.1f}" if metadata.bpm else "—"
        self._bpm_label = Gtk.Label(label=f"♪ {bpm_text} BPM")
        self._bpm_label.set_xalign(0.0)
        self._bpm_label.add_css_class("caption")
        box.append(self._bpm_label)

        # Energy mini-bar
        energy_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        energy_box.set_size_request(_ENERGY_WIDTH, 8)

        self._energy_bar = Gtk.ProgressBar()
        self._energy_bar.set_fraction(features.rms_energy)
        self._energy_bar.set_size_request(_ENERGY_WIDTH, 8)

        # Color based on energy level
        if features.rms_energy >= 0.7:
            self._energy_bar.add_css_class("error")  # High energy = red-ish
        elif features.rms_energy >= 0.4:
            self._energy_bar.add_css_class("warning")  # Medium = amber
        else:
            self._energy_bar.add_css_class("success")  # Low = green

        energy_box.append(self._energy_bar)
        box.append(energy_box)

        # Transition indicator row
        transition_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        self._transition_dot = Gtk.DrawingArea()
        self._transition_dot.set_size_request(_TRANSITION_SIZE, _TRANSITION_SIZE)
        self._transition_dot.set_draw_func(self._draw_transition_dot, None)

        transition_box.append(self._transition_dot)

        self._transition_label = Gtk.Label()
        self._transition_label.set_markup("<small>→</small>")
        self._transition_label.set_xalign(0.0)
        transition_box.append(self._transition_label)

        box.append(transition_box)

        self.set_child(box)
        self.add_css_class("card")

    def _draw_transition_dot(
        self,
        area: Gtk.DrawingArea,
        cr: object,
        width: int,
        height: int,
        _data: object,
    ) -> None:
        """Draw the transition indicator dot."""
        import math

        center_x = width / 2.0
        center_y = height / 2.0
        radius = min(width, height) / 2.0 - 1

        # Set color based on transition compatibility
        colors = {
            "green": (0.2, 0.8, 0.2),
            "amber": (1.0, 0.65, 0.0),
            "red": (0.9, 0.2, 0.2),
            "gray": (0.7, 0.7, 0.7),
        }
        r, g, b = colors.get(self._transition_color, colors["gray"])

        cr.set_source_rgba(r, g, b, 1.0)
        cr.arc(center_x, center_y, radius, 0, 2 * math.pi)
        cr.fill()

    def set_transition_color(self, color: str) -> None:
        """Set the transition indicator color (green, amber, red, gray)."""
        self._transition_color = color
        self._transition_dot.queue_draw()

    def set_sequence(self, sequence: int) -> None:
        """Update the sequence number."""
        self._sequence = sequence
        self._seq_label.set_markup(f"<b>{sequence:02d}</b>")

    @property
    def metadata(self) -> TrackMetadata:
        """Get the track metadata."""
        return self._metadata

    @property
    def features(self) -> IntensityFeatures:
        """Get the track intensity features."""
        return self._features

    @property
    def filepath(self) -> Path:
        """Get the track filepath."""
        return self._metadata.filepath


class SetBuilderView(Gtk.Box):
    """Main Set Builder workspace widget.

    Features:
    - Energy block strip at top for visualizing set structure
    - Timeline lane with TrackCard widgets (drag-and-drop reorder)
    - Left browser panel with library ColumnView
    - Auto-fill button for greedy track suggestions
    - Export Set button to M3U
    - Footer StatusBar with duration, mean BPM, mean rms_energy
    """

    __gsignals__ = {
        "track-selected": (GObject.SignalFlags.RUN_FIRST, None, (str,)),  # filepath
        "set-exported": (GObject.SignalFlags.RUN_FIRST, None, (str,)),  # export path
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

        # Timeline state
        self._timeline_tracks: list[tuple[Path, TrackMetadata, IntensityFeatures]] = []
        self._track_cards: list[TrackCard] = []

        # Energy block strip (top)
        self._block_strip = self._build_block_strip()
        self.append(self._block_strip)

        # Add Block button
        self._add_block_button = self._build_add_block_button()
        self.append(self._add_block_button)

        # Main content area (split pane: browser + timeline)
        self._content_paned = self._build_content_paned()
        self.append(self._content_paned)

        # Button row
        self._button_row = self._build_button_row()
        self.append(self._button_row)

        # Footer StatusBar
        self._footer = self._build_footer()
        self.append(self._footer)

        # Next Track suggestions expander (below main content)
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

    def _build_content_paned(self) -> Gtk.Paned:
        """Build the split-pane content area (browser + timeline)."""
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_wide_handle(True)

        # Left: Browser panel
        self._browser_panel = self._build_browser_panel()
        paned.set_start_child(self._browser_panel)

        # Right: Timeline lane
        self._timeline_scroll = self._build_timeline_lane()
        paned.set_end_child(self._timeline_scroll)
        paned.set_position(300)

        return paned

    def _build_browser_panel(self) -> Gtk.Box:
        """Build the left browser panel with library ColumnView."""
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        panel.set_size_request(250, -1)

        # Header
        header = Gtk.Label()
        header.set_markup("<b>Library</b>")
        header.set_xalign(0.0)
        header.set_margin_start(8)
        header.set_margin_top(8)
        panel.append(header)

        # Energy block filter chips
        self._block_filter_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._block_filter_box.set_margin_start(8)
        self._block_filter_box.set_margin_end(8)
        self._block_filter_box.set_margin_top(4)
        panel.append(self._block_filter_box)

        # ColumnView for library tracks
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        # Create ListStore model for browser
        self._browser_store: Gio.ListStore = Gio.ListStore(item_type=BrowserTrackItem)
        # Columns: title, artist, bpm, filepath

        self._browser_view = Gtk.ColumnView()
        self._browser_view.set_model(Gtk.SingleSelection(model=self._browser_store))
        self._browser_view.set_show_row_separators(True)

        # Title column
        title_col = Gtk.ColumnViewColumn(
            title="Title",
            factory=self._create_browser_factory(0),
        )
        title_col.set_expand(True)
        self._browser_view.append_column(title_col)

        # Artist column
        artist_col = Gtk.ColumnViewColumn(
            title="Artist",
            factory=self._create_browser_factory(1),
        )
        artist_col.set_expand(True)
        self._browser_view.append_column(artist_col)

        # BPM column
        bpm_col = Gtk.ColumnViewColumn(
            title="BPM",
            factory=self._create_browser_factory(2),
        )
        bpm_col.set_fixed_width(60)
        self._browser_view.append_column(bpm_col)

        # Double-click to add to timeline
        dbl_click = Gtk.GestureClick()
        dbl_click.set_button(1)
        dbl_click.connect("pressed", self._on_browser_double_click)
        self._browser_view.add_controller(dbl_click)

        scroll.set_child(self._browser_view)
        panel.append(scroll)

        return panel

    def _create_browser_factory(self, column_index: int) -> Gtk.SignalListItemFactory:
        """Create a list item factory for browser columns."""
        factory = Gtk.SignalListItemFactory()

        def on_setup(_factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
            label = Gtk.Label()
            label.set_xalign(0.0 if column_index < 2 else 1.0)
            label.set_ellipsize(Pango.EllipsizeMode.END)
            item.set_child(label)

        def on_bind(_factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
            row = item.get_item()
            label = item.get_child()
            if row and label:
                label.set_text(str(row[column_index]))

        factory.connect("setup", on_setup)
        factory.connect("bind", on_bind)
        return factory

    def _build_timeline_lane(self) -> Gtk.ScrolledWindow:
        """Build the timeline lane with drag-and-drop support."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
        scroll.set_vexpand(True)

        self._timeline_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._timeline_box.set_margin_start(8)
        self._timeline_box.set_margin_end(8)
        self._timeline_box.set_margin_top(8)
        self._timeline_box.set_margin_bottom(8)

        # Setup drag-and-drop target
        drop_target = Gtk.DropTarget.new(GObject.TYPE_INT, Gdk.DragAction.MOVE)
        drop_target.connect("drop", self._on_timeline_drop)
        self._timeline_box.add_controller(drop_target)

        scroll.set_child(self._timeline_box)
        return scroll

    def _build_button_row(self) -> Gtk.Box:
        """Build the button row with Auto-fill and Export Set buttons."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.set_margin_top(8)
        box.set_halign(Gtk.Align.CENTER)

        # Auto-fill button
        self._auto_fill_button = Gtk.Button(label="Auto-fill")
        self._auto_fill_button.set_tooltip_text(
            "Automatically add compatible tracks from the last track"
        )
        self._auto_fill_button.connect("clicked", self._on_auto_fill_clicked)
        box.append(self._auto_fill_button)

        # Export Set button
        self._export_button = Gtk.Button(label="Export Set")
        self._export_button.set_tooltip_text("Export the current set as an M3U playlist")
        self._export_button.connect("clicked", self._on_export_clicked)
        box.append(self._export_button)

        return box

    def _build_footer(self) -> Gtk.StatusBar:
        """Build the footer StatusBar with set statistics."""
        # Use a box for custom footer layout
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        # Total duration
        self._duration_label = Gtk.Label()
        self._duration_label.set_xalign(0.0)
        box.append(self._duration_label)

        # Separator
        sep1 = Gtk.Label(label="•")
        box.append(sep1)

        # Mean BPM
        self._mean_bpm_label = Gtk.Label()
        self._mean_bpm_label.set_xalign(0.5)
        box.append(self._mean_bpm_label)

        # Separator
        sep2 = Gtk.Label(label="•")
        box.append(sep2)

        # Mean rms_energy
        self._mean_energy_label = Gtk.Label()
        self._mean_energy_label.set_xalign(1.0)
        box.append(self._mean_energy_label)

        self._update_footer()
        return box

    def _update_footer(self) -> None:
        """Update footer labels with current set statistics."""
        total_duration = 0.0
        total_bpm = 0.0
        bpm_count = 0
        total_energy = 0.0

        for _, meta, features in self._timeline_tracks:
            total_duration += meta.duration or 0
            if meta.bpm:
                total_bpm += meta.bpm
                bpm_count += 1
            total_energy += features.rms_energy

        mean_bpm = total_bpm / bpm_count if bpm_count > 0 else 0.0
        mean_energy = total_energy / len(self._timeline_tracks) if self._timeline_tracks else 0.0

        self._duration_label.set_text(f"Duration: {_format_duration(total_duration)}")
        self._mean_bpm_label.set_text(
            f"Mean BPM: {mean_bpm:.1f}" if mean_bpm > 0 else "Mean BPM: —"
        )
        self._mean_energy_label.set_text(f"Mean Energy: {mean_energy:.2f}")

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
        self._update_block_filter_chips()

    def _update_block_filter_chips(self) -> None:
        """Update energy block filter chips in browser panel."""
        # Clear existing chips
        child = self._block_filter_box.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._block_filter_box.remove(child)
            child = next_child

        # Add "All" chip
        all_btn = Gtk.ToggleButton(label="All")
        all_btn.set_active(True)
        all_btn.connect("toggled", self._on_block_filter_toggled, "")
        self._block_filter_box.append(all_btn)

        # Add chips for each energy block
        for block in self._energy_blocks:
            btn = Gtk.ToggleButton(label=block.name)
            btn.set_group(all_btn)
            btn.connect("toggled", self._on_block_filter_toggled, block.name)
            self._block_filter_box.append(btn)

    def _on_block_filter_toggled(self, btn: Gtk.ToggleButton, block_name: str) -> None:
        """Handle energy block filter chip toggled."""
        if btn.get_active():
            self._filter_browser_by_block(block_name)

    def _filter_browser_by_block(self, block_name: str) -> None:
        """Filter browser by energy block name."""
        # Clear and repopulate browser store based on filter
        self._browser_store.remove_all()

        for path, meta in self._metadata_map.items():
            # If no filter or track matches block filter criteria
            if not block_name:
                # Show all tracks
                title = meta.title or path.name
                artist = meta.artist or "Unknown Artist"
                bpm = f"{meta.bpm:.1f}" if meta.bpm else "—"
                self._browser_store.append(BrowserTrackItem(title, artist, bpm, str(path)))
            else:
                # Filter logic: check if track belongs to selected block
                # This is simplified - in real implementation, check block.cluster_ids
                features = self._features_map.get(path)
                if features:
                    for block in self._energy_blocks:
                        if block.name == block_name:
                            # Check if track energy is in block range
                            if block.energy_min <= features.rms_energy <= block.energy_max:
                                title = meta.title or path.name
                                artist = meta.artist or "Unknown Artist"
                                bpm = f"{meta.bpm:.1f}" if meta.bpm else "—"
                                self._browser_store.append(
                                    BrowserTrackItem(title, artist, bpm, str(path))
                                )
                                break

    def _load_block_tracks(self, block: EnergyBlock) -> None:
        """Load tracks for the selected block into the browser."""
        self._browser_store.remove_all()

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
                bpm = f"{meta.bpm:.1f}" if meta.bpm else "—"
                self._browser_store.append(BrowserTrackItem(title, artist, bpm, str(path)))

    def _on_browser_double_click(
        self, _gesture: Gtk.GestureClick, n_press: int, _x: float, _y: float
    ) -> None:
        """Handle double-click on browser to add track to timeline."""
        if n_press == 2:  # Double click
            selection = self._browser_view.get_model()
            if selection:
                selected_item = selection.get_selected_item()
                if selected_item:
                    filepath = Path(selected_item[3])  # filepath is column 3
                    self._add_track_to_timeline(filepath)

    def _add_track_to_timeline(self, filepath: Path) -> None:
        """Add a track to the timeline."""
        meta = self._metadata_map.get(filepath)
        features = self._features_map.get(filepath)

        if meta and features:
            self._timeline_tracks.append((filepath, meta, features))
            self._refresh_timeline()
            self._update_footer()

    def _refresh_timeline(self) -> None:
        """Refresh the timeline display with current tracks."""
        # Clear existing cards
        child = self._timeline_box.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._timeline_box.remove(child)
            child = next_child

        self._track_cards = []

        # Create TrackCards for each track
        for i, (path, meta, features) in enumerate(self._timeline_tracks, 1):
            card = TrackCard(i, meta, features)

            # Add drag source for reordering
            drag_source = Gtk.DragSource()
            drag_source.set_actions(Gdk.DragAction.MOVE)
            drag_source.connect("prepare", self._on_drag_prepare, i - 1)
            card.add_controller(drag_source)

            # Calculate transition color based on next track
            if i < len(self._timeline_tracks):
                next_path, _, next_features = self._timeline_tracks[i]
                next_meta = self._metadata_map.get(next_path)
                if next_meta and next_meta.bpm and meta.bpm:
                    score = compatibility_score(
                        features,
                        next_features,
                        meta.bpm,
                        next_meta.bpm,
                    )
                    if score >= 0.8:
                        card.set_transition_color("green")
                    elif score >= 0.5:
                        card.set_transition_color("amber")
                    else:
                        card.set_transition_color("red")
                else:
                    card.set_transition_color("gray")
            else:
                card.set_transition_color("gray")

            self._timeline_box.append(card)
            self._track_cards.append(card)

    def _on_drag_prepare(
        self, _source: Gtk.DragSource, _x: float, _y: float, index: int
    ) -> Gdk.ContentProvider:
        """Prepare drag data (track index)."""
        return Gdk.ContentProvider.new_for_value(GObject.Value(GObject.TYPE_INT, index))

    def _on_timeline_drop(
        self, _target: Gtk.DropTarget, value: GObject.Value, _x: float, _y: float
    ) -> bool:
        """Handle track drop for reordering."""
        try:
            from_index = value.get_int()

            # Calculate drop position based on x coordinate
            # For simplicity, drop at end or calculate based on card positions
            # In a real implementation, calculate which card position is closest
            to_index = len(self._timeline_tracks) - 1

            if 0 <= from_index < len(self._timeline_tracks) and from_index != to_index:
                # Swap tracks
                track = self._timeline_tracks.pop(from_index)
                self._timeline_tracks.insert(to_index, track)
                self._refresh_timeline()
                return True
        except Exception:
            pass
        return False

    def _on_auto_fill_clicked(self, _button: Gtk.Button) -> None:
        """Handle Auto-fill button click - greedily add compatible tracks."""
        if not self._timeline_tracks:
            return

        # Get last track in timeline
        last_path, last_meta, last_features = self._timeline_tracks[-1]
        last_bpm = last_meta.bpm or 0.0

        if last_bpm == 0.0:
            return

        # Build candidates list
        candidates: list[tuple[Path, IntensityFeatures, float]] = []
        for path, meta in self._metadata_map.items():
            if path not in [t[0] for t in self._timeline_tracks]:  # Not already in set
                features = self._features_map.get(path)
                if features and meta.bpm:
                    candidates.append((path, features, meta.bpm))

        # Greedy add until _MAX_SET_SIZE or no candidates
        while len(self._timeline_tracks) < _MAX_SET_SIZE and candidates:
            suggestions = next_track_suggestions(
                last_path,
                last_features,
                last_bpm,
                candidates,
                n=1,  # Just get the best one
            )

            if not suggestions:
                break

            best_path, _score = suggestions[0]
            best_meta = self._metadata_map.get(best_path)
            best_features = self._features_map.get(best_path)

            if best_meta and best_features:
                self._timeline_tracks.append((best_path, best_meta, best_features))

                # Update for next iteration
                last_path = best_path
                last_meta = best_meta
                last_features = best_features
                last_bpm = best_meta.bpm or 0.0

                # Remove from candidates
                candidates = [(p, f, b) for p, f, b in candidates if p != best_path]
            else:
                break

        self._refresh_timeline()
        self._update_footer()

    def _on_export_clicked(self, _button: Gtk.Button) -> None:
        """Handle Export Set button click - export to M3U."""
        if not self._timeline_tracks:
            return

        # Create export directory if needed
        export_dir = Path.home() / ".local" / "share" / "playchitect" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Use M3UExporter
        exporter = M3UExporter(export_dir, playlist_prefix="Set")

        # Create a cluster-like object for export
        from playchitect.core.clustering import ClusterResult

        tracks = [path for path, _, _ in self._timeline_tracks]
        # Calculate mean BPM
        bpms = [meta.bpm for _, meta, _ in self._timeline_tracks if meta.bpm]
        mean_bpm = sum(bpms) / len(bpms) if bpms else 0.0

        cluster = ClusterResult(
            cluster_id=0,
            tracks=tracks,
            bpm_mean=mean_bpm,
            bpm_std=0.0,
            track_count=len(tracks),
            total_duration=sum(meta.duration or 0 for _, meta, _ in self._timeline_tracks),
        )

        export_path = exporter.export_cluster(
            cluster, cluster_index=0, metadata_dict=self._metadata_map
        )

        self.emit("set-exported", str(export_path))

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

        # Update filter chips
        self._update_block_filter_chips()

        # Populate browser with all tracks
        self._browser_store.remove_all()
        for path, meta in metadata_map.items():
            title = meta.title or path.name
            artist = meta.artist or "Unknown Artist"
            bpm = f"{meta.bpm:.1f}" if meta.bpm else "—"
            self._browser_store.append(BrowserTrackItem(title, artist, bpm, str(path)))

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

        # Refresh browser
        self._browser_store.remove_all()
        for path, meta in metadata_map.items():
            title = meta.title or path.name
            artist = meta.artist or "Unknown Artist"
            bpm = f"{meta.bpm:.1f}" if meta.bpm else "—"
            self._browser_store.append(BrowserTrackItem(title, artist, bpm, str(path)))

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

    def get_timeline_tracks(self) -> list[tuple[Path, TrackMetadata, IntensityFeatures]]:
        """Get the current timeline tracks.

        Returns:
            List of (filepath, metadata, features) tuples in timeline order
        """
        return self._timeline_tracks.copy()

    def get_track_cards(self) -> list[TrackCard]:
        """Get the current TrackCard widgets.

        Returns:
            List of TrackCard widgets in the timeline
        """
        return self._track_cards.copy()

    @property
    def auto_fill_button(self) -> Gtk.Button:
        """Get the Auto-fill button for testing."""
        return self._auto_fill_button

    @property
    def export_button(self) -> Gtk.Button:
        """Get the Export Set button for testing."""
        return self._export_button

    @property
    def duration_label(self) -> Gtk.Label:
        """Get the duration footer label for testing."""
        return self._duration_label

    @property
    def mean_bpm_label(self) -> Gtk.Label:
        """Get the mean BPM footer label for testing."""
        return self._mean_bpm_label

    @property
    def mean_energy_label(self) -> Gtk.Label:
        """Get the mean energy footer label for testing."""
        return self._mean_energy_label

    def add_track_to_timeline(self, filepath: Path) -> None:
        """Public method to add a track to timeline (for testing)."""
        self._add_track_to_timeline(filepath)

    def set_timeline_tracks(
        self, tracks: list[tuple[Path, TrackMetadata, IntensityFeatures]]
    ) -> None:
        """Set timeline tracks directly (for testing)."""
        self._timeline_tracks = tracks.copy()
        self._refresh_timeline()
        self._update_footer()
