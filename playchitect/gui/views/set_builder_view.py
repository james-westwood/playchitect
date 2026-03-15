"""Set Builder view for interactive playlist construction.

Provides a workspace for building DJ sets with energy block visualization,
timeline management, and next-track suggestions.
"""

from __future__ import annotations

from pathlib import Path

import gi

from playchitect.core.compatibility import compatibility_score
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    GObject,
    Gtk,
    Pango,
)


class SetBuilderView(Gtk.Box):
    """Main Set Builder workspace widget.

    Features:
    - Energy block strip at top for visualizing set structure
    - Timeline for sequencing tracks
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
        self._selected_track_path: Path | None = None

        # Block strip placeholder
        self._block_strip = self._build_block_strip()
        self.append(self._block_strip)

        # Timeline area (placeholder for now)
        self._timeline = self._build_timeline()
        self.append(self._timeline)

        # Next Track suggestions expander
        self._next_track_expander = self._build_next_track_expander()
        self.append(self._next_track_expander)

    def _build_block_strip(self) -> Gtk.Box:
        """Build the energy block strip at the top."""
        strip = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        strip.set_margin_bottom(8)

        # Placeholder label - actual energy blocks implemented in TASK-21
        placeholder = Gtk.Label(label="Energy Blocks — placeholder")
        placeholder.set_opacity(0.5)
        strip.append(placeholder)

        return strip

    def _build_timeline(self) -> Gtk.ScrolledWindow:
        """Build the timeline area for track sequencing."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        # Placeholder box - actual timeline implemented in TASK-33
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_top(8)

        label = Gtk.Label(label="Timeline — select a track to see suggestions")
        label.set_opacity(0.5)
        box.append(label)

        scroll.set_child(box)
        return scroll

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
