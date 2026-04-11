"""Track preview panel with cover art, metadata, and GStreamer playback.

Provides a side panel displaying track details with embedded audio playback using
GStreamer. Cover art is extracted from APIC/METADATA_BLOCK_PICTURE tags and
cached to disk for fast subsequent loads. Also supports structural analysis
to inject cue points into Mixxx.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")

try:
    gi.require_version("Gst", "1.0")
    from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
        GLib,
        GObject,
        Gst,
        Gtk,
    )
except (ImportError, ValueError):
    # Gst not available - create mock objects for tests
    Gst = None  # type: ignore[misc]
    from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
        GLib,
        GObject,
        Gtk,
    )

# Import GdkPixbuf with proper error handling
try:
    from gi.repository import GdkPixbuf  # type: ignore[unresolved-import]  # noqa: E402
except ImportError:
    GdkPixbuf = None  # type: ignore[misc]

from playchitect.core.mixxx_sync import MixxxSync  # noqa: E402
from playchitect.core.structural_analyzer import (  # noqa: E402
    StructuralAnalyzer,
    predict_cue_points,
)
from playchitect.core.vibe_tags import VibeTagStore  # noqa: E402
from playchitect.gui.views.library_view import LibraryTrackModel  # noqa: E402

logger = logging.getLogger(__name__)

# Constants
COVER_ART_SIZE = 240
CACHE_DIR = Path.home() / ".cache" / "playchitect" / "covers"
SEEK_UPDATE_INTERVAL_MS = 100
VOLUME_DEFAULT = 0.8


def _ensure_cache_dir() -> None:
    """Ensure the cover art cache directory exists."""
    if CACHE_DIR.exists() and not CACHE_DIR.is_dir():
        CACHE_DIR.unlink()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(filepath: str) -> Path:
    """Get the cache path for a file's cover art."""
    file_hash = hashlib.md5(filepath.encode()).hexdigest()  # noqa: S324
    return CACHE_DIR / f"{file_hash}.jpg"


def _extract_cover_art(filepath: str) -> bytes | None:
    """Extract embedded cover art from audio file using mutagen.

    Supports APIC (MP3/ID3) and METADATA_BLOCK_PICTURE (FLAC/Vorbis).

    Args:
        filepath: Path to the audio file.

    Returns:
        Raw image bytes or None if no cover art found.
    """
    try:
        from mutagen import File  # type: ignore[import-untyped]
        from mutagen.flac import FLAC  # type: ignore[import-untyped]
        from mutagen.mp3 import MP3  # type: ignore[import-untyped]

        audio = File(filepath)
        if audio is None:
            return None

        # Handle FLAC (Vorbis comments with pictures)
        if isinstance(audio, FLAC):
            if audio.pictures:
                return audio.pictures[0].data

        # Handle MP3 (ID3 APIC frames)
        if isinstance(audio, MP3):
            for tag in audio.tags.values():
                if tag.FrameID == "APIC":
                    return tag.data

        # Generic fallback for other formats
        if hasattr(audio, "tags") and audio.tags:
            for key in ["APIC:Cover", "APIC:", "cover", "COVER"]:
                if key in audio.tags:
                    tag = audio.tags[key]
                    if hasattr(tag, "data"):
                        return tag.data

        return None
    except Exception:
        logger.exception("Failed to extract cover art from %s", filepath)
        return None


class TrackPreviewPanel(Gtk.Box):
    """Track preview panel with cover art, metadata, and playback controls.

    Signals:
        prev-track: Emitted when the previous track button is clicked
        next-track: Emitted when the next track button is clicked

    The panel is hidden by default and should be toggled via set_visible().
    """

    __gsignals__ = {
        "prev-track": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "next-track": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        self.set_margin_start(12)
        self.set_margin_end(12)
        self.set_margin_top(12)
        self.set_margin_bottom(12)

        # State
        self._current_track: LibraryTrackModel | None = None
        self._playbin: Any = None
        self._is_playing = False
        self._duration_ns = 0
        self._seek_timeout_id: int | None = None
        self._tag_store = VibeTagStore()

        # Build UI sections
        self._build_cover_art_section()
        self._build_metadata_section()
        self._build_info_pills_section()
        self._build_tags_section()
        self._build_controls_section()

        # Initialize GStreamer pipeline
        self._init_gstreamer()

        # Ensure cache directory exists
        _ensure_cache_dir()

    def _build_cover_art_section(self) -> None:
        """Build the cover art display section."""
        self._cover_frame = Gtk.Frame()
        self._cover_frame.add_css_class("card")

        # Use a Box to center the content
        cover_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        cover_box.set_halign(Gtk.Align.CENTER)
        cover_box.set_valign(Gtk.Align.CENTER)

        # Cover art picture (240x240)
        self._cover_picture = Gtk.Picture()
        self._cover_picture.set_content_fit(Gtk.ContentFit.COVER)
        self._cover_picture.set_size_request(COVER_ART_SIZE, COVER_ART_SIZE)

        # Placeholder icon (shown when no cover art)
        self._cover_icon = Gtk.Image.new_from_icon_name("audio-x-generic")
        self._cover_icon.set_pixel_size(COVER_ART_SIZE)
        self._cover_icon.set_visible(False)

        cover_box.append(self._cover_picture)
        cover_box.append(self._cover_icon)

        self._cover_frame.set_child(cover_box)
        self.append(self._cover_frame)

    def _build_metadata_section(self) -> None:
        """Build the metadata labels section."""
        metadata_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        # Title (large, bold)
        self._title_label = Gtk.Label()
        self._title_label.add_css_class("title-1")
        self._title_label.set_xalign(0.0)
        self._title_label.set_ellipsize(True)
        self._title_label.set_max_width_chars(30)
        self._title_label.set_tooltip_text("Track title")
        metadata_box.append(self._title_label)

        # Artist (subtitle style)
        self._artist_label = Gtk.Label()
        self._artist_label.add_css_class("subtitle")
        self._artist_label.set_xalign(0.0)
        self._artist_label.set_ellipsize(True)
        self._artist_label.set_max_width_chars(30)
        self._artist_label.set_tooltip_text("Artist")
        metadata_box.append(self._artist_label)

        # Album + Year (caption style)
        self._album_label = Gtk.Label()
        self._album_label.add_css_class("caption")
        self._album_label.set_xalign(0.0)
        self._album_label.set_ellipsize(True)
        self._album_label.set_max_width_chars(30)
        self._album_label.set_tooltip_text("Album and year")
        metadata_box.append(self._album_label)

        self.append(metadata_box)

    def _build_info_pills_section(self) -> None:
        """Build the info pills row (BPM, Key, Duration, Format)."""
        pills_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        pills_box.set_halign(Gtk.Align.START)

        # Create pill labels
        self._bpm_pill = self._create_pill("— BPM")
        self._key_pill = self._create_pill("—")
        self._duration_pill = self._create_pill("0:00")
        self._format_pill = self._create_pill("—")

        pills_box.append(self._bpm_pill)
        pills_box.append(self._key_pill)
        pills_box.append(self._duration_pill)
        pills_box.append(self._format_pill)

        self.append(pills_box)

    def _build_tags_section(self) -> None:
        """Build the vibe tags section with FlowBox chips and entry."""
        # Tags container
        tags_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        tags_box.set_margin_top(12)

        # Section label
        tags_label = Gtk.Label()
        tags_label.set_xalign(0.0)
        tags_label.add_css_class("caption")
        tags_label.add_css_class("dim-label")
        tags_label.set_text("Vibe Tags")
        tags_box.append(tags_label)

        # FlowBox for tag chips
        self._tags_flowbox = Gtk.FlowBox()
        self._tags_flowbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self._tags_flowbox.set_column_spacing(6)
        self._tags_flowbox.set_row_spacing(6)
        self._tags_flowbox.set_homogeneous(False)
        self._tags_flowbox.set_max_children_per_line(10)
        tags_box.append(self._tags_flowbox)

        # Entry with completion for adding tags
        entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        self._tag_entry = Gtk.Entry()
        self._tag_entry.set_placeholder_text("Add tag...")
        self._tag_entry.set_hexpand(True)
        self._tag_entry.connect("activate", self._on_tag_entry_activate)

        # Set up auto-completion
        self._setup_tag_completion()

        entry_box.append(self._tag_entry)

        # Add button
        add_btn = Gtk.Button.new_from_icon_name("list-add-symbolic")
        add_btn.set_tooltip_text("Add tag")
        add_btn.connect("clicked", self._on_add_tag_clicked)
        entry_box.append(add_btn)

        tags_box.append(entry_box)

        self.append(tags_box)

    def _setup_tag_completion(self) -> None:
        """Set up entry completion for tag suggestions."""
        completion = Gtk.EntryCompletion()
        completion.set_inline_completion(True)
        completion.set_popup_completion(True)

        # Create list store for completion (single column of strings)
        self._completion_model = Gtk.ListStore.new([str])  # type: ignore[misc]
        completion.set_model(self._completion_model)
        completion.set_text_column(0)

        self._tag_entry.set_completion(completion)

    def _update_tag_completion(self) -> None:
        """Update completion model with all available tags."""
        self._completion_model.clear()
        for tag in self._tag_store.all_tags():
            self._completion_model.append([tag])

    def _create_tag_chip(self, tag: str) -> Gtk.Button:
        """Create a tag chip button with dismiss functionality."""
        # Create a horizontal box with tag text and dismiss button
        chip_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        chip_box.set_margin_start(8)
        chip_box.set_margin_end(4)
        chip_box.set_margin_top(2)
        chip_box.set_margin_bottom(2)

        # Tag label
        label = Gtk.Label(label=tag)
        chip_box.append(label)

        # Dismiss button (×)
        dismiss_btn = Gtk.Button.new_from_icon_name("window-close-symbolic")
        dismiss_btn.add_css_class("small-button")
        dismiss_btn.set_has_frame(False)
        dismiss_btn.set_tooltip_text(f"Remove '{tag}'")
        dismiss_btn.connect("clicked", lambda _btn, t=tag: self._on_remove_tag(t))
        chip_box.append(dismiss_btn)

        # Create the main button to hold the box
        chip = Gtk.Button()
        chip.set_child(chip_box)
        chip.add_css_class("tag-chip")
        chip.add_css_class("pill")

        return chip

    def _refresh_tags_display(self) -> None:
        """Refresh the tags FlowBox display for current track."""
        # Clear existing tags
        while True:
            child = self._tags_flowbox.get_first_child()
            if child is None:
                break
            self._tags_flowbox.remove(child)

        if self._current_track is None:
            return

        # Get tags for current track
        track_path = Path(self._current_track.filepath)
        tags = self._tag_store.get_tags(track_path)

        # Add chip for each tag
        for tag in tags:
            chip = self._create_tag_chip(tag)
            self._tags_flowbox.append(chip)

        # Update completion suggestions
        self._update_tag_completion()

    def _on_tag_entry_activate(self, entry: Gtk.Entry) -> None:
        """Handle Enter key in tag entry."""
        self._on_add_tag_clicked(None)

    def _on_add_tag_clicked(self, _btn: Gtk.Button | None) -> None:
        """Add tag from entry to current track."""
        if self._current_track is None:
            return

        tag = self._tag_entry.get_text().strip()
        if not tag:
            return

        track_path = Path(self._current_track.filepath)
        self._tag_store.add_tag(track_path, tag)

        # Clear entry and refresh display
        self._tag_entry.set_text("")
        self._refresh_tags_display()

    def _on_remove_tag(self, tag: str) -> None:
        """Remove tag from current track."""
        if self._current_track is None:
            return

        track_path = Path(self._current_track.filepath)
        self._tag_store.remove_tag(track_path, tag)
        self._refresh_tags_display()

    def _create_pill(self, text: str) -> Gtk.Label:
        """Create a pill-style label with the 'pill' CSS class."""
        label = Gtk.Label(label=text)
        label.add_css_class("pill")
        return label

    def _build_controls_section(self) -> None:
        """Build the playback controls section."""
        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)

        # Seekbar
        seek_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        self._current_time_label = Gtk.Label(label="0:00")
        self._current_time_label.add_css_class("caption")
        self._current_time_label.set_width_chars(5)
        seek_box.append(self._current_time_label)

        self._seek_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self._seek_scale.set_range(0, 100)
        self._seek_scale.set_value(0)
        self._seek_scale.set_draw_value(False)
        self._seek_scale.set_hexpand(True)
        self._seek_scale.connect("change-value", self._on_seek_changed)
        seek_box.append(self._seek_scale)

        self._total_time_label = Gtk.Label(label="0:00")
        self._total_time_label.add_css_class("caption")
        self._total_time_label.set_width_chars(5)
        seek_box.append(self._total_time_label)

        controls_box.append(seek_box)

        # Transport buttons (prev, play/pause, next)
        transport_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        transport_box.set_halign(Gtk.Align.CENTER)

        self._prev_btn = Gtk.Button.new_from_icon_name("media-skip-backward-symbolic")
        self._prev_btn.set_tooltip_text("Previous track")
        self._prev_btn.connect("clicked", self._on_prev_clicked)
        transport_box.append(self._prev_btn)

        self._play_btn = Gtk.Button.new_from_icon_name("media-playback-start-symbolic")
        self._play_btn.set_tooltip_text("Play")
        self._play_btn.add_css_class("suggested-action")
        self._play_btn.connect("clicked", self._on_play_clicked)
        transport_box.append(self._play_btn)

        self._next_btn = Gtk.Button.new_from_icon_name("media-skip-forward-symbolic")
        self._next_btn.set_tooltip_text("Next track")
        self._next_btn.connect("clicked", self._on_next_clicked)
        transport_box.append(self._next_btn)

        controls_box.append(transport_box)

        # Volume slider
        volume_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        volume_box.set_halign(Gtk.Align.CENTER)

        volume_icon = Gtk.Image.new_from_icon_name("audio-volume-medium-symbolic")
        volume_box.append(volume_icon)

        self._volume_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self._volume_scale.set_range(0.0, 1.0)
        self._volume_scale.set_value(VOLUME_DEFAULT)
        self._volume_scale.set_draw_value(False)
        self._volume_scale.set_size_request(100, -1)
        self._volume_scale.connect("value-changed", self._on_volume_changed)
        volume_box.append(self._volume_scale)

        controls_box.append(volume_box)

        # Cue injection section
        cues_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        cues_box.set_halign(Gtk.Align.CENTER)
        cues_box.set_margin_top(8)

        self._apply_cues_btn = Gtk.Button(label="Apply Suggested Cues")
        self._apply_cues_btn.set_tooltip_text(
            "Analyze track structure and inject cue points into Mixxx"
        )
        self._apply_cues_btn.connect("clicked", self._on_apply_cues_clicked)
        cues_box.append(self._apply_cues_btn)

        self._cues_spinner = Gtk.Spinner()
        self._cues_spinner.set_visible(False)
        cues_box.append(self._cues_spinner)

        controls_box.append(cues_box)

        self.append(controls_box)

    def _init_gstreamer(self) -> None:
        """Initialize the GStreamer playbin element."""
        if Gst is None:
            logger.warning("GStreamer not available - audio playback disabled")
            self._playbin = None
            return

        try:
            # Initialize GStreamer
            Gst.init(None)

            # Create playbin
            self._playbin = Gst.ElementFactory.make("playbin", "playbin")
            if self._playbin is None:
                logger.error("Failed to create GStreamer playbin element")
                return

            # Set initial volume
            self._playbin.set_property("volume", VOLUME_DEFAULT)

            # Connect to bus messages
            bus = self._playbin.get_bus()
            if bus:
                bus.add_signal_watch()
                bus.connect("message", self._on_bus_message)

        except Exception:
            logger.exception("Failed to initialize GStreamer")
            self._playbin = None

    def _on_bus_message(self, bus: Any, message: Any) -> None:
        """Handle GStreamer bus messages."""
        if Gst is None:
            return
        if message.type == Gst.MessageType.EOS:
            # End of stream - reset to beginning
            self._stop_playback()
            self._seek_scale.set_value(0)
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("GStreamer error: %s (debug: %s)", err, debug)
            self._stop_playback()
        elif message.type == Gst.MessageType.DURATION_CHANGED:
            # Update duration
            self._update_duration()

    def _update_duration(self) -> bool:
        """Update the duration display from GStreamer."""
        if self._playbin is None or Gst is None:
            return False

        success, duration = self._playbin.query_duration(Gst.Format.TIME)
        if success:
            self._duration_ns = duration
            seconds = duration / Gst.SECOND
            self._total_time_label.set_text(self._format_time(seconds))
            self._seek_scale.set_range(0, max(1, int(seconds)))

        return False

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as M:SS."""
        total = max(0, int(seconds))
        mins = total // 60
        secs = total % 60
        return f"{mins}:{secs:02d}"

    def _start_seek_updates(self) -> None:
        """Start the seek bar update timeout."""
        if self._seek_timeout_id is None:
            self._seek_timeout_id = GLib.timeout_add(SEEK_UPDATE_INTERVAL_MS, self._update_position)

    def _stop_seek_updates(self) -> None:
        """Stop the seek bar update timeout."""
        if self._seek_timeout_id is not None:
            GLib.source_remove(self._seek_timeout_id)
            self._seek_timeout_id = None

    def _update_position(self) -> bool:
        """Update the seek bar position from GStreamer."""
        if self._playbin is None or not self._is_playing or Gst is None:
            return False

        success, position = self._playbin.query_position(Gst.Format.TIME)
        if success:
            seconds = position / Gst.SECOND
            self._seek_scale.set_value(int(seconds))
            self._current_time_label.set_text(self._format_time(seconds))

        return True  # Continue timeout

    def _on_seek_changed(self, scale: Gtk.Scale, scroll_type: Any, value: float) -> bool:
        """Handle seek bar value change."""
        if self._playbin is not None and self._duration_ns > 0 and Gst is not None:
            ns = int(value * Gst.SECOND)
            self._playbin.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                ns,
            )
        return False

    def _on_volume_changed(self, scale: Gtk.Scale) -> None:
        """Handle volume slider change."""
        if self._playbin is not None:
            self._playbin.set_property("volume", scale.get_value())

    def _on_prev_clicked(self, _btn: Gtk.Button) -> None:
        """Emit prev-track signal."""
        self.emit("prev-track")

    def _on_play_clicked(self, _btn: Gtk.Button) -> None:
        """Toggle play/pause state."""
        if self._is_playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _on_next_clicked(self, _btn: Gtk.Button) -> None:
        """Emit next-track signal."""
        self.emit("next-track")

    def _start_playback(self) -> None:
        """Start or resume playback."""
        if self._playbin is None or self._current_track is None or Gst is None:
            return

        if self._playbin.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to start playback")
            return

        self._is_playing = True
        self._play_btn.set_icon_name("media-playback-pause-symbolic")
        self._play_btn.set_tooltip_text("Pause")
        self._start_seek_updates()

    def _pause_playback(self) -> None:
        """Pause playback."""
        if self._playbin is None or Gst is None:
            return

        self._playbin.set_state(Gst.State.PAUSED)
        self._is_playing = False
        self._play_btn.set_icon_name("media-playback-start-symbolic")
        self._play_btn.set_tooltip_text("Play")
        self._stop_seek_updates()

    def _stop_playback(self) -> None:
        """Stop playback completely."""
        if self._playbin is not None and Gst is not None:
            self._playbin.set_state(Gst.State.NULL)
        self._is_playing = False
        self._play_btn.set_icon_name("media-playback-start-symbolic")
        self._play_btn.set_tooltip_text("Play")
        self._stop_seek_updates()

    def _load_cover_art(self, filepath: str) -> None:
        """Load cover art from cache or extract and cache it."""
        cache_path = _get_cache_path(filepath)

        # Check cache first
        if cache_path.exists():
            try:
                self._display_cover_art(str(cache_path))
                return
            except Exception:
                logger.exception("Failed to load cached cover art")

        # Extract from file
        art_data = _extract_cover_art(filepath)
        if art_data:
            try:
                # Save to cache
                cache_path.write_bytes(art_data)
                self._display_cover_art(str(cache_path))
                return
            except Exception:
                logger.exception("Failed to cache cover art")

        # No cover art - show placeholder
        self._show_placeholder()

    def _display_cover_art(self, path: str) -> None:
        """Display cover art from a file path."""
        try:
            if GdkPixbuf is not None:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                    path, COVER_ART_SIZE, COVER_ART_SIZE, True
                )
                self._cover_picture.set_pixbuf(pixbuf)
                self._cover_picture.set_visible(True)
                self._cover_icon.set_visible(False)
            else:
                # Fallback if GdkPixbuf is not available
                self._show_placeholder()
        except Exception:
            logger.exception("Failed to display cover art")
            self._show_placeholder()

    def _show_placeholder(self) -> None:
        """Show the placeholder icon instead of cover art."""
        self._cover_picture.set_visible(False)
        self._cover_icon.set_visible(True)

    def load_track(self, track: LibraryTrackModel) -> None:
        """Load a track into the preview panel.

        Extracts cover art, sets metadata labels, and resets playback state.

        Args:
            track: The track model to load.
        """
        self._current_track = track

        # Stop any current playback
        self._stop_playback()

        # Reset seek bar
        self._seek_scale.set_value(0)
        self._current_time_label.set_text("0:00")

        # Load cover art
        self._load_cover_art(track.filepath)

        # Update metadata labels
        self._title_label.set_text(track.title or track.display_title)
        self._artist_label.set_text(track.artist or "Unknown Artist")

        # Album and year (extracted from metadata if available)
        album_text = self._extract_album_info(track.filepath)
        self._album_label.set_text(album_text)

        # Update info pills
        self._bpm_pill.set_text(f"{track.bpm:.0f} BPM" if track.bpm > 0 else "— BPM")
        self._duration_pill.set_text(track.duration_formatted)
        self._format_pill.set_text(track.file_format.upper() if track.file_format else "—")

        # Key (would be extracted from metadata if available)
        key = self._extract_key(track.filepath)
        self._key_pill.set_text(key if key else "—")

        # Refresh tags display
        self._refresh_tags_display()

        # Set up GStreamer source
        if self._playbin is not None and Gst is not None:
            self._playbin.set_property("uri", f"file://{track.filepath}")
            self._update_duration()

    def _extract_album_info(self, filepath: str) -> str:
        """Extract album and year from metadata."""
        try:
            from mutagen import File  # type: ignore[import-untyped]

            audio = File(filepath)
            if audio is None or not hasattr(audio, "tags") or audio.tags is None:
                return ""

            album = ""
            year = ""

            # Try various tag formats
            if hasattr(audio.tags, "get"):
                # Vorbis comments (FLAC, OGG)
                album = audio.tags.get("ALBUM", [""])[0] if "ALBUM" in audio.tags else ""
                year = audio.tags.get("DATE", [""])[0] if "DATE" in audio.tags else ""
                if not year:
                    year = audio.tags.get("YEAR", [""])[0] if "YEAR" in audio.tags else ""
            elif hasattr(audio.tags, "values"):
                # ID3 (MP3)
                for tag in audio.tags.values():
                    if hasattr(tag, "FrameID"):
                        if tag.FrameID == "TALB":
                            album = str(tag)
                        elif tag.FrameID in ("TDRC", "TYER"):
                            year = str(tag)

            if album and year:
                return f"{album} • {year}"
            return album or year or ""

        except Exception:
            logger.exception("Failed to extract album info")
            return ""

    def _extract_key(self, filepath: str) -> str | None:
        """Extract musical key from metadata."""
        try:
            from mutagen import File  # type: ignore[import-untyped]

            audio = File(filepath)
            if audio is None or not hasattr(audio, "tags") or audio.tags is None:
                return None

            # Try various key tag formats
            if hasattr(audio.tags, "get"):
                for key in ["KEY", "TKEY", "INITIALKEY", "INITIAL_KEY"]:
                    if key in audio.tags:
                        val = audio.tags[key]
                        if isinstance(val, list):
                            return val[0] if val else None
                        return str(val) if val else None

            return None
        except Exception:
            logger.exception("Failed to extract key")
            return None

    def _on_apply_cues_clicked(self, _btn: Gtk.Button) -> None:
        """Handle Apply Suggested Cues button click.

        Runs structural analysis and cue injection in a background thread.
        """
        if self._current_track is None:
            return

        # Disable button and show spinner
        self._apply_cues_btn.set_sensitive(False)
        self._cues_spinner.set_visible(True)
        self._cues_spinner.start()

        # Run analysis in background thread
        thread = threading.Thread(
            target=self._run_cue_analysis,
            args=(Path(self._current_track.filepath),),
            daemon=True,
        )
        thread.start()

    def _run_cue_analysis(self, track_path: Path) -> None:
        """Run structural analysis and inject cues into Mixxx.

        This method runs in a background thread.
        """
        try:
            # Analyze track structure
            analyzer = StructuralAnalyzer()
            analysis = analyzer.analyze(track_path)

            # Predict cue points
            cue_points = predict_cue_points(analysis)

            # Write to Mixxx DB
            mixxx_sync = MixxxSync()
            if mixxx_sync.available and mixxx_sync.db_path is not None:
                count = mixxx_sync.write_cue_points(mixxx_sync.db_path, track_path, cue_points)
                GLib.idle_add(self._on_cues_applied, True, f"Applied {count} cue points")
            else:
                GLib.idle_add(self._on_cues_applied, False, "Mixxx database not found")

        except Exception as e:
            logger.exception("Failed to apply cue points")
            GLib.idle_add(self._on_cues_applied, False, str(e))

    def _on_cues_applied(self, success: bool, message: str) -> bool:
        """Callback for when cue analysis completes.

        Runs on the main thread via GLib.idle_add.

        Args:
            success: Whether the operation succeeded
            message: Status message to display

        Returns:
            False to remove the idle callback
        """
        # Re-enable button and hide spinner
        self._apply_cues_btn.set_sensitive(True)
        self._cues_spinner.stop()
        self._cues_spinner.set_visible(False)

        # Show toast notification
        if success:
            self._show_toast(message)
        else:
            self._show_toast(f"Failed: {message}")

        return False  # Remove idle callback

    def _show_toast(self, message: str) -> None:
        """Show a toast notification.

        Args:
            message: Message to display in the toast
        """
        # Get the parent window to find a ToastOverlay
        parent = self.get_parent()
        while parent is not None:
            if hasattr(parent, "add_toast"):
                # Adw.ToastOverlay or similar
                try:
                    import gi

                    gi.require_version("Adw", "1")
                    from gi.repository import Adw  # type: ignore[import-unresolved]  # noqa: E402

                    toast = Adw.Toast.new(message)
                    parent.add_toast(toast)
                    return
                except (ImportError, ValueError):
                    pass
            parent = parent.get_parent()

        # Fallback: just log the message
        logger.info(f"Toast: {message}")

    def dispose(self) -> None:
        """Clean up GStreamer resources."""
        self._stop_playback()
        if self._playbin is not None and Gst is not None:
            bus = self._playbin.get_bus()
            if bus:
                bus.remove_signal_watch()
            self._playbin.set_state(Gst.State.NULL)
            self._playbin = None
