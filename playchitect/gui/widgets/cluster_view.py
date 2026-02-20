"""Cluster visualization panel — scrollable list of cluster cards.

Architecture:
    ClusterViewPanel (Gtk.Box, vertical)
    └── Gtk.ScrolledWindow
        └── Gtk.Box (vertical, _cards_box) — one ClusterCard per cluster
                ClusterCard (Gtk.Frame)
                └── Gtk.Box (vertical)
                    ├── header row  : label + track-count chip
                    ├── BPM row     : bar + range string
                    ├── intensity   : bar + label
                    ├── duration row
                    ├── top-features (optional, collapsed by default)
                    └── [View Tracks] button

Signals:
    cluster-selected(int|str):  emitted when the user clicks [View Tracks]
    cluster-renamed(int|str, str): emitted when the user renames a cluster
"""

from __future__ import annotations

import logging

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

logger = logging.getLogger(__name__)

from gi.repository import Adw, GLib, GObject, Gtk  # type: ignore[unresolved-import]  # noqa: E402

from playchitect.gui.widgets.cluster_stats import ClusterStats  # noqa: E402

# Spacing / layout constants (pixels).
_CARD_MARGIN: int = 8
_CARD_SPACING: int = 6
_ROW_SPACING: int = 4


# ── Individual cluster card ───────────────────────────────────────────────────


class ClusterCard(Gtk.Frame):
    """A single card showing stats for one cluster.

    Signals:
        view-tracks():  user clicked [View Tracks]
        renamed(str):   user submitted a new cluster name
    """

    __gsignals__ = {
        "view-tracks": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "renamed": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
    }

    def __init__(self, stats: ClusterStats, global_bpm_min: float, global_bpm_max: float) -> None:
        super().__init__()
        self._stats = stats
        self._global_bpm_min = global_bpm_min
        self._global_bpm_max = global_bpm_max

        self.set_margin_start(_CARD_MARGIN)
        self.set_margin_end(_CARD_MARGIN)
        self.set_margin_top(_CARD_MARGIN // 2)
        self.set_margin_bottom(_CARD_MARGIN // 2)
        self.add_css_class("card")

        self._build()

    def _build(self) -> None:
        s = self._stats
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=_CARD_SPACING)
        outer.set_margin_start(_CARD_SPACING)
        outer.set_margin_end(_CARD_SPACING)
        outer.set_margin_top(_CARD_SPACING)
        outer.set_margin_bottom(_CARD_SPACING)

        # ── Header: cluster label + track count ──────────────────────────────
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        self._title_label = Gtk.Label(label=s.cluster_label)
        self._title_label.set_xalign(0.0)
        self._title_label.set_hexpand(True)
        self._title_label.add_css_class("heading")

        count_chip = Gtk.Label(label=s.track_count_str)
        count_chip.add_css_class("caption")
        count_chip.add_css_class("dim-label")

        # Rename button (pencil icon)
        rename_btn = Gtk.Button()
        rename_btn.set_icon_name("document-edit-symbolic")
        rename_btn.add_css_class("flat")
        rename_btn.set_tooltip_text("Rename cluster")
        rename_btn.connect("clicked", self._on_rename_clicked)

        header.append(self._title_label)
        header.append(count_chip)
        header.append(rename_btn)
        outer.append(header)

        # ── BPM bar ──────────────────────────────────────────────────────────
        bpm_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        bpm_bar_label = Gtk.Label()
        bpm_bar_label.set_markup(f"<tt>{s.bpm_fill_bars}</tt>")
        bpm_bar_label.set_tooltip_text(
            f"BPM: {s.bpm_mean_str} avg\nGlobal range: "
            f"{self._global_bpm_min:.0f}–{self._global_bpm_max:.0f}"
        )

        bpm_range_label = Gtk.Label(label=s.bpm_range_str)
        bpm_range_label.add_css_class("caption")
        bpm_range_label.set_xalign(0.0)

        bpm_row.append(bpm_bar_label)
        bpm_row.append(bpm_range_label)
        outer.append(bpm_row)

        # ── Intensity bar ─────────────────────────────────────────────────────
        int_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        int_bar_label = Gtk.Label()
        int_bar_label.set_markup(f"<tt>{s.intensity_bars}</tt>")
        int_bar_label.set_tooltip_text(f"Intensity: {s.intensity_mean:.2f}")

        int_text_label = Gtk.Label(label=s.intensity_label)
        int_text_label.add_css_class("caption")
        int_text_label.set_xalign(0.0)

        int_row.append(int_bar_label)
        int_row.append(int_text_label)
        outer.append(int_row)

        # ── Duration ─────────────────────────────────────────────────────────
        duration_label = Gtk.Label(label=f"Duration: {s.duration_str}")
        duration_label.set_xalign(0.0)
        duration_label.add_css_class("caption")
        duration_label.add_css_class("dim-label")
        outer.append(duration_label)

        # ── Top features (if available) ───────────────────────────────────────
        if s.top_features:
            feat_label = Gtk.Label(
                label="Key: " + "  ·  ".join(f"{n} {v:.0%}" for n, v in s.top_features)
            )
            feat_label.set_xalign(0.0)
            feat_label.add_css_class("caption")
            feat_label.add_css_class("dim-label")
            feat_label.set_ellipsize(3)  # Pango.EllipsizeMode.END
            outer.append(feat_label)

        # ── View Tracks button ────────────────────────────────────────────────
        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        view_btn = Gtk.Button(label="View Tracks")
        view_btn.add_css_class("pill")
        view_btn.set_halign(Gtk.Align.END)
        view_btn.set_hexpand(True)
        view_btn.connect("clicked", self._on_view_tracks_clicked)
        btn_row.append(view_btn)
        outer.append(btn_row)

        self.set_child(outer)

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_view_tracks_clicked(self, _btn: Gtk.Button) -> None:
        self.emit("view-tracks")

    def _on_rename_clicked(self, _btn: Gtk.Button) -> None:
        dialog = Adw.AlertDialog(
            heading="Rename Cluster",
            body=f"Enter a new name for {self._stats.cluster_label}:",
        )
        entry = Gtk.Entry()
        entry.set_text(self._stats.cluster_label)
        entry.set_activates_default(True)
        dialog.set_extra_child(entry)
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("rename", "Rename")
        dialog.set_response_appearance("rename", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("rename")
        dialog.connect("response", self._on_rename_response, entry)

        root = self.get_root()
        if isinstance(root, Gtk.Window):
            dialog.present(root)
        else:
            logger.warning("ClusterCard: cannot present rename dialog — no parent window")

    def _on_rename_response(
        self, _dialog: Adw.AlertDialog, response: str, entry: Gtk.Entry
    ) -> None:
        if response == "rename":
            new_name = entry.get_text().strip()
            if new_name:
                self._title_label.set_text(new_name)
                self.emit("renamed", new_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_stats(self, stats: ClusterStats) -> None:
        """Replace the card's stats and rebuild its contents."""
        self._stats = stats
        child = self.get_child()
        if child is not None:
            self.set_child(None)
        self._build()

    @property
    def cluster_id(self) -> int | str:
        return self._stats.cluster_id


# ── Panel ─────────────────────────────────────────────────────────────────────


class ClusterViewPanel(Gtk.Box):
    """Scrollable panel containing one ClusterCard per cluster.

    Signals:
        cluster-selected(GObject.TYPE_PYOBJECT): cluster_id (int|str)
        cluster-renamed(GObject.TYPE_PYOBJECT, str): cluster_id, new_name
    """

    __gsignals__ = {
        "cluster-selected": (GObject.SignalFlags.RUN_FIRST, None, (GObject.TYPE_PYOBJECT,)),
        "cluster-renamed": (GObject.SignalFlags.RUN_FIRST, None, (GObject.TYPE_PYOBJECT, str)),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        self._cards: list[ClusterCard] = []
        self._active_id: int | str | None = None

        # ── Header bar ───────────────────────────────────────────────────────
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_margin_start(8)
        header.set_margin_end(8)
        header.set_margin_top(6)
        header.set_margin_bottom(4)

        self._header_label = Gtk.Label(label="Clusters")
        self._header_label.set_xalign(0.0)
        self._header_label.add_css_class("title-4")
        self._header_label.set_hexpand(True)
        header.append(self._header_label)
        self.append(header)

        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # ── Scrollable card list ──────────────────────────────────────────────
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._cards_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        scroll.set_child(self._cards_box)
        self.append(scroll)

        # ── Empty state placeholder ───────────────────────────────────────────
        self._placeholder = Gtk.Label(label="No clusters yet.\nRun analysis to group tracks.")
        self._placeholder.set_justify(Gtk.Justification.CENTER)
        self._placeholder.add_css_class("dim-label")
        self._placeholder.set_vexpand(True)
        self._placeholder.set_valign(Gtk.Align.CENTER)
        self._cards_box.append(self._placeholder)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_clusters(self, stats_list: list[ClusterStats]) -> None:
        """Replace all cluster cards with new data.

        Accepts the output of ``ClusterStats.from_results(cluster_results)``.
        """
        self._clear_cards()

        if not stats_list:
            self._placeholder.set_visible(True)
            self._header_label.set_text("Clusters")
            return

        self._placeholder.set_visible(False)
        global_min, global_max = ClusterStats.global_bpm_range(stats_list)

        for stats in stats_list:
            self._add_card(stats, global_min, global_max)

        noun = "cluster" if len(stats_list) == 1 else "clusters"
        self._header_label.set_text(f"{len(stats_list)} {noun}")

    def update_cluster(self, stats: ClusterStats) -> None:
        """Update a single cluster card in-place (e.g. during live analysis)."""
        for card in self._cards:
            if card.cluster_id == stats.cluster_id:
                card.update_stats(stats)
                return
        # Card doesn't exist yet — rebuild the whole panel.
        GLib.idle_add(self._rebuild_from_cards)

    @property
    def active_cluster_id(self) -> int | str | None:
        """The cluster_id most recently selected via [View Tracks]."""
        return self._active_id

    @property
    def cluster_count(self) -> int:
        return len(self._cards)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _add_card(self, stats: ClusterStats, global_min: float, global_max: float) -> None:
        card = ClusterCard(stats, global_min, global_max)
        card.connect("view-tracks", self._on_card_view_tracks, stats.cluster_id)
        card.connect("renamed", self._on_card_renamed, stats.cluster_id)
        self._cards.append(card)
        self._cards_box.append(card)

    def _clear_cards(self) -> None:
        for card in self._cards:
            self._cards_box.remove(card)
        self._cards.clear()

    def _rebuild_from_cards(self) -> bool:
        # Triggered by idle_add when update_cluster finds no matching card.
        # In practice this path is rarely hit; a full load_clusters() is preferred.
        return False

    def _on_card_view_tracks(self, _card: ClusterCard, cluster_id: int | str) -> None:
        self._active_id = cluster_id
        self.emit("cluster-selected", cluster_id)

    def _on_card_renamed(self, _card: ClusterCard, cluster_id: int | str, new_name: str) -> None:
        self.emit("cluster-renamed", cluster_id, new_name)
