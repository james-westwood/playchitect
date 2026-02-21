"""GUI unit tests for ClusterViewPanel and ClusterCard.

Uses the same GTK mock infrastructure as test_track_list.py (installed by
tests/gui/conftest.py before any test module is imported).

All tests bypass __init__ via __new__ so they never touch the real GTK runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from playchitect.gui.widgets.cluster_stats import ClusterStats

if TYPE_CHECKING:
    from playchitect.gui.widgets.cluster_view import ClusterCard, ClusterViewPanel


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_stats(
    cluster_id: int = 1,
    track_count: int = 20,
    bpm_min: float = 120.0,
    bpm_max: float = 128.0,
    bpm_mean: float = 124.0,
    intensity_mean: float = 0.6,
    total_duration: float = 5400.0,
    feature_importance: list[tuple[str, float]] | None = None,
    opener_name: str | None = None,
    closer_name: str | None = None,
) -> ClusterStats:
    return ClusterStats(
        cluster_id=cluster_id,
        track_count=track_count,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        bpm_mean=bpm_mean,
        intensity_mean=intensity_mean,
        hardness_mean=intensity_mean,
        total_duration=total_duration,
        opener_name=opener_name,
        closer_name=closer_name,
        feature_importance=feature_importance or [],
    )


def _make_panel() -> ClusterViewPanel:
    """Return a ClusterViewPanel with __init__ bypassed."""
    from playchitect.gui.widgets.cluster_view import ClusterViewPanel

    panel = ClusterViewPanel.__new__(ClusterViewPanel)
    panel._cards = []
    panel._active_id = None
    panel._header_label = MagicMock()
    panel._cards_box = MagicMock()
    panel._placeholder = MagicMock()
    return panel


def _make_card(stats: ClusterStats | None = None) -> ClusterCard:
    """Return a ClusterCard with __init__ bypassed."""
    from playchitect.gui.widgets.cluster_view import ClusterCard

    card = ClusterCard.__new__(ClusterCard)
    card._stats = stats or _make_stats()
    card._global_bpm_min = 100.0
    card._global_bpm_max = 180.0
    card._title_label = MagicMock()
    return card


# ── TestClusterCardProperties ─────────────────────────────────────────────────


class TestClusterCardProperties:
    def test_cluster_id_matches_stats(self):
        card = _make_card(_make_stats(cluster_id=3))
        assert card.cluster_id == 3

    def test_cluster_id_string(self):
        card = _make_card(_make_stats(cluster_id="2a"))  # type: ignore[arg-type]
        assert card.cluster_id == "2a"


class TestClusterCardBuild:
    """Test the _build method of ClusterCard (mocking GTK calls)."""

    def test_build_with_recommendations(self):
        stats = _make_stats(opener_name="Start Track", closer_name="End Track")
        card = _make_card(stats)
        card.set_child = MagicMock()

        # Mock GTK classes used in _build
        with (
            patch("playchitect.gui.widgets.cluster_view.Gtk.Box"),
            patch("playchitect.gui.widgets.cluster_view.Gtk.Label") as mock_label,
            patch("playchitect.gui.widgets.cluster_view.Gtk.Button"),
        ):
            card._build()

            # Verify that some labels were created with the track names (Finding the exact
            # call is hard with many widgets, but we can check if the names were used)
            label_texts = [call.kwargs.get("label") for call in mock_label.call_args_list]
            assert any("Start: Start Track" in str(t) for t in label_texts)
            assert any("End: End Track" in str(t) for t in label_texts)


# ── TestClusterCardViewTracksSignal ───────────────────────────────────────────


class TestClusterCardViewTracksSignal:
    def test_view_tracks_emitted_on_click(self):
        card = _make_card()
        card.emit = MagicMock()
        card._on_view_tracks_clicked(MagicMock())
        card.emit.assert_called_once_with("view-tracks")


# ── TestClusterCardRenameSignal ───────────────────────────────────────────────


class TestClusterCardRenameSignal:
    def test_rename_emitted_with_new_name(self):
        card = _make_card()
        card.emit = MagicMock()

        fake_entry = MagicMock()
        fake_entry.get_text.return_value = "  Deep House  "
        fake_dialog = MagicMock()

        card._on_rename_response(fake_dialog, "rename", fake_entry)

        card.emit.assert_called_once_with("renamed", "Deep House")
        card._title_label.set_text.assert_called_once_with("Deep House")

    def test_rename_not_emitted_on_cancel(self):
        card = _make_card()
        card.emit = MagicMock()

        fake_entry = MagicMock()
        fake_entry.get_text.return_value = "New Name"
        card._on_rename_response(MagicMock(), "cancel", fake_entry)

        card.emit.assert_not_called()

    def test_rename_not_emitted_when_name_empty(self):
        card = _make_card()
        card.emit = MagicMock()

        fake_entry = MagicMock()
        fake_entry.get_text.return_value = "   "
        card._on_rename_response(MagicMock(), "rename", fake_entry)

        card.emit.assert_not_called()


# ── TestClusterViewPanelLoadClusters ─────────────────────────────────────────


class TestClusterViewPanelLoadClusters:
    def test_load_empty_shows_placeholder(self):
        panel = _make_panel()
        panel.load_clusters([])

        panel._placeholder.set_visible.assert_called_with(True)
        assert panel.cluster_count == 0

    def test_load_empty_resets_header(self):
        panel = _make_panel()
        panel.load_clusters([])

        panel._header_label.set_text.assert_called_with("Clusters")

    def test_load_multiple_hides_placeholder(self):
        panel = _make_panel()
        with patch.object(panel, "_add_card") as mock_add:
            panel.load_clusters([_make_stats(1), _make_stats(2)])
            panel._placeholder.set_visible.assert_called_with(False)
            assert mock_add.call_count == 2

    def test_load_singular_header(self):
        panel = _make_panel()
        with patch.object(panel, "_add_card"):
            panel.load_clusters([_make_stats(1)])
        panel._header_label.set_text.assert_called_with("1 cluster")

    def test_load_plural_header(self):
        panel = _make_panel()
        with patch.object(panel, "_add_card"):
            panel.load_clusters([_make_stats(1), _make_stats(2), _make_stats(3)])
        panel._header_label.set_text.assert_called_with("3 clusters")

    def test_existing_cards_cleared_before_load(self):
        panel = _make_panel()
        old_card = MagicMock()
        panel._cards = [old_card]

        with patch.object(panel, "_add_card"):
            panel.load_clusters([_make_stats(1)])

        panel._cards_box.remove.assert_called_once_with(old_card)
        # _cards was cleared, then _add_card was patched so nothing was re-added
        # directly; length depends on mock. The important thing is remove was called.


# ── TestClusterViewPanelUpdateCluster ─────────────────────────────────────────


class TestClusterViewPanelUpdateCluster:
    def test_update_existing_card(self):
        from playchitect.gui.widgets.cluster_view import ClusterCard

        panel = _make_panel()
        card = _make_card(_make_stats(cluster_id=2))
        panel._cards = [card]

        new_stats = _make_stats(cluster_id=2, track_count=30)
        with patch.object(ClusterCard, "update_stats") as mock_update:
            panel.update_cluster(new_stats)
            mock_update.assert_called_once_with(new_stats)

    def test_update_nonexistent_card_schedules_rebuild(self):
        panel = _make_panel()
        panel._cards = []

        with patch("playchitect.gui.widgets.cluster_view.GLib") as mock_glib:
            panel.update_cluster(_make_stats(cluster_id=99))
            mock_glib.idle_add.assert_called_once()


# ── TestClusterViewPanelClusterSelected ───────────────────────────────────────


class TestClusterViewPanelClusterSelected:
    def test_active_id_set_on_view_tracks(self):
        panel = _make_panel()
        panel.emit = MagicMock()
        fake_card = MagicMock()

        panel._on_card_view_tracks(fake_card, 3)

        assert panel._active_id == 3
        panel.emit.assert_called_once_with("cluster-selected", 3)

    def test_active_id_initially_none(self):
        panel = _make_panel()
        assert panel.active_cluster_id is None

    def test_active_cluster_id_property(self):
        panel = _make_panel()
        panel._active_id = 5
        assert panel.active_cluster_id == 5


# ── TestClusterViewPanelClusterRenamed ────────────────────────────────────────


class TestClusterViewPanelClusterRenamed:
    def test_renamed_signal_forwarded(self):
        panel = _make_panel()
        panel.emit = MagicMock()
        fake_card = MagicMock()

        panel._on_card_renamed(fake_card, 2, "Techno Set")

        panel.emit.assert_called_once_with("cluster-renamed", 2, "Techno Set")


# ── TestClusterViewPanelClusterCount ──────────────────────────────────────────


class TestClusterViewPanelClusterCount:
    def test_count_matches_cards(self):
        panel = _make_panel()
        panel._cards = [_make_card(), _make_card(), _make_card()]
        assert panel.cluster_count == 3

    def test_count_zero_when_empty(self):
        panel = _make_panel()
        assert panel.cluster_count == 0
