"""Tests for the seed-playlist generation wiring on PlaychitectWindow.

These tests define the contract for TASK-12: connecting the
``make-playlist-seed`` signal from LibraryView to a handler on
PlaychitectWindow that runs ``generate_playlist_from_seed`` on a
background thread, passes the result to PlaylistsView, and navigates
to the playlists page.

All tests currently FAIL because the signal connection, handler method,
and background-thread orchestration do not exist yet.  Once TASK-11 is
implemented:

  - test_signal_connection_exists             — ``make-playlist-seed`` connected
  - test_handler_triggers_intensity_analysis  — IntensityAnalyzer.analyze_batch called
  - test_handler_calls_generate_playlist_from_seed — core function called with correct args
  - test_result_loaded_into_playlists_view    — PlaylistsView.load_clusters called
  - test_navigation_switches_to_playlists_view — view_stack navigates to "playlists"
  - test_error_shows_toast_and_reenables_ui   — UI re-enabled on failure
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from playchitect.core.clustering import ClusterResult  # noqa: E402
from playchitect.core.intensity_analyzer import IntensityFeatures  # noqa: E402
from playchitect.core.metadata_extractor import TrackMetadata  # noqa: E402

# Import after conftest has installed gi mocks.
from playchitect.gui.windows.main_window import PlaychitectWindow  # noqa: E402

# ── Shared helpers ───────────────────────────────────────────────────────────


def _make_bare_window() -> PlaychitectWindow:
    """Return a PlaychitectWindow via __new__ with all required attributes mocked.

    This avoids any GTK widget construction and lets us test individual
    methods in isolation.
    """
    w = PlaychitectWindow.__new__(PlaychitectWindow)
    w._track_title = "Playchitect"
    w._previewer = MagicMock()
    w._preview_chip = MagicMock()
    w._spinner = MagicMock()
    w._arc_dropdown = MagicMock()
    w._menu_button = MagicMock()
    w._nav_list = MagicMock()
    w._view_stack = MagicMock()
    w._split_view = MagicMock()
    w._playlists_view = MagicMock()
    w._library_view = MagicMock()
    w._set_builder_view = MagicMock()
    w._export_view = MagicMock()
    w._metadata_map: dict[Path, TrackMetadata] = {}
    w._intensity_map: dict[Path, IntensityFeatures] = {}
    w._clusters: list[ClusterResult] = []
    w._active_arc = None
    w._original_clusters: list[ClusterResult] = []
    w._playlist_namer = MagicMock()
    w._cluster_names: dict[int | str, str] = {}
    w._cluster_btn = MagicMock()
    w._target_spin = MagicMock()
    w._target_unit = MagicMock()
    w._play_history = MagicMock()
    w._prefer_fresh = False
    return w


class _InlineThread:
    """Stand-in for ``threading.Thread`` that runs its target synchronously.

    ``_on_make_playlist_seed`` dispatches ``_seed_generation_worker`` to a real
    daemon thread, so assertions on the worker's effects would otherwise race
    it.  ``.start()`` here runs the target on the calling thread instead, which
    makes those effects deterministic without changing what is asserted.
    """

    def __init__(
        self,
        target: Callable[..., object] | None = None,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
        daemon: bool | None = None,
    ) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self) -> None:
        """Run the target immediately on the calling thread."""
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout: float | None = None) -> None:
        """No-op: the target has already run to completion in ``start()``."""


@contextmanager
def _inline_thread() -> Iterator[None]:
    """Make background worker dispatch synchronous for the duration of the block."""
    with patch("threading.Thread", _InlineThread):
        yield


def _make_sample_metadata(n: int = 3) -> dict[Path, TrackMetadata]:
    """Return a small metadata map with *n* synthetic tracks."""
    result: dict[Path, TrackMetadata] = {}
    for i in range(n):
        p = Path(f"/music/track{i}.flac")
        result[p] = TrackMetadata(
            filepath=p,
            title=f"Track {i}",
            artist=f"Artist {i}",
            bpm=120.0 + i,
            duration=240.0 + i * 30,
        )
    return result


def _make_sample_intensity(
    paths: list[Path],
) -> dict[Path, IntensityFeatures]:
    """Return a minimal intensity map for the given paths.

    Uses the real IntensityFeatures field names from intensity_analyzer.py.
    The ``file_path`` kw_only parameter is required by __post_init__.
    """
    result: dict[Path, IntensityFeatures] = {}
    for p in paths:
        result[p] = IntensityFeatures(
            file_hash="deadbeef",
            rms_energy=0.5,
            brightness=0.4,
            sub_bass_energy=0.1,
            kick_energy=0.15,
            bass_harmonics=0.05,
            percussiveness=0.6,
            file_path=str(p),
        )
    return result


def _make_cluster_result() -> ClusterResult:
    """Return a synthetic ClusterResult representing a seed-generated playlist."""
    return ClusterResult(
        cluster_id="seed",
        tracks=[Path("/music/track0.flac"), Path("/music/track1.flac")],
        bpm_mean=121.0,
        bpm_std=1.0,
        track_count=2,
        total_duration=510.0,
        genre="Like: Track 0",
    )


# ── Test: Signal connection ─────────────────────────────────────────────────


class TestSignalConnection:
    """Verify that PlaychitectWindow connects to the make-playlist-seed signal."""

    def test_signal_connection_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PlaychitectWindow.__init__ must connect to 'make-playlist-seed' on LibraryView.

        The connection should happen in _build_view_stack(), alongside the
        existing signal connections (scan-complete, track-selected, preview-toggled).
        """
        # Patch all external deps so __init__ can run.
        mock_previewer = MagicMock()
        mock_previewer.launcher_name.return_value = None

        mock_config = MagicMock()
        mock_config.get_test_music_path.return_value = None

        # Capture what .connect() is called with on LibraryView.
        mock_library_view = MagicMock()
        connect_calls: list[tuple[object, ...]] = []
        mock_library_view.connect.side_effect = lambda *args, **kwargs: (
            connect_calls.append(args),
            0,  # return a signal handler ID
        )[-1]

        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.TrackPreviewer",
            MagicMock(return_value=mock_previewer),
        )
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.LibraryView",
            MagicMock(return_value=mock_library_view),
        )
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.PlaylistsView",
            MagicMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.SetBuilderView",
            MagicMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.ExportView",
            MagicMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.get_config",
            MagicMock(return_value=mock_config),
        )
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.PreferencesWindow",
            MagicMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.TrackPreviewPanel",
            MagicMock(return_value=MagicMock()),
        )

        PlaychitectWindow()

        # Assert that "make-playlist-seed" was among the connected signals.
        signal_names = [call[0] for call in connect_calls if call]
        assert "make-playlist-seed" in signal_names, (
            "PlaychitectWindow must connect to 'make-playlist-seed' signal "
            "from LibraryView in _build_view_stack()"
        )

        # Also verify the handler is callable (second arg to .connect()).
        for call in connect_calls:
            if call and call[0] == "make-playlist-seed":
                handler = call[1]
                assert callable(handler), "The handler for 'make-playlist-seed' must be callable"


# ── Test: Handler triggers intensity analysis ──────────────────────────────


class TestHandlerIntensityAnalysis:
    """Verify the handler runs intensity analysis when _intensity_map is empty."""

    def test_handler_triggers_intensity_analysis(self) -> None:
        """_on_make_playlist_seed must call IntensityAnalyzer.analyze_batch
        when self._intensity_map is empty.

        Seed generation requires intensity features for all candidate tracks.
        If clustering hasn't been run yet (intensity_map is empty), the
        handler must compute them before calling generate_playlist_from_seed.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        window._metadata_map = metadata
        window._intensity_map = {}  # empty — should trigger analysis

        mock_intensity_result = _make_sample_intensity(list(metadata.keys()))

        with (
            _inline_thread(),
            patch("playchitect.gui.windows.main_window.IntensityAnalyzer") as mock_analyzer_cls,
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ),
            patch("playchitect.gui.windows.main_window.get_config") as mock_config_fn,
            patch("playchitect.gui.windows.main_window.GLib"),
        ):
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_batch.return_value = mock_intensity_result
            mock_analyzer_cls.return_value = mock_analyzer
            mock_config = MagicMock()
            mock_config.get_cache_dir.return_value = Path("/fake/cache")
            mock_config_fn.return_value = mock_config

            # Call the handler — this method does not exist yet (TASK-11).
            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            # Verify IntensityAnalyzer was constructed and analyze_batch called.
            mock_analyzer_cls.assert_called_once()
            mock_analyzer.analyze_batch.assert_called_once()
            # The batch should be called with the metadata keys.
            called_paths = mock_analyzer.analyze_batch.call_args[0][0]
            assert set(called_paths) == set(metadata.keys()), (
                "analyze_batch must be called with all paths from _metadata_map"
            )

    def test_handler_skips_intensity_analysis_when_map_populated(self) -> None:
        """_on_make_playlist_seed should NOT call IntensityAnalyzer.analyze_batch
        when self._intensity_map already has entries.

        Avoids redundant work when clustering has already been performed.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        window._metadata_map = metadata
        window._intensity_map = _make_sample_intensity(list(metadata.keys()))

        with (
            _inline_thread(),
            patch("playchitect.gui.windows.main_window.IntensityAnalyzer") as mock_analyzer_cls,
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ),
            patch("playchitect.gui.windows.main_window.get_config") as mock_config_fn,
            patch("playchitect.gui.windows.main_window.GLib"),
        ):
            mock_config = MagicMock()
            mock_config.get_cache_dir.return_value = Path("/fake/cache")
            mock_config_fn.return_value = mock_config

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            mock_analyzer_cls.assert_not_called()


# ── Test: Handler calls generate_playlist_from_seed ────────────────────────


class TestHandlerCallsGenerate:
    """Verify that the handler calls generate_playlist_from_seed with correct args."""

    def test_handler_calls_generate_playlist_from_seed(self) -> None:
        """_on_make_playlist_seed must call generate_playlist_from_seed with
        the seed path, candidate features, metadata, target duration, and
        sequence mode.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        seed_path = Path("/music/track0.flac")
        duration_mins = 60.0
        sequence_mode = "build"

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ) as mock_generate,
            patch("playchitect.gui.windows.main_window.GLib"),
        ):
            window._on_make_playlist_seed(
                window._library_view,
                filepath=str(seed_path),
                duration_mins=duration_mins,
                sequence_mode=sequence_mode,
            )

            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args

            # Positional args: seed_path, candidate_features, metadata_dict,
            #                  target_duration_mins
            assert call_kwargs[0][0] == seed_path, "First arg must be the seed path"
            assert call_kwargs[0][1] == intensity, "Second arg must be the intensity features dict"
            assert call_kwargs[0][2] == metadata, "Third arg must be the metadata dict"
            assert call_kwargs[0][3] == duration_mins, "Fourth arg must be target_duration_mins"
            # Keyword arg: sequence_mode
            assert call_kwargs[1].get("sequence_mode") == sequence_mode, (
                "sequence_mode must be passed as keyword arg"
            )

    def test_handler_passes_filepath_as_path(self) -> None:
        """The handler must convert the filepath string to a Path object
        before calling generate_playlist_from_seed.

        The signal emits a string (GObject limitation), but the core
        function expects a Path.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        filepath_str = "/music/track0.flac"

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ) as mock_generate,
            patch("playchitect.gui.windows.main_window.GLib"),
        ):
            window._on_make_playlist_seed(
                window._library_view,
                filepath=filepath_str,
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            # First positional arg must be a Path, not a string.
            first_arg = mock_generate.call_args[0][0]
            assert isinstance(first_arg, Path), (
                f"seed_path must be a Path, got {type(first_arg).__name__}"
            )
            assert str(first_arg) == filepath_str


# ── Test: Result loaded into PlaylistsView ─────────────────────────────────


class TestResultLoadedIntoPlaylistsView:
    """Verify the ClusterResult is passed to PlaylistsView.load_clusters()."""

    def test_result_loaded_into_playlists_view(self) -> None:
        """On success, the ClusterResult from generate_playlist_from_seed
        must be passed to self._playlists_view.load_clusters([result]).
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        expected_result = _make_cluster_result()

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=expected_result,
            ),
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):
            # Simulate calling the handler — it runs generate_playlist_from_seed
            # synchronously in the test context (no real threading).
            # In the real implementation, GLib.idle_add would schedule the
            # UI update. We mock GLib.idle_add to call the callback immediately.
            idle_add_calls: list[object] = []

            def fake_idle_add(callback: object, *args: object) -> int:
                idle_add_calls.append((callback, args))
                # Execute the callback immediately so we can test the result.
                if callable(callback) and not args:
                    callback()
                elif callable(callback) and args:
                    callback(*args)
                return 0

            mock_glib.idle_add = fake_idle_add

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            window._playlists_view.load_clusters.assert_called_once()
            call_args = window._playlists_view.load_clusters.call_args[0][0]
            assert isinstance(call_args, list), "load_clusters must receive a list"
            assert len(call_args) == 1, "Seed generation produces exactly 1 cluster"
            assert call_args[0] is expected_result, (
                "The ClusterResult must be the one returned by generate_playlist_from_seed"
            )


# ── Test: Navigation switches to playlists view ───────────────────────────


class TestNavigationSwitchesToPlaylistsView:
    """Verify the view stack switches to "playlists" on successful generation."""

    def test_navigation_switches_to_playlists_view(self) -> None:
        """After successful seed generation, _view_stack.set_visible_child_name
        must be called with "playlists" so the user sees the result.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ),
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):
            # Mock GLib.idle_add to execute callbacks immediately.
            def fake_idle_add(callback: object, *args: object) -> int:
                if callable(callback) and not args:
                    callback()
                elif callable(callback) and args:
                    callback(*args)
                return 0

            mock_glib.idle_add = fake_idle_add

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            window._view_stack.set_visible_child_name.assert_called_with("playlists")


# ── Test: Error handling ───────────────────────────────────────────────────


class TestErrorHandling:
    """Verify error handling when generate_playlist_from_seed raises."""

    def test_error_shows_toast_and_reenables_ui(self) -> None:
        """When generate_playlist_from_seed raises an exception:
        - The spinner must stop
        - The cluster button must be re-enabled (set_sensitive(True))
        - An error toast should be shown to the user
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                side_effect=ValueError("seed_path not found in candidate_features"),
            ),
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):
            # Mock GLib.idle_add to execute the error callback immediately.
            def fake_idle_add(callback: object, *args: object) -> int:
                if callable(callback) and not args:
                    callback()
                elif callable(callback) and args:
                    callback(*args)
                return 0

            mock_glib.idle_add = fake_idle_add

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            # Spinner must stop.
            window._spinner.stop.assert_called()

            # Cluster button must be re-enabled.
            window._cluster_btn.set_sensitive.assert_called_with(True)

    def test_error_does_not_load_clusters(self) -> None:
        """When generate_playlist_from_seed raises, load_clusters must NOT be called.

        No partial or empty result should be pushed to the playlists view.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                side_effect=ValueError("empty candidate_features"),
            ),
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):

            def fake_idle_add(callback: object, *args: object) -> int:
                if callable(callback) and not args:
                    callback()
                elif callable(callback) and args:
                    callback(*args)
                return 0

            mock_glib.idle_add = fake_idle_add

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            window._playlists_view.load_clusters.assert_not_called()

    def test_error_does_not_switch_view(self) -> None:
        """When generate_playlist_from_seed raises, view stack must NOT switch."""
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                side_effect=ValueError("test error"),
            ),
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):

            def fake_idle_add(callback: object, *args: object) -> int:
                if callable(callback) and not args:
                    callback()
                elif callable(callback) and args:
                    callback(*args)
                return 0

            mock_glib.idle_add = fake_idle_add

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            window._view_stack.set_visible_child_name.assert_not_called()


# ── Test: Handler signature ────────────────────────────────────────────────


class TestHandlerSignature:
    """Verify the handler method exists and has the correct signature."""

    def test_on_make_playlist_seed_method_exists(self) -> None:
        """PlaychitectWindow must have an _on_make_playlist_seed method."""
        window = _make_bare_window()
        assert hasattr(window, "_on_make_playlist_seed"), (
            "PlaychitectWindow must define _on_make_playlist_seed handler"
        )
        assert callable(window._on_make_playlist_seed), "_on_make_playlist_seed must be callable"

    def test_handler_accepts_four_parameters(self) -> None:
        """_on_make_playlist_seed must accept (self, view, filepath, duration_mins, sequence_mode).

        This matches the GObject signal signature:
          "make-playlist-seed": (RUN_FIRST, None, (str, float, str))

        The signal handler receives the emitting widget as the first
        parameter after self, followed by the three signal parameters.
        """
        import inspect

        window = _make_bare_window()
        method = window._on_make_playlist_seed

        sig = inspect.signature(method)
        param_names = list(sig.parameters.keys())
        # Expected: view, filepath, duration_mins, sequence_mode (plus self is implicit)
        assert len(param_names) == 4, (
            f"_on_make_playlist_seed must accept 4 parameters (view, filepath, "
            f"duration_mins, sequence_mode), got {len(param_names)}: {param_names}"
        )


# ── Test: Spinner state during generation ──────────────────────────────────


class TestSpinnerState:
    """Verify spinner and cluster button state during seed generation."""

    def test_spinner_starts_and_cluster_btn_disabled_on_handler_call(self) -> None:
        """When _on_make_playlist_seed is called, the spinner must start
        and the cluster button must be disabled before the background work begins.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ),
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):
            # Don't let idle_add run callbacks — we want to check the
            # state *before* the idle callbacks fire.
            mock_glib.idle_add = MagicMock()

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            # Spinner must start.
            window._spinner.start.assert_called()

            # Cluster button must be disabled.
            window._cluster_btn.set_sensitive.assert_any_call(False)

    def test_spinner_stops_on_success(self) -> None:
        """After successful generation, the spinner must stop."""
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ),
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):

            def fake_idle_add(callback: object, *args: object) -> int:
                if callable(callback) and not args:
                    callback()
                elif callable(callback) and args:
                    callback(*args)
                return 0

            mock_glib.idle_add = fake_idle_add

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            window._spinner.stop.assert_called()

    def test_cluster_btn_reenabled_on_success(self) -> None:
        """After successful generation, the cluster button must be re-enabled."""
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        intensity = _make_sample_intensity(list(metadata.keys()))
        window._metadata_map = metadata
        window._intensity_map = intensity

        with (
            _inline_thread(),
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ),
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):

            def fake_idle_add(callback: object, *args: object) -> int:
                if callable(callback) and not args:
                    callback()
                elif callable(callback) and args:
                    callback(*args)
                return 0

            mock_glib.idle_add = fake_idle_add

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            window._cluster_btn.set_sensitive.assert_any_call(True)


# ── Test: Background thread dispatch ───────────────────────────────────────


class TestBackgroundThreadDispatch:
    """Verify seed generation runs off the GTK main thread.

    ``_seed_generation_worker`` performs unbounded work: on a cold cache it
    calls ``IntensityAnalyzer.analyze_batch`` across the whole library, which
    takes minutes for a few thousand tracks.  Running it inline on the GTK
    main thread freezes the window for the entire duration — the spinner
    started immediately beforehand never even paints, because the main loop
    never regains control.

    The worker must therefore be dispatched exactly like the other workers in
    main_window.py (``_scan_worker``, ``_cluster_worker``): on a daemon
    ``threading.Thread``, with results marshalled back via ``GLib.idle_add``.
    """

    def test_handler_dispatches_worker_to_background_thread(self) -> None:
        """_on_make_playlist_seed must launch _seed_generation_worker on a daemon thread.

        Mirrors the dispatch convention already used for the scan and cluster
        workers:

            threading.Thread(target=self._scan_worker, args=(...), daemon=True).start()
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        window._metadata_map = metadata
        window._intensity_map = _make_sample_intensity(list(metadata.keys()))

        with patch("threading.Thread") as mock_thread_cls:
            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            mock_thread_cls.assert_called_once()
            _, kwargs = mock_thread_cls.call_args

            assert kwargs.get("target") == window._seed_generation_worker, (
                f"Thread must target _seed_generation_worker, got {kwargs.get('target')!r}"
            )
            assert kwargs.get("args") == (Path("/music/track0.flac"), 90.0, "ramp"), (
                "Thread must receive (seed_path, duration_mins, sequence_mode) as args, "
                f"got {kwargs.get('args')!r}"
            )
            assert kwargs.get("daemon") is True, (
                "Worker thread must be a daemon thread so it cannot block application exit, "
                "matching _scan_worker and _cluster_worker dispatch"
            )
            mock_thread_cls.return_value.start.assert_called_once()

    def test_handler_does_not_run_generation_inline(self) -> None:
        """_on_make_playlist_seed must return before any generation work happens.

        With threading.Thread patched out, the worker body is never executed,
        so a correct implementation performs no analysis and no generation
        during the handler call.  If the handler instead calls the worker
        directly, these mocks are hit and the GTK main loop would have been
        blocked for the whole of that work.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        window._metadata_map = metadata
        window._intensity_map = {}  # cold cache — worst case for main-thread blocking

        with (
            patch("threading.Thread"),
            patch("playchitect.gui.windows.main_window.IntensityAnalyzer") as mock_analyzer_cls,
            patch(
                "playchitect.gui.windows.main_window.generate_playlist_from_seed",
                create=True,
                return_value=_make_cluster_result(),
            ) as mock_generate,
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
        ):
            mock_glib.idle_add = MagicMock()

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

            mock_analyzer_cls.assert_not_called()
            mock_generate.assert_not_called()
            mock_glib.idle_add.assert_not_called()

    def test_ui_state_is_updated_before_thread_dispatch(self) -> None:
        """Spinner start, button disable and title change must all happen on the
        calling (main) thread, before the worker thread is created.

        Those three calls touch GTK widgets, so they must not be moved into the
        worker; and they must precede dispatch, otherwise there is a window in
        which the UI shows no progress at all.
        """
        window = _make_bare_window()
        metadata = _make_sample_metadata(3)
        window._metadata_map = metadata
        window._intensity_map = _make_sample_intensity(list(metadata.keys()))
        window.set_title = MagicMock()

        order: list[str] = []
        window._spinner.start.side_effect = lambda *_a, **_kw: order.append("spinner_start")
        window._cluster_btn.set_sensitive.side_effect = lambda *_a, **_kw: order.append(
            "btn_disabled"
        )
        window.set_title.side_effect = lambda *_a, **_kw: order.append("title_set")

        with patch("threading.Thread") as mock_thread_cls:
            mock_thread_cls.side_effect = lambda *_a, **_kw: (
                order.append("thread_created"),
                MagicMock(),
            )[-1]

            window._on_make_playlist_seed(
                window._library_view,
                filepath="/music/track0.flac",
                duration_mins=90.0,
                sequence_mode="ramp",
            )

        assert "thread_created" in order, f"No worker thread was created; call order was {order}"
        dispatch_index = order.index("thread_created")
        for ui_step in ("spinner_start", "btn_disabled", "title_set"):
            assert ui_step in order, f"{ui_step} never happened; call order was {order}"
            assert order.index(ui_step) < dispatch_index, (
                f"{ui_step} must happen on the calling thread before dispatch; "
                f"call order was {order}"
            )
