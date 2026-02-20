"""Tests for desktop_install module and static data file validity."""

from __future__ import annotations

import configparser
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from playchitect.utils.desktop_install import APP_ID, PNG_SIZES, install, uninstall

# Repo root for static file paths
_REPO_ROOT = Path(__file__).parent.parent.parent


class TestInstallLinux:
    """install() on Linux copies files and runs post-install commands."""

    def test_copies_desktop_file(self, tmp_path):
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.shutil.copy2") as mock_copy,
            patch("playchitect.utils.desktop_install.subprocess.run"),
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            install(system_wide=False)

        dest_dir = tmp_path / ".local" / "share" / "applications"
        assert any(
            c
            == call(
                _REPO_ROOT / "data" / "playchitect.desktop",
                dest_dir / "playchitect.desktop",
            )
            for c in mock_copy.call_args_list
        )

    def test_copies_appdata_xml(self, tmp_path):
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.shutil.copy2") as mock_copy,
            patch("playchitect.utils.desktop_install.subprocess.run"),
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            install(system_wide=False)

        dest_dir = tmp_path / ".local" / "share" / "metainfo"
        assert any(
            c
            == call(
                _REPO_ROOT / "data" / f"{APP_ID}.appdata.xml",
                dest_dir / f"{APP_ID}.appdata.xml",
            )
            for c in mock_copy.call_args_list
        )

    def test_copies_all_png_sizes(self, tmp_path):
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.shutil.copy2") as mock_copy,
            patch("playchitect.utils.desktop_install.subprocess.run"),
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            install(system_wide=False)

        dest_calls = [str(c[0][1]) for c in mock_copy.call_args_list]
        for size in PNG_SIZES:
            expected = str(
                tmp_path
                / ".local"
                / "share"
                / "icons"
                / "hicolor"
                / f"{size}x{size}"
                / "apps"
                / f"{APP_ID}.png"
            )
            assert expected in dest_calls, f"PNG {size}x{size} not copied"

    def test_runs_update_desktop_database_with_correct_path(self, tmp_path):
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.shutil.copy2"),
            patch("playchitect.utils.desktop_install.subprocess.run") as mock_run,
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            install(system_wide=False)

        expected_app_dir = str(tmp_path / ".local" / "share" / "applications")
        full_cmds = [c.args[0] for c in mock_run.call_args_list]
        assert ["update-desktop-database", expected_app_dir] in full_cmds

    def test_runs_gtk_update_icon_cache_with_force_flag(self, tmp_path):
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.shutil.copy2"),
            patch("playchitect.utils.desktop_install.subprocess.run") as mock_run,
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            install(system_wide=False)

        expected_icon_base = str(tmp_path / ".local" / "share" / "icons" / "hicolor")
        full_cmds = [c.args[0] for c in mock_run.call_args_list]
        assert ["gtk-update-icon-cache", "--force", expected_icon_base] in full_cmds

    def test_system_wide_uses_usr_local(self, tmp_path):
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.shutil.copy2") as mock_copy,
            patch("playchitect.utils.desktop_install.subprocess.run"),
            patch("pathlib.Path.mkdir"),
        ):
            install(system_wide=True)

        dest_args = [str(c[0][1]) for c in mock_copy.call_args_list]
        assert any("/usr/local/share" in d for d in dest_args)


class TestUninstallLinux:
    """uninstall() on Linux removes the installed files."""

    def test_removes_desktop_file(self, tmp_path):
        app_dir = tmp_path / ".local" / "share" / "applications"
        app_dir.mkdir(parents=True)
        desktop_file = app_dir / "playchitect.desktop"
        desktop_file.touch()

        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            uninstall(system_wide=False)

        assert not desktop_file.exists()

    def test_removes_appdata_xml(self, tmp_path):
        metainfo_dir = tmp_path / ".local" / "share" / "metainfo"
        metainfo_dir.mkdir(parents=True)
        xml_file = metainfo_dir / f"{APP_ID}.appdata.xml"
        xml_file.touch()

        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            uninstall(system_wide=False)

        assert not xml_file.exists()

    def test_removes_png_icons(self, tmp_path):
        for size in PNG_SIZES:
            icon_dir = (
                tmp_path / ".local" / "share" / "icons" / "hicolor" / f"{size}x{size}" / "apps"
            )
            icon_dir.mkdir(parents=True)
            (icon_dir / f"{APP_ID}.png").touch()

        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            uninstall(system_wide=False)

        for size in PNG_SIZES:
            icon = (
                tmp_path
                / ".local"
                / "share"
                / "icons"
                / "hicolor"
                / f"{size}x{size}"
                / "apps"
                / f"{APP_ID}.png"
            )
            assert not icon.exists(), f"Icon {size}x{size} was not removed"

    def test_absent_files_are_skipped(self, tmp_path):
        """uninstall() should not raise when files are already absent."""
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value="Linux"),
            patch("playchitect.utils.desktop_install.Path.home", return_value=tmp_path),
        ):
            uninstall(system_wide=False)  # should not raise


@pytest.mark.parametrize("os_name", ["Darwin", "Windows"])
class TestNonLinuxPlatforms:
    """install/uninstall on non-Linux platforms should be a no-op."""

    def test_install_does_not_copy(self, os_name, tmp_path):
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value=os_name),
            patch("playchitect.utils.desktop_install.shutil.copy2") as mock_copy,
        ):
            install(system_wide=False)

        mock_copy.assert_not_called()

    def test_uninstall_does_not_remove(self, os_name, tmp_path):
        unlinkable = MagicMock()
        with (
            patch("playchitect.utils.desktop_install.platform.system", return_value=os_name),
            patch("pathlib.Path.unlink", unlinkable),
        ):
            uninstall(system_wide=False)

        unlinkable.assert_not_called()


class TestDesktopFileFormat:
    """Static validation of data/playchitect.desktop."""

    @pytest.fixture(scope="class")
    def desktop(self):
        # RawConfigParser avoids interpolation of desktop-file % placeholders
        parser = configparser.RawConfigParser(strict=False)
        parser.read(_REPO_ROOT / "data" / "playchitect.desktop")
        return parser

    def test_type_is_application(self, desktop):
        assert desktop.get("Desktop Entry", "type") == "Application"

    def test_terminal_is_false(self, desktop):
        assert desktop.get("Desktop Entry", "terminal") == "false"

    def test_icon_is_app_id(self, desktop):
        assert desktop.get("Desktop Entry", "icon") == APP_ID

    def test_name_present(self, desktop):
        assert desktop.get("Desktop Entry", "name") == "Playchitect"

    def test_exec_present(self, desktop):
        exec_val = desktop.get("Desktop Entry", "exec")
        assert exec_val.startswith("playchitect-gui")


class TestAppDataXML:
    """Static validation of AppStream metainfo XML."""

    @pytest.fixture(scope="class")
    def tree(self):
        path = _REPO_ROOT / "data" / f"{APP_ID}.appdata.xml"
        return ET.parse(path)

    def test_root_tag_is_component(self, tree):
        assert tree.getroot().tag == "component"

    def test_id_matches_app_id(self, tree):
        id_elem = tree.getroot().find("id")
        assert id_elem is not None
        assert id_elem.text == APP_ID

    def test_metadata_license_present(self, tree):
        elem = tree.getroot().find("metadata_license")
        assert elem is not None
        assert elem.text == "MIT"

    def test_project_license_is_gpl3(self, tree):
        elem = tree.getroot().find("project_license")
        assert elem is not None
        assert elem.text == "GPL-3.0"

    def test_name_present(self, tree):
        elem = tree.getroot().find("name")
        assert elem is not None and elem.text


class TestIconSourceExists:
    """Canary test: source JPEG must exist so generate_icons.py can run."""

    def test_logo_jpg_exists(self):
        assert (_REPO_ROOT / "img" / "playchitect_logo.jpg").exists()

    @pytest.mark.parametrize("size", PNG_SIZES)
    def test_generated_png_exists(self, size):
        path = (
            _REPO_ROOT / "data" / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{APP_ID}.png"
        )
        assert path.exists(), (
            f"PNG {size}x{size} not found — run: uv run python scripts/generate_icons.py"
        )

    def test_ico_exists(self):
        assert (_REPO_ROOT / "data" / "icons" / "playchitect.ico").exists()
