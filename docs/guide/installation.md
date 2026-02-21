# Installation

Playchitect offers three installation methods depending on your needs. The **Flatpak bundle** is recommended for most users wanting the GUI, while **PyPI** is great for CLI usage.

## Method 1: PyPI (CLI Only)

For command-line usage, the recommended method is using `uv tool install` (or `pipx`):

```bash
uv tool install playchitect
# or: pip install playchitect
```

::: warning
**GUI Note**: The GTK4 interface requires system-level `python3-gobject` libraries which cannot be installed via `pip`. If you need the GUI, use the **Flatpak** or the **From Source** method.
:::

## Method 2: Flatpak Bundle (GUI, Recommended)

The Flatpak bundle includes all dependencies, including GTK4 and PyGObject, in a sandboxed environment.

1.  Download `playchitect.flatpak` from the [GitHub Releases page](https://github.com/james-westwood/playchitect/releases).
2.  Install the bundle:

```bash
flatpak install playchitect.flatpak
```

3.  Run the application:

```bash
flatpak run com.github.jameswestwood.Playchitect
```

::: tip
This requires the [GNOME Platform runtime](https://flathub.org/apps/org.gnome.Platform) (version 49). If missing, install it via:
`flatpak install flathub org.gnome.Platform//49`
:::

## Method 3: From Source (Development)

This method is for developers or users on Linux distributions who want the latest code.

### Prerequisites

You need Python 3.13+ and the `uv` package manager.
For the GUI, you **must** install `python3-gobject` via your system package manager first:

*   **Fedora**: `sudo dnf install python3-gobject gtk4`
*   **Ubuntu**: `sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0`
*   **Arch**: `sudo pacman -S python-gobject gtk4`

### Installation Steps

1.  Clone the repository:

    ```bash
    git clone https://github.com/james-westwood/playchitect
    cd playchitect
    ```

2.  Create a virtual environment with access to system packages:

    ```bash
    # The --system-site-packages flag is CRITICAL for GTK4/PyGObject access
    uv venv --python /usr/bin/python3 --system-site-packages
    ```

3.  Install the project in editable mode:

    ```bash
    uv pip install -e ".[dev]"
    ```

4.  Run the CLI:

    ```bash
    uv run playchitect --help
    ```

5.  Run the GUI:

    ```bash
    uv run playchitect-gui
    ```

::: info Why system-site-packages?
PyGObject links against system GTK4 libraries (C code) and is distributed as an OS package (`python3-gobject`). It cannot be built easily from PyPI without specific development headers (Cairo, GObject Introspection). Using `--system-site-packages` allows the virtual environment to "see" the system-installed PyGObject while keeping other Python dependencies isolated.
:::
