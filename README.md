<p align="center">
  <img src="https://raw.githubusercontent.com/james-westwood/playchitect/main/img/playchitect_logo.jpg" alt="Playchitect logo" width="180">
</p>

# Playchitect

[![CI](https://github.com/james-westwood/playchitect/actions/workflows/ci.yml/badge.svg)](https://github.com/james-westwood/playchitect/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/james-westwood/playchitect/graph/badge.svg)](https://codecov.io/gh/james-westwood/playchitect)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/Package%20Manager-uv-orange.svg)](https://github.com/astral-sh/uv)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-blue.svg)](https://github.com/astral-sh/ruff)
[![Type Checking: Ty](https://img.shields.io/badge/Type%20Checking-Ty-purple.svg)](https://github.com/astral-sh/ty)

**Smart DJ Playlist Manager with Intelligent BPM Clustering**

Playchitect transforms DJ playlist creation from rigid BPM-based grouping to intelligent multi-dimensional clustering. Using K-means analysis of BPM, spectral brightness, energy, and percussiveness, it creates coherent playlists that feel right—not just mathematically similar.

> **Where this is heading:** Playchitect is being rebuilt around a learned transition model — co-occurrence graph embeddings mined from real DJ sets, audio embeddings (essentia discogs-effnet), and a personal taste layer trained on crossfaded A/B/C/D mix judgements, all validated against an eval harness with a BPM-window baseline. For the purpose, approach, and ML training methods, read **[docs/planning/ml-playlist-generator-plan.md](docs/planning/ml-playlist-generator-plan.md)**. The task-level execution list is `prd.json`.

## Key Features

- **Intelligent Clustering**: K-means analysis on BPM + 7 audio intensity features
- **Semantic Embeddings**: Optional MusiCNN neural embeddings for genre-aware clustering
- **Smart Track Selection**: Recommends ideal first tracks (long intros, ambient) and closers (high energy or smooth outros)
- **Audio Intensity Analysis**: Librosa-powered spectral analysis for track "hardness" scoring
- **Adaptive Playlist Splitting**: Automatically divides clusters to meet target playlist lengths
- **Native GNOME GUI**: GTK4 + libadwaita interface with cluster visualisation and track preview
- **CUE Sheet Export**: Frame-accurate CUE sheets alongside M3U playlists
- **Desktop Integration**: `.desktop` file, AppStream metainfo, hicolor icon theme

## Requirements

- **Python**: 3.13+
- **GTK4 GUI** (optional): requires system-level `python3-gobject` — cannot be pip-installed

## Installation

### PyPI (CLI)

```bash
uv tool install playchitect   # recommended
# or: pip install playchitect
```

> **GUI note**: the GTK4 interface requires `python3-gobject` from your OS package manager
> and cannot be installed via pip. See [From source](#from-source-development) below.

### Flatpak (bundle)

Download `playchitect.flatpak` from the [latest GitHub Release](https://github.com/james-westwood/playchitect/releases) and install:

```bash
flatpak install playchitect.flatpak
flatpak run com.github.jameswestwood.Playchitect
```

> Requires the [GNOME Platform runtime](https://flathub.org/apps/org.gnome.Platform) (version 49).
> Install it via: `flatpak install flathub org.gnome.Platform//49`

### From source (development)

```bash
git clone https://github.com/james-westwood/playchitect
cd playchitect

# CLI only
uv venv --python /usr/bin/python3 --system-site-packages
uv pip install -e ".[dev]"
uv run playchitect --help

# GUI — requires python3-gobject from the OS package manager first:
#   Fedora:  sudo dnf install python3-gobject gtk4
#   Ubuntu:  sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0
# The venv must use --system-site-packages (already set above)
uv run playchitect-gui
```

> **Why system-site-packages?** PyGObject links against system GTK4 libraries and is
> distributed as an OS package (`python3-gobject`). It cannot be built from PyPI without
> Cairo development headers. Using `--system-site-packages` lets the venv find it without
> needing to compile anything. The Flatpak release bundles everything and avoids this entirely.

## Quick Start

### CLI
```bash
# Scan a music directory and create playlists of ~25 tracks each
playchitect scan ~/Music/Techno --output ~/Playlists --target-tracks 25

# Target playlist duration instead of track count
playchitect scan ~/Music/House --output ~/Playlists --target-duration 90

# Preview what would be created without writing files
playchitect scan ~/Music --dry-run --target-tracks 20

# Show information about a music directory
playchitect info ~/Music
playchitect info ~/Music --format json
```

### GUI
```bash
uv run playchitect-gui
```

1. Open Folder → select music directory
2. Wait for analysis (BPM extraction)
3. View generated clusters
4. Export → M3U playlists

## How It Works

### Traditional BPM Grouping (Old Approach)
```
120-130 BPM → All tracks lumped together
Problem: A 125 BPM ambient intro sounds nothing like a 125 BPM hard techno track
```

### Intelligent Clustering (Playchitect)
```
K-means on 8D feature space:
[bpm, spectral_centroid, hf_energy, rms_energy, percussiveness,
 sub_bass_energy, kick_energy, bass_harmonics]

Result: Tracks grouped by both tempo AND intensity/character
```

### Hardness Score Calculation
```python
hardness = (
    0.4 * spectral_centroid +  # Brightness (treble content)
    0.3 * hf_energy_ratio +     # High-frequency energy (>8kHz)
    0.2 * rms_energy +          # Loudness
    0.1 * percussiveness        # Kick drum strength
)
```

## GUI Design

The native GNOME desktop interface (Milestone 3) uses GTK4 + libadwaita. The interactive HTML wireframe is at [`docs/wireframe.html`](docs/wireframe.html) — open it locally in a browser to click between views.

### Library View
```
┌─ Playchitect ───────────────────────────────────────────────────────────────┐
│  [≡]           Playchitect               [⚙ Settings]  [? Help]            │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Sidebar ──────────────┐  ┌─ Track Library ────────────────────────────┐ │
│ │  ● Library             │  │  [📂 Open Folder]  [🔍 Scan]  [____] 🔎    │ │
│ │  ○ Analysis            │  │                                             │ │
│ │  ○ Playlists           │  │  Title          Artist    BPM   Duration    │ │
│ │                        │  │  ──────────────────────────────────────     │ │
│ │  📁 Music Folders      │  │  ★ Dark Matter   Surgeon   138   6:42      │ │
│ │  ┌────────────────┐    │  │    Phase IV      Dax J     132   7:15      │ │
│ │  │ ~/music        │    │  │    Redline       DVS1      140   8:03      │ │
│ │  │ /media/usb     │    │  │    Obsidian      Truncate  136   9:12      │ │
│ │  └────────────────┘    │  │                                             │ │
│ │  [+ Add] [– Remove]    │  │  ♪ Dark Matter — Surgeon                   │ │
│ │                        │  │  [◀◀] [▶ Play] [▶▶]  ══════●══════        │ │
│ │  Tracks found:   847   │  └─────────────────────────────────────────────┘ │
│ │  Analysed:       312   │                                                   │
│ └────────────────────────┘                                                   │
│  ░░░░░░░░░░░░░░████████████████  Scanning… 312 / 847 tracks                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Analysis View
```
┌─ Playchitect ───────────────────────────────────────────────────────────────┐
│ ┌─ Sidebar ──────────────┐  ┌─ Intensity Analysis ───────────────────────┐ │
│ │  Feature Weights       │  │  [▶ Analyse All]  [▶ Analyse New]          │ │
│ │                        │  │  Show: [All tracks ▼]   Sort: [BPM ▼]      │ │
│ │  BPM        [══●══] 1.0│  │                                             │ │
│ │  RMS Energy [═●═══] 0.8│  │  Title       BPM  RMS   Bright  Perc  Bass │ │
│ │  Brightness [●═════] 0.4│  │  ──────────────────────────────────────── │ │
│ │  Percussive [═════●] 0.9│  │  Dark Matter 138  ████  ██░░    ████  ███░ │ │
│ │  Sub-bass   [═══●═] 0.7│  │  Phase IV    132  ███░  ███░    ███░  ████ │ │
│ │  Kick       [═════●] 0.8│  │  Redline     140  █████ █░░░    █████ ██░░ │ │
│ │  Onset str. [══●══] 0.5│  │  Untitled 03  —   —     —  Not analysed    │ │
│ │                        │  │  ─────────────────────────────────────────  │ │
│ │  [Reset to defaults]   │  │  Dark Matter — Surgeon                      │ │
│ │                        │  │  BPM ████████░░ 138  Kick  █████████░ 0.81 │ │
│ └────────────────────────┘  │  RMS ███████░░░ 0.72 Harm  ████░░░░░░ 0.38 │ │
│                              │  Bri ███░░░░░░░ 0.31 Perc  █████████░ 0.91 │ │
│                              └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Playlists View
```
┌─ Playchitect ───────────────────────────────────────────────────────────────┐
│ ┌─ Sidebar ──────────────┐  ┌─ Playlists ────────────────────────────────┐ │
│ │  ● Cluster 1  Hard     │  │  Clusters: (± 6)  Length: (± 12)           │ │
│ │  ● Cluster 2  Peak     │  │  Method: [K-means ▼]  [Auto-detect K]      │ │
│ │  ● Cluster 3  Dark     │  │  ────────────────────────  [▶▶ Generate]   │ │
│ │  ● Cluster 4  Atmo     │  │                                             │ │
│ │  ● Cluster 5  Acid     │  │  ┌─ Cluster 1 — Hard Techno (138 BPM) ──┐  │ │
│ │  ● Cluster 6  Deep     │  │  │  [✓] Dark Matter  Surgeon  138  6:42  │  │ │
│ │                        │  │  │  [✓] Redline      DVS1     140  8:03  │  │ │
│ │  Export All            │  │  │  [✓] Headbanger   Perc     139  7:28  │  │ │
│ │  Format: [M3U ▼]       │  │  │  Avg BPM: 138.5  ·  12 tracks         │  │ │
│ │  Dest: [~/music/…] 📁  │  │  │  [Export as M3U ▼]  [⬇ Export]        │  │ │
│ │  [⬇ Export All]        │  │  └──────────────────────────────────────┘  │ │
│ └────────────────────────┘  └─────────────────────────────────────────────┘ │
│  6 playlists ready · 847 tracks · Last generated: just now                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Interactive version**: Open [`docs/wireframe.html`](docs/wireframe.html) in a browser for the full clickable prototype with all controls, feature bars, and the preferences window.

## Development Status

| Milestone | Description | Status |
|---|---|---|
| 1 | Foundation & Core Refactoring | ✅ Complete |
| 2 | Intelligent Analysis Engine | ✅ Complete |
| 3 | GTK4 GUI | ✅ Complete |
| 4 | Export & Integration | ✅ Complete |
| 5 | Testing & Quality Assurance | ✅ Complete |
| 6 | Packaging & Distribution | 🚧 In progress |

**Current phase**: Milestone 6 — Flatpak and PyPI packaging.

## Technology Stack

- **Python**: 3.13+
- **Audio Analysis**: librosa, mutagen
- **Clustering**: scikit-learn (K-means)
- **GUI**: GTK4, libadwaita, PyGObject (system package)
- **Testing**: pytest, ruff, ty
- **Packaging**: Flatpak, PyPI

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.

## Credits & AI Orchestration

Playchitect is orchestrated and architected by **James Westwood**, a senior Data Scientist and Developer.

The development process leveraged advanced AI collaboration:
- **Claude** (via Claude CLI): Responsible for implementing the majority of the codebase, core algorithms, and UI logic.
- **Gemini** (via Gemini CLI): Managed the majority of pull requests, documentation research, and code review processes.

This project stands as a testament to human-AI pair programming at scale.
