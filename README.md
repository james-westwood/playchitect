# Playchitect

[![Build Status](https://img.shields.io/badge/Build-WIP-lightgrey.svg)](https://github.com/james-westwood/playchitect/actions)
[![Coverage](https://img.shields.io/badge/Coverage-85%25%2B-brightgreen.svg)](https://codecov.io/gh/james-westwood/playchitect)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/Package%20Manager-uv-orange.svg)](https://github.com/astral-sh/uv)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)
[![Type Checking: Mypy](https://img.shields.io/badge/Type%20Checking-Mypy-blue.svg)](https://mypy-lang.org/)

**Smart DJ Playlist Manager with Intelligent BPM Clustering**

Playchitect transforms DJ playlist creation from rigid BPM-based grouping to intelligent multi-dimensional clustering. Using K-means analysis of BPM, spectral brightness, energy, and percussiveness, it creates coherent playlists that feel right—not just mathematically similar.

## Key Features

- **Intelligent Clustering**: K-means analysis on BPM + 4 audio intensity features (spectral centroid, high-frequency energy, RMS, percussiveness)
- **Smart Track Selection**: Recommends ideal first tracks (long intros, ambient) and closers (high energy or smooth outros)
- **Audio Intensity Analysis**: Librosa-powered spectral analysis for track "hardness" scoring
- **Adaptive Playlist Splitting**: Automatically divides clusters to meet target playlist lengths
- **Native GNOME GUI**: GTK4 + libadwaita interface with GNOME Sushi preview integration
- **Flexible Export**: M3U and CUE sheet generation

## Installation

### Flatpak (Recommended)
```bash
flatpak install flathub com.github.jameswestwood.Playchitect
```

### PyPI
```bash
pip install playchitect

# CLI usage
playchitect scan ~/Music --output ~/Playlists --target-length 25

# GUI usage
playchitect-gui
```

## Quick Start

### CLI
```bash
# Analyze music directory and create intelligent playlists
playchitect scan ~/Music/Techno --output ~/Playlists/Techno --target-length 25

# Use custom cluster count
playchitect scan ~/Music/House --clusters 8 --target-length 20
```

### GUI
Launch the GUI with `playchitect-gui`:
1. File → Open Folder → Select music directory
2. Wait for analysis (BPM extraction + intensity analysis)
3. View clusters in the cluster panel
4. Select tracks, preview with spacebar (GNOME Sushi)
5. Export → M3U or CUE sheets

## How It Works

### Traditional BPM Grouping (Old Approach)
```
120-130 BPM → All tracks lumped together
Problem: A 125 BPM ambient intro sounds nothing like a 125 BPM hard techno track
```

### Intelligent Clustering (Playchitect)
```
K-means on 5D feature space:
[normalized_bpm, spectral_brightness, high_freq_energy, rms_energy, percussiveness]

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

**Current Phase**: Milestone 2 - Intelligent Analysis Engine
**Timeline**: 10-week MVP development

See [ROADMAP.md](docs/ROADMAP.md) for detailed milestones.

## Technology Stack

- **Audio Analysis**: librosa, mutagen
- **Clustering**: scikit-learn (K-means)
- **GUI**: GTK4, libadwaita, PyGObject
- **Testing**: pytest, pytest-gtk
- **Packaging**: Flatpak, PyPI

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

GPL-3.0 (to be confirmed by maintainer)

## Credits

Built by James Westwood. Extends functionality from original `create_random_playlists.py` script.
