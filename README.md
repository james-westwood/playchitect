# Playchitect

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

## Development Status

**Current Phase**: Milestone 1 - Foundation & Core Refactoring
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
