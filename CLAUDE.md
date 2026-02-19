# Claude Session Context - Playchitect Project

## Project Overview

**Name**: Playchitect
**Purpose**: Smart DJ Playlist Manager with Intelligent BPM Clustering
**Repository**: https://github.com/james-westwood/playchitect
**Location**: `/home/james/audio/playchitect/`
**Status**: Milestone 1 Complete (2026-02-19)

## What is Playchitect?

Playchitect transforms DJ playlist creation from rigid BPM-based grouping to intelligent multi-dimensional clustering. Instead of grouping tracks solely by tempo (e.g., "all 125 BPM tracks together"), it uses K-means analysis on BPM + 4 audio intensity features to create playlists that feel coherent—not just mathematically similar.

**Key Innovation**: A 125 BPM ambient intro sounds nothing like a 125 BPM hard techno track. Playchitect understands this by analyzing:
- BPM (tempo)
- Spectral brightness (treble content)
- High-frequency energy (>8kHz)
- RMS energy (loudness)
- Percussiveness (kick drum strength)

## Technology Stack

**Backend**:
- Python 3.11+ with native type hints
- librosa (audio intensity analysis)
- mutagen (metadata extraction)
- scikit-learn (K-means clustering)
- numpy, scipy

**Frontend** (Milestone 3):
- GTK4 + libadwaita (native GNOME)
- PyGObject bindings
- GNOME Sushi integration (spacebar preview)

**Development**:
- Package management: uv
- Testing: pytest with 27 unit tests
- Pre-commit hooks: black, flake8, mypy, pytest
- CI/CD: GitHub Actions (to be set up)

## Project Structure

```
playchitect/
├── playchitect/                    # Main package
│   ├── core/                       # Core business logic
│   │   ├── audio_scanner.py        # File discovery (92% coverage)
│   │   ├── metadata_extractor.py   # BPM extraction (61% coverage)
│   │   ├── intensity_analyzer.py   # TODO: Milestone 2
│   │   ├── clustering.py           # TODO: Milestone 2
│   │   ├── track_selector.py       # TODO: Milestone 2
│   │   ├── playlist_generator.py   # TODO: Milestone 2
│   │   └── export.py               # TODO: Milestone 4
│   ├── cli/                        # CLI interface
│   │   └── commands.py             # scan, info commands
│   ├── gui/                        # GTK4 interface (Milestone 3)
│   └── utils/                      # Config, logging
├── tests/
│   ├── unit/                       # 27 tests passing
│   ├── integration/                # TODO
│   └── gui/                        # TODO: Milestone 3
├── docs/
│   └── MILESTONE1_COMPLETE.md      # Milestone 1 summary
├── pyproject.toml                  # Project metadata
├── uv.lock                         # Locked dependencies
└── CLAUDE.md                       # This file
```

## Development Workflow

### Setup
```bash
cd /home/james/audio/playchitect
uv venv  # Already done
uv pip install -e ".[dev]"  # Already done
```

### Common Commands
```bash
# Run tests
uv run pytest -v
uv run pytest --cov=playchitect --cov-report=html

# Run CLI
uv run playchitect --help
uv run playchitect info /path/to/music
uv run playchitect scan /path/to/music --output /path/to/playlists

# Run pre-commit checks
uv run pre-commit run --all-files

# Format code
uv run black playchitect/ tests/

# Type check
uv run mypy playchitect/
```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/intensity-analysis

# Make changes, run tests
uv run pytest -v

# Commit (pre-commit hooks run automatically)
git commit -m "feat(analysis): implement librosa intensity analyzer"

# Push
git push -u origin feature/intensity-analysis
```

## Milestone Status

### ✅ Milestone 1: Foundation & Core Refactoring (Complete)
**Date Completed**: 2026-02-19
**Deliverables**:
- [x] GitHub repository with project structure
- [x] Extracted audio_scanner module (92% coverage, 13 tests)
- [x] Extracted metadata_extractor module (61% coverage, 14 tests)
- [x] Basic CLI interface (scan, info commands)
- [x] Pre-commit hooks configured and working
- [x] Documentation (README, CONTRIBUTING)
- [x] Package management with uv

**Key Files Created**:
- `playchitect/core/audio_scanner.py` - Recursive audio file discovery
- `playchitect/core/metadata_extractor.py` - BPM and metadata extraction with caching
- `playchitect/cli/commands.py` - Click-based CLI
- `tests/unit/test_audio_scanner.py` - 13 tests
- `tests/unit/test_metadata_extractor.py` - 14 tests

**Commits**: 2 commits on main branch, all pre-commit hooks passing

### 🚧 Milestone 2: Intelligent Analysis Engine (Next)
**Timeline**: 2 weeks
**Priority Tasks**:
1. **intensity_analyzer.py** - Most technically complex
   - Implement librosa spectral analysis
   - Calculate brightness, HF energy, RMS, percussiveness
   - Create hardness score: `0.4*brightness + 0.3*hf + 0.2*rms + 0.1*perc`
   - Add JSON caching system

2. **clustering.py** - K-means implementation
   - Create 5D feature vectors: [bpm, brightness, hf, rms, perc]
   - Implement elbow method for optimal K
   - Add cluster splitting for target playlist length

3. **track_selector.py** - First/last track intelligence
   - Intro detection (onset analysis)
   - Score tracks: low intensity + long intro (>30s) + no kick
   - Recommend top 5 first/last tracks per cluster

4. **playlist_generator.py** - Refactor from original script
   - Integrate clustering results
   - Generate M3U playlists
   - Smart track ordering within playlists

**Key Algorithms**:
- Hardness score: `0.4 * spectral_centroid + 0.3 * hf_energy + 0.2 * rms + 0.1 * percussiveness`
- Feature vector: `[normalized_bpm, brightness, hf_energy, rms, percussiveness]`
- Elbow method: Find optimal K by analyzing inertia vs. K curve

### 📋 Milestone 3: GTK4 GUI (Future)
- Main application window
- Track list widget
- Cluster visualization
- GNOME Sushi integration
- Preferences dialog

### 📋 Milestone 4: Export & Integration (Future)
- M3U playlist exporter
- CUE sheet generator
- Desktop file and icon
- File manager integration

### 📋 Milestone 5: Testing & QA (Future)
- Integration tests
- GUI smoke tests
- Performance benchmarking
- Pre-commit GUI checks

### 📋 Milestone 6: Packaging & Distribution (Future)
- Flatpak manifest for Flathub
- PyPI package publishing

## Origin Story

Playchitect originated from `/home/james/audio-management/scripts/create_random_playlists.py` (314 lines), which used rigid BPM range grouping. The new approach uses intelligent clustering to understand track character beyond just tempo.

**Original Script Location**: `/home/james/audio-management/scripts/create_random_playlists.py`

## Key Decisions

1. **License**: GPL-3.0 (copyleft, requires derivatives to be open source)
2. **Package Manager**: uv (fast, reproducible builds)
3. **Python Version**: 3.11+ (native type hints, modern features)
4. **GUI Framework**: GTK4 + libadwaita (native GNOME)
5. **Testing Strategy**: TDD with pytest, >85% coverage target
6. **Type Checking**: Strict mypy with native type hints
7. **Code Style**: Black with 100-char line length

## Important Notes

- **Working Directory**: `/home/james/audio/playchitect/`
- **Virtual Environment**: `.venv/` (managed by uv)
- **Pre-commit**: Always run before commits (installed via `uv run pre-commit install`)
- **Tests**: Must pass before merging to main
- **Coverage Target**: >85% for core modules

## Related Documentation

- **General Audio System**: `/home/james/audio-management/claude.md`
- **Original Script**: `/home/james/audio-management/scripts/create_random_playlists.py`
- **Project Plan**: Full implementation plan stored in session transcript

## Contact

**Developer**: James Westwood
**Machine**: Muddlehead (primary audio workstation)
**Platform**: Fedora 42 with GNOME

---

**Last Updated**: 2026-02-19
**Current Milestone**: 1 Complete, Starting Milestone 2
**Next Session**: Implement librosa intensity analyzer
