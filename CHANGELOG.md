# Changelog

All notable changes to Playchitect are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-02-22

First stable release of Playchitect.

### Added

#### Core analysis engine
- **Intelligent clustering** — K-means on an 8-dimensional feature space: BPM,
  spectral centroid, high-frequency energy, RMS energy, percussiveness, sub-bass
  energy, kick energy, and bass harmonics
- **Librosa intensity analyser** — full STFT-once pipeline with JSON caching and
  MD5 hash validation; 92% test coverage
- **Genre-aware multi-clustering** — EWKM per-cluster refinement and genre-specific
  PCA + EWKM feature weighting
- **Semantic embeddings** — optional MusiCNN neural embeddings for genre-aware
  clustering (`[embeddings]` extra)
- **Smart track selector** — scores tracks for opener/closer suitability (long
  intros, low intensity, no kick); supports user overrides persisted in config
- **Robust BPM calculation** — librosa fallback when tags are missing or suspicious
  (non-whole numbers, genre mismatches); `recalculate()` method to force a cache bypass
- **Adaptive playlist splitting** — automatically divides clusters to meet a target
  track count or duration

#### GTK4 desktop application
- Native GNOME interface built with GTK4 + libadwaita
- Split-pane main window with scan, analyse, and export controls
- Track list widget using `Gtk.ColumnView` with sorting and column visibility
- Cluster visualisation panel
- Spacebar preview via GNOME Sushi / xdg-open

#### Export & OS integration
- M3U playlist export
- CUE sheet generator with frame-accurate timing (75 fps standard)
- Freedesktop `.desktop` file with MIME associations for M3U and CUE files
- AppStream metainfo (`com.github.jameswestwood.Playchitect.appdata.xml`)
- Hicolor icon theme — 9 PNG sizes (16 px → 512 px) generated from source artwork
- `playchitect-install-desktop` entry point for per-user or system-wide install

#### CLI
- `playchitect scan <dir>` — analyse and generate playlists
- `playchitect info <dir>` — show library statistics
- `--target-tracks`, `--target-duration`, `--dry-run` flags
- `--use-embeddings`, `--cluster-mode`, feature-weight overrides

#### Packaging & distribution
- PyPI package — `pip install playchitect` / `uv tool install playchitect`
- OIDC trusted publishing via GitHub Actions (no long-lived tokens)
- Self-hosted Flatpak bundle — built by CI and attached to each GitHub Release
- `playchitect-gui` and `playchitect-install-desktop` entry points

#### Developer tooling
- Pre-commit hooks: ruff, ty, pytest-unit, cli-smoke-test, GUI smoke tests
- GitHub Actions CI: lint + type-check + unit tests (Ubuntu Python 3.13, Fedora 41
  container), extended CLI integration tests, codecov coverage reporting
- `pytest-benchmark` suite with `synthetic_library` factory fixture; regression
  alerts via `--benchmark-compare`
- Gemini 2.5 Pro automated PR review (`scripts/review_pr.sh`)

### Configuration

User settings live at `~/.config/playchitect/config.yaml`. The intensity analysis
cache defaults to `~/.cache/playchitect/intensity/` and can be overridden via the
`PLAYCHITECT_CACHE_DIR` environment variable or the `cache_dir` config key.

### Requirements

- Python 3.13+
- GTK4 GUI requires `python3-gobject` from the OS package manager — not installable
  via pip. See the README for per-distro instructions.

### Known limitations

- COPR (Fedora DNF) and Flathub packages are not yet available; both are planned
  post-1.0.0.
- The `[embeddings]` extra requires `essentia-tensorflow`, which has its own system
  dependencies; it is not installed by default.
- `.icns` macOS icon generation is documented but not automated (requires macOS
  `iconutil`).

---

## [0.1.0] — 2026-02-19

Initial development release. Established project structure, core audio scanner,
metadata extractor with BPM caching, basic BPM-only clustering, and Click CLI.

[1.0.0]: https://github.com/james-westwood/playchitect/releases/tag/v1.0.0
[0.1.0]: https://github.com/james-westwood/playchitect/releases/tag/v0.1.0
