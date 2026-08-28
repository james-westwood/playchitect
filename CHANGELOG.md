# Changelog

All notable changes to Playchitect are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added

- **Semantic embedding cache** — content-addressed parquet-backed cache for neural audio
  embeddings (via `EmbeddingCache`). Entries are keyed by content hash — sha256 of the
  first 1 MB of the file plus its total size — so a renamed or moved track retains its
  cached embedding without recomputation. Re-computing the same content upserts rather
  than appends, and cache files can be persisted and shared across library reorganisations.
  Also includes `fit_and_save_pca()` to reduce raw embedding dimensionality (e.g., 1280-D
  discogs-effnet) to a smaller whitened PCA space (default 64 components) for downstream
  clustering and feature weighting.
- **Library embedding ETL script** — `scripts/embed_library.py` idempotently walks a
  library, skips already-cached tracks, and embeds the rest. Useful for pre-computing
  embeddings for a library once, then incrementally updating as new tracks arrive. Also
  adds optional `discogs-effnet` embedding model (1280-D vectors) alongside the existing
  MusiCNN model, auto-downloaded into `~/.cache/playchitect/models/` on first use. The
  `[embeddings]` extra is optional and not installed by default; the cache and ETL
  infrastructure work out of the box, but computing embeddings requires installing it.
- **discogs-effnet inference path** — `EmbeddingExtractor.analyze_discogs_effnet()` runs
  the discogs-effnet model and returns the raw mean-pooled 1280-D vector. Wired into
  `scripts/embed_library.py`'s `_real_embed_fn`, which previously called the MusiCNN
  `analyze()` pathway despite the cache being labelled `discogs-effnet-1` — the batch
  ETL now writes vectors that actually match the model_version recorded alongside them,
  and refuses to write a row if the produced dimensionality ever disagrees with that
  label.
- **Model download integrity pins** — `EmbeddingExtractor._download_model()` now
  verifies each downloaded model's sha256 against a pinned digest (`_MSD_MUSICNN_SHA256`,
  `_MIREX_MOODS_SHA256`, `_DISCOGS_EFFNET_SHA256`), raising `ModelIntegrityError` and
  deleting the file on mismatch. essentia.upf.edu publishes no checksums, so this is
  trust-on-first-use: it protects against a model file silently changing under us on a
  later download, not the very first one.
- **Pre-batch embedding smoke check** — `run_embedding_smoke_check()` runs a small audio
  fixture through discogs-effnet twice (two independent `EmbeddingExtractor` instances)
  and validates dimensionality, finiteness, L2-norm sanity, and cross-instance
  reproducibility before a full library batch run, raising `EmbeddingSmokeCheckError` on
  failure. Intended as a pre-flight gate for `scripts/embed_library.py`.
- **Exact essentia-tensorflow pin** — the `embeddings` extra now pins
  `essentia-tensorflow==2.1b6.dev1389` exactly, the last dev-line build with cp313
  wheels, rather than an unpinned `essentia-tensorflow`.

### Fixed

- **Silent track loss in large libraries** — when clustering libraries with 80+ tracks
  (where EWKM per-cluster weight refinement engages), a bug in `_deduplicate_clusters`
  filtered out boundary tracks that EWKM had moved without ever reassigning them,
  silently removing them from playlists with no warning. The number of affected tracks
  depends on library feature geometry; one test library lost 7 of 405 tracks. Deduplication
  now conservatively reassigns every track to its nearest centroid, guaranteeing the
  output contains exactly the same tracks as the input. Cluster statistics
  (BPM mean/std, duration, feature means) are recomputed for any cluster that loses
  or gains members during reassignment.

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
