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

- **Tracks shorter than ~3 s could cache a NaN embedding** — MusiCNN needs at least
  2.9600625 s of audio (47,361 samples at 16 kHz) before it emits a single embedding
  frame. Below that it returned zero frames, and mean-pooling zero frames collapsed to
  NaN rather than raising, so `EmbeddingExtractor.analyze()` returned a NaN vector and
  wrote it straight to the embedding cache. Any library containing a short interlude,
  intro or skit was therefore silently poisoning its own cache. `analyze()` now raises
  `AudioTooShortForEmbeddingError` before pooling and before any cache write, matching
  the guard `analyze_discogs_effnet()` already had. The threshold is recorded as
  `_MUSICNN_MIN_AUDIO_SECONDS`, measured by binary search against the installed
  `msd-musicnn-1.pb` rather than derived from the nominal patch length; it differs from
  discogs-effnet's 2.0320625 s, so the existing effnet constant could not be reused.
  Note that cache rows written before this fix are not re-validated, so any NaN entries
  already persisted remain until the cache is rebuilt.
- **`scan --use-embeddings` failed with a misleading error when nothing could be
  embedded** — a run in which no track produced an embedding fell through to the
  clusterer and reported `Error: Clustering failed`, which sends the reader looking in
  the wrong place. The embedding stage now always reports `Embedded N of M tracks`, and
  on zero successes it fails there with an aggregate error naming the dominant cause,
  the counts, the minimum track length and the option of re-running without
  `--use-embeddings`. Partial failures are unchanged: the run proceeds on whatever did
  embed. Per-track failures beyond the first three drop from `WARNING` to `DEBUG` so a
  library-wide failure does not bury the summary.
- **Embeddings benchmark failed whenever essentia was actually installed** —
  `test_playchitect_scan_with_embeddings_dry_run_cli` drove `scan --use-embeddings` over
  the shared `synthetic_library` fixture, whose 0.5 s clips are far below the MusiCNN
  minimum, so the run always exited 1. The benchmark now builds its own 4.0 s library
  from a dedicated fixture. The shared fixture stays at 0.5 s deliberately: lengthening
  it pushes `test_intensity_analyzer_analyze` from 0.087 s to 0.349 s against a 0.150 s
  threshold and makes `test_metadata_extractor_extract_batch` flaky. The failure was
  pre-existing rather than a regression, and invisible in CI because CI does not install
  the `embeddings` extra.

- **`scripts/embed_library.py` could exhaust system memory on large libraries** — the
  batch embedding script rebuilt its entire TensorFlow/essentia model (~924 MB) for
  every single track instead of once per run, growing resident memory by roughly 1 GB
  per track. On a 1,730-track library this froze the workstation and crashed other
  applications after only 7 tracks had been written. The model is now constructed
  exactly once per run and reused for every track. The script also gained a
  configurable `--memory-ceiling-mb` safety net (`MemoryCeilingExceededError`) that
  aborts the run before a future leak can repeat the incident, and its progress log now
  reports current resident memory so growth is visible during a long run.
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
