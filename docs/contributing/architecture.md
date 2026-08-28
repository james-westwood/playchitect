# Architecture

Playchitect follows a modular architecture, separating core logic from the user interface and CLI.

## Directory Structure

*   `playchitect/` - Main package
    *   `core/` - The analysis and clustering pipeline
        *   `audio_scanner.py` - File discovery and metadata extraction
        *   `metadata_extractor.py` - BPM extraction (librosa/mutagen)
        *   `intensity_analyzer.py` - Audio feature extraction (spectral centroid, etc.)
        *   `clustering.py` - K-means clustering logic
        *   `sequencer.py` - Playlist ordering (ramp mode)
        *   `export.py` - M3U/CUE generation
    *   `cli/` - Click-based command-line interface
        *   `commands.py` - Entry points for `scan` and `info`
    *   `gui/` - GTK4 application
        *   `app.py` - Main application class
        *   `windows/` - Window definitions
        *   `widgets/` - Custom widgets
    *   `utils/` - Shared utilities
        *   `config.py` - Configuration handling
        *   `desktop_install.py` - Desktop file installation
*   `tests/` - Test suite
    *   `unit/` - Unit tests for core logic
    *   `integration/` - Integration tests (CLI smoke tests)
    *   `gui/` - GUI tests (headless GTK mocking)
    *   `benchmarks/` - Performance benchmarks

## Data Flow

The core pipeline processes audio files in this order:

```
AudioScanner (File Discovery)
      ↓
MetadataExtractor (BPM/Key/Duration)
      ↓
IntensityAnalyzer (Audio Features: Centroid, RMS, etc.)
      ↓
Clusterer (K-means Grouping)
      ↓
Sequencer (Ordering within clusters)
      ↓
Exporter (M3U/CUE Generation)
```

## Semantic Embeddings (optional)

The `embeddings` extra (`playchitect[embeddings]`) adds `essentia-tensorflow`, used by
`playchitect/core/embedding_extractor.py` for MusiCNN/MIREX mood analysis and by the
`discogs-effnet` inference path feeding `playchitect/core/embedding_cache.py` and
`scripts/embed_library.py`. It is pinned to an exact dev-line build,
`essentia-tensorflow==2.1b6.dev1389` — the last release with cp313 wheels (upstream has
since moved on to cp314) — and deliberately kept out of the core `[project.dependencies]`:
that build ships no linux-aarch64 wheel for any CPython version, and it bundles a
TensorFlow C library that core GUI/CLI users should never be forced to carry.

Each auto-downloaded model file (`msd-musicnn-1.pb`, `moods_mirex-msd-musicnn-1.pb`,
`discogs-effnet-bs64-1.pb`) is verified against a pinned sha256 digest on download,
since essentia.upf.edu publishes no checksums of its own — see the module docstrings in
`embedding_extractor.py` for how that trust-on-first-use model works and what it does
and doesn't protect against.

### Sidecar fallback: running the batch ETL on a different interpreter

`scripts/embed_library.py` and the main GUI/CLI package are decoupled by design: the
embedding cache is a parquet file keyed by track content hash
(`playchitect/core/embedding_cache.py`), not by which process or interpreter produced
it. If the pinned `essentia-tensorflow` wheel ever misbehaves under the project's
primary Python 3.13 environment, `scripts/embed_library.py` can be run standalone from a
separate virtualenv with no changes to the main package:

```bash
uv venv --python 3.12 .venv-embed
source .venv-embed/bin/activate
uv pip install 'essentia-tensorflow==2.1b6.dev1389' pyarrow numpy click
python scripts/embed_library.py /path/to/music/library --cache-path data/embeddings.parquet
```

A cp312 manylinux x86_64 wheel exists for the `2.1b6.dev1389` build, so this sidecar
venv can run the batch embedding job independently of the main `.venv`. The GUI/CLI
consumer side simply reads whatever `data/embeddings.parquet` it finds — it never needs
to import `essentia` itself, so producer and consumer can safely run on different
interpreters.

## GUI Architecture

The GUI is built with **GTK4** and **Libadwaita**. It uses `PyGObject` for Python bindings.

*   **Model-View-Controller**: While not strictly enforced, the code separates data (core logic) from presentation (widgets).
*   **Async Operations**: Long-running tasks (scanning/analysis) should run in a separate thread to keep the UI responsive.
*   **Signals**: Custom GObject signals are used for communication between components.
