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

## GUI Architecture

The GUI is built with **GTK4** and **Libadwaita**. It uses `PyGObject` for Python bindings.

*   **Model-View-Controller**: While not strictly enforced, the code separates data (core logic) from presentation (widgets).
*   **Async Operations**: Long-running tasks (scanning/analysis) should run in a separate thread to keep the UI responsive.
*   **Signals**: Custom GObject signals are used for communication between components.
