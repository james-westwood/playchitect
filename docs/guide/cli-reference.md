# CLI Reference

Playchitect provides two primary commands: `scan` for analysis/playlist generation and `info` for library inspection.

## `playchitect scan`

Scan a music directory and create intelligent playlists based on audio features. Multi-dimensional clustering (BPM + 7 intensity features) is used by default, with intensity results cached for fast re-scans. Use `--fast` to select the old BPM-only behaviour and skip intensity analysis. Note that `--fast` cannot be combined with `--use-embeddings` or `--cluster-mode per-genre|mixed-genre`.

When auto-selected clustering fails to separate a library (where the dominant cluster holds >= 70% of tracks with auto-K <= 2 across 50+ tracks), scan prints a warning suggesting `--use-embeddings` or `--genre` hints for better separation.

When intensity analysis runs by default, playlists are named after each cluster's musical character (e.g. 'Dark Journey'). Playlists sharing the same character receive a BPM-range suffix for disambiguation, and export filenames are automatically sanitized (lowercase with underscores). Passing `--fast` keeps the legacy `Playlist N [bpm-range]` naming.

```bash
playchitect scan [OPTIONS] [MUSIC_PATH]
```

### Arguments

*   `MUSIC_PATH`: Directory containing audio files to analyze. Defaults to current directory if omitted.

### Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-o, --output` | `PATH` | (same as input) | Output directory for generated playlists. |
| `-t, --target-tracks` | `INTEGER` | `None` | Target number of tracks per playlist. |
| `-d, --target-duration` | `FLOAT` | `None` | Target duration per playlist in minutes. |
| `-n, --playlist-name` | `TEXT` | `Playlist` | Base name for playlists (e.g., "MyMix 1", "MyMix 2"). |
| `--dry-run` | `FLAG` | `False` | Analyze and cluster but don't create any files on disk. |
| `--use-test-path` | `FLAG` | `False` | Use test music path from config (for development). |
| `--first-override` | `PATH` | `None` | Override the auto-selected opener track. |
| `--last-override` | `PATH` | `None` | Override the auto-selected closer track. |
| `--save-overrides` | `FLAG` | `False` | Persist `--first-override` / `--last-override` to config for future runs. |
| `--use-embeddings` | `FLAG` | `False` | Use MusiCNN semantic embeddings for richer clustering (requires `essentia-tensorflow`). |
| `--model-path` | `PATH` | `None` | Path to `msd-musicnn-1.pb` (auto-downloaded if absent). |
| `--fast` | `FLAG` | `False` | BPM-only clustering; skips intensity analysis for speed. |
| `--cue` | `FLAG` | `False` | Also write a CUE sheet alongside each M3U playlist. |
| `--cluster-mode` | `ENUM` | `single-genre` | Clustering mode: `single-genre` (default), `per-genre`, or `mixed-genre`. |
| `--genre-map` | `PATH` | `None` | YAML file with manual genre assignments (`manual_assignments: {path: genre}`). |
| `--sequence-mode` | `ENUM` | `fixed` | Track sequencing mode: `ramp` (intensity build) or `fixed` (no change, default). |
| `--help` | `FLAG` | - | Show this message and exit. |

## `playchitect info`

Display information about a music directory without running full analysis.

```bash
playchitect info [OPTIONS] [MUSIC_PATH]
```

### Arguments

*   `MUSIC_PATH`: Directory to analyze. Defaults to current directory if omitted.

### Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-f, --format` | `ENUM` | `text` | Output format: `text` (human readable) or `json`. |
| `--help` | `FLAG` | - | Show this message and exit. |
