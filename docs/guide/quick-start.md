# Quick Start

Get started analyzing your music library with Playchitect.

## Command Line Interface (CLI)

The CLI is powerful and scriptable. It allows you to analyze directories, preview cluster output, and generate playlists with full control.

### 1. Dry Run / Preview

Before creating any files, use `--dry-run` to see what Playchitect would do.

```bash
playchitect scan ~/Music/Techno --dry-run --target-tracks 25
```

This will output something like:

```
Scanning... 142 tracks found.
Analysed: 142 / 142 tracks (100%)

Clusters: 6
Playlist 1: 24 tracks (Avg BPM: 132.5, Intensity: High)
Playlist 2: 26 tracks (Avg BPM: 128.0, Intensity: Med)
...
```

### 2. Generate Playlists

When you're happy with the preview, remove the `--dry-run` flag to write `.m3u` files.

```bash
# Create playlists with ~25 tracks each
playchitect scan ~/Music/Techno --output ~/Playlists --target-tracks 25
```

### 3. Target Duration

Alternatively, target a specific playlist length (e.g., 90 minutes).

```bash
# Create 90-minute playlists
playchitect scan ~/Music/House --output ~/Playlists --target-duration 90
```

### 4. Info Command

Get a quick summary of a music directory without full analysis.

```bash
playchitect info ~/Music
```

## Graphical User Interface (GUI)

The GUI provides a visual way to interact with your library, see clusters, and preview tracks.

1.  **Launch the application**:

    ```bash
    playchitect-gui
    # or via flatpak:
    flatpak run com.github.jameswestwood.Playchitect
    ```

2.  **Open Folder**: Click the folder icon or use the menu to select your music directory.

3.  **Wait for Analysis**: The app will scan and analyze your tracks. This might take a moment for large libraries as it extracts audio features.

4.  **Explore Clusters**:
    *   The sidebar shows generated clusters (e.g., "Cluster 1 - Hard", "Cluster 2 - Deep").
    *   Click a cluster to view the tracks within it.
    *   Press **Spacebar** on a selected track to preview it (uses GNOME Sushi).

5.  **Export**:
    *   Use the "Export" button in the sidebar or menu to save your playlists as `.m3u` files.
    *   Optionally generate `.cue` sheets for burning to CD or precise digital playback.
