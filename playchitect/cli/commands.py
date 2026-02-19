"""
CLI commands for Playchitect.
"""

import logging
import sys
from pathlib import Path

import click

from playchitect.core.audio_scanner import AudioScanner
from playchitect.core.clustering import PlaylistClusterer
from playchitect.core.export import M3UExporter
from playchitect.core.metadata_extractor import MetadataExtractor
from playchitect.utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Playchitect - Smart DJ Playlist Manager with Intelligent BPM Clustering."""
    pass


@cli.command()
@click.argument(
    "music_path",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for playlists (default: same as music_path)",
)
@click.option("--target-tracks", "-t", type=int, help="Target number of tracks per playlist")
@click.option("--target-duration", "-d", type=int, help="Target duration per playlist in minutes")
@click.option(
    "--playlist-name",
    "-n",
    type=str,
    default="Playlist",
    help="Base name for playlists (default: Playlist)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Analyze and cluster but don't create playlist files",
)
@click.option(
    "--use-test-path",
    is_flag=True,
    help="Use test music path from config (for testing)",
)
def scan(
    music_path: Path | None,
    output: Path | None,
    target_tracks: int | None,
    target_duration: int | None,
    playlist_name: str,
    dry_run: bool,
    use_test_path: bool,
) -> None:
    """
    Scan music directory and create intelligent playlists.

    MUSIC_PATH: Directory containing audio files to analyze

    Specify either --target-tracks or --target-duration to control playlist size.

    Use --dry-run to analyze without creating files (for testing).
    Use --use-test-path to use the test path from config.
    """
    # Get config
    config = get_config()

    # Determine music path
    if use_test_path:
        music_path = config.get_test_music_path()
        if not music_path:
            click.echo("Error: No test_music_path configured", err=True)
            click.echo(f"Set it in: {config.config_path}", err=True)
            sys.exit(1)
        click.echo(f"Using test path from config: {music_path}")
    elif music_path is None:
        click.echo("Error: MUSIC_PATH is required (or use --use-test-path)", err=True)
        sys.exit(1)

    # Validate target parameters
    if target_tracks is None and target_duration is None:
        target_tracks = 25  # Default to 25 tracks
    elif target_tracks and target_duration:
        click.echo("Error: Specify either --target-tracks or --target-duration, not both", err=True)
        sys.exit(1)

    if dry_run:
        click.echo("🔍 DRY RUN MODE - No files will be created")

    click.echo(f"Scanning directory: {music_path}")

    # Set output directory
    output_dir = output or music_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for audio files
    scanner = AudioScanner()
    try:
        audio_files = scanner.scan(music_path)
    except Exception as e:
        click.echo(f"Error scanning directory: {e}", err=True)
        sys.exit(1)

    if not audio_files:
        click.echo("No audio files found!", err=True)
        sys.exit(1)

    click.echo(f"Found {len(audio_files)} audio files")

    # Extract metadata
    click.echo("\nExtracting metadata...")
    extractor = MetadataExtractor()

    with click.progressbar(audio_files, label="Processing files", show_pos=True) as files:
        metadata_dict = {}
        for file_path in files:
            metadata = extractor.extract(file_path)
            metadata_dict[file_path] = metadata

    # Count tracks with BPM
    tracks_with_bpm = sum(1 for m in metadata_dict.values() if m.bpm is not None)
    click.echo(f"Extracted BPM from {tracks_with_bpm}/{len(audio_files)} tracks")

    if tracks_with_bpm == 0:
        click.echo("Error: No tracks with BPM metadata found", err=True)
        sys.exit(1)

    # Perform clustering
    click.echo("\nClustering tracks by BPM...")
    clusterer = PlaylistClusterer(
        target_tracks_per_playlist=target_tracks, target_duration_per_playlist=target_duration
    )

    clusters = clusterer.cluster_by_bpm(metadata_dict)

    if not clusters:
        click.echo("Error: Clustering failed", err=True)
        sys.exit(1)

    click.echo(f"\nCreated {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        duration_min = cluster.total_duration / 60 if cluster.total_duration else 0
        click.echo(
            f"  Cluster {i + 1}: {cluster.track_count} tracks, "
            f"BPM: {cluster.bpm_mean:.1f} ± {cluster.bpm_std:.1f}, "
            f"Duration: {duration_min:.1f} min"
        )

    # Export playlists (unless dry-run)
    if dry_run:
        click.echo(f"\n✓ DRY RUN: Would create {len(clusters)} playlists in {output_dir}")
        click.echo("\nPlaylist preview:")
        for i, cluster in enumerate(clusters):
            bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
            filename = f"{playlist_name} {i + 1} [{bpm_label}].m3u"
            click.echo(f"  {filename} ({cluster.track_count} tracks)")
        click.echo("\n💡 Remove --dry-run to create actual playlist files")
    else:
        click.echo(f"\nExporting playlists to {output_dir}...")
        exporter = M3UExporter(output_dir, playlist_prefix=playlist_name)
        playlist_paths = exporter.export_clusters(clusters)

        click.echo(f"\n✓ Successfully created {len(playlist_paths)} playlists:")
        for path in playlist_paths:
            click.echo(f"  {path.name}")

        click.echo(f"\nPlaylists saved to: {output_dir}")


@cli.command()
@click.argument("music_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format (default: text)",
)
def info(music_path: Path, format: str) -> None:
    """
    Display information about music directory.

    MUSIC_PATH: Directory to analyze
    """
    scanner = AudioScanner()

    try:
        audio_files = scanner.scan(music_path)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if format == "json":
        import json

        data = {
            "path": str(music_path),
            "total_files": len(audio_files),
            "files": [str(f) for f in audio_files],
        }
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"Directory: {music_path}")
        click.echo(f"Total audio files: {len(audio_files)}")

        # Group by extension
        extensions: dict[str, int] = {}
        for file in audio_files:
            ext = file.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1

        click.echo("\nFile types:")
        for ext, count in sorted(extensions.items()):
            click.echo(f"  {ext}: {count}")


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
