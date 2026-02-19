"""
CLI commands for Playchitect.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import click

from playchitect.core.audio_scanner import AudioScanner
from playchitect.core.metadata_extractor import MetadataExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """Playchitect - Smart DJ Playlist Manager with Intelligent BPM Clustering."""
    pass


@cli.command()
@click.argument('music_path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    help='Output directory for playlists (default: same as music_path)'
)
@click.option(
    '--target-length',
    '-t',
    type=int,
    default=25,
    help='Target number of tracks per playlist (default: 25)'
)
@click.option(
    '--clusters',
    '-c',
    type=int,
    help='Number of clusters (default: auto-determine using elbow method)'
)
@click.option(
    '--playlist-name',
    '-n',
    type=str,
    default='Playlist',
    help='Base name for playlists (default: Playlist)'
)
def scan(
    music_path: Path,
    output: Optional[Path],
    target_length: int,
    clusters: Optional[int],
    playlist_name: str
):
    """
    Scan music directory and create intelligent playlists.

    MUSIC_PATH: Directory containing audio files to analyze
    """
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

    with click.progressbar(
        audio_files,
        label='Processing files',
        show_pos=True
    ) as files:
        metadata_dict = {}
        for file_path in files:
            metadata = extractor.extract(file_path)
            metadata_dict[file_path] = metadata

    # Count tracks with BPM
    tracks_with_bpm = sum(1 for m in metadata_dict.values() if m.bpm is not None)
    click.echo(f"Extracted BPM from {tracks_with_bpm}/{len(audio_files)} tracks")

    # TODO: Implement clustering and playlist generation in Milestone 2
    click.echo("\n⚠️  Clustering and playlist generation not yet implemented")
    click.echo("This will be added in Milestone 2 (Intelligent Analysis Engine)")

    click.echo(f"\nOutput directory: {output_dir}")


@cli.command()
@click.argument('music_path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--format',
    '-f',
    type=click.Choice(['text', 'json']),
    default='text',
    help='Output format (default: text)'
)
def info(music_path: Path, format: str):
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

    if format == 'json':
        import json
        data = {
            'path': str(music_path),
            'total_files': len(audio_files),
            'files': [str(f) for f in audio_files]
        }
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"Directory: {music_path}")
        click.echo(f"Total audio files: {len(audio_files)}")

        # Group by extension
        extensions = {}
        for file in audio_files:
            ext = file.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1

        click.echo("\nFile types:")
        for ext, count in sorted(extensions.items()):
            click.echo(f"  {ext}: {count}")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
