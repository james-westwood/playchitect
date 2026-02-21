"""
CLI commands for Playchitect.
"""

import logging
import sys
from pathlib import Path

import click

from playchitect.core.audio_scanner import AudioScanner
from playchitect.core.clustering import PlaylistClusterer
from playchitect.core.export import CUEExporter, M3UExporter
from playchitect.core.metadata_extractor import MetadataExtractor
from playchitect.core.track_selector import TrackSelector
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
@click.option(
    "--first-override",
    type=click.Path(exists=True),
    default=None,
    help="Override the auto-selected opener track.",
)
@click.option(
    "--last-override",
    type=click.Path(exists=True),
    default=None,
    help="Override the auto-selected closer track.",
)
@click.option(
    "--save-overrides",
    is_flag=True,
    default=False,
    help="Persist --first-override / --last-override to config for future runs.",
)
@click.option(
    "--use-embeddings",
    is_flag=True,
    default=False,
    help="Use MusiCNN semantic embeddings for richer clustering (requires essentia-tensorflow).",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default=None,
    help="Path to msd-musicnn-1.pb (auto-downloaded if absent).",
)
@click.option(
    "--cue",
    "export_cue",
    is_flag=True,
    default=False,
    help="Also write a CUE sheet alongside each M3U playlist.",
)
@click.option(
    "--cluster-mode",
    type=click.Choice(["single-genre", "per-genre", "mixed-genre"]),
    default="single-genre",
    help="Clustering mode: single-genre (default), per-genre, or mixed-genre.",
)
@click.option(
    "--genre-map",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML file with manual genre assignments (manual_assignments: {path: genre}).",
)
@click.option(
    "--sequence-mode",
    type=click.Choice(["ramp", "fixed"]),
    default="fixed",
    help="Track sequencing mode: ramp (intensity build) or fixed (no change, default).",
)
def scan(
    music_path: Path | None,
    output: Path | None,
    target_tracks: int | None,
    target_duration: int | None,
    playlist_name: str,
    dry_run: bool,
    use_test_path: bool,
    first_override: str | None,
    last_override: str | None,
    save_overrides: bool,
    use_embeddings: bool,
    model_path: str | None,
    export_cue: bool,
    cluster_mode: str,
    genre_map: Path | None,
    sequence_mode: str,
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
        click.echo(
            "Error: Specify either --target-tracks or --target-duration, not both",
            err=True,
        )
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

    emb_extractor = None  # Initialize outside conditional block

    # Optional: intensity analysis + MusiCNN embeddings for Block PCA clustering
    # Intensity required when: use_embeddings OR genre-aware clustering OR ramp sequencing
    embedding_dict = None
    intensity_dict: dict = {}
    auto_genre: str | None = None
    genre_dict_resolved: dict | None = None
    need_intensity = (
        use_embeddings or cluster_mode in ("per-genre", "mixed-genre") or sequence_mode == "ramp"
    )

    if need_intensity:
        # Intensity analysis is required by cluster_by_features
        from playchitect.core.intensity_analyzer import (
            IntensityAnalyzer,
        )  # noqa: PLC0415

        click.echo("\nExtracting audio intensity features...")
        int_analyzer = IntensityAnalyzer(cache_dir=config.get_cache_dir() / "intensity")
        with click.progressbar(audio_files, label="Intensity analysis", show_pos=True) as files:
            for file_path in files:
                try:
                    intensity_dict[file_path] = int_analyzer.analyze(file_path)
                except Exception as exc:
                    logger.warning("Intensity analysis failed for %s: %s", file_path.name, exc)

        # Embedding extraction when requested (lazy import keeps ImportError contained)
        if use_embeddings:
            try:
                from playchitect.core.embedding_extractor import (
                    EmbeddingExtractor,
                )  # noqa: PLC0415

                resolved_model = (
                    Path(model_path)
                    if model_path
                    else (
                        Path(config.get("embedding_model_path"))
                        if config.get("embedding_model_path")
                        else None
                    )
                )
                emb_extractor = EmbeddingExtractor(model_path=resolved_model)  # Store the instance
                click.echo(
                    "\nExtracting MusiCNN embeddings (may download ~50 MB model on first run)..."
                )
                with click.progressbar(
                    audio_files, label="Embedding files", show_pos=True
                ) as files:
                    embedding_dict = {}
                    for file_path in files:
                        try:
                            embedding_dict[file_path] = emb_extractor.analyze(file_path)
                        except Exception as exc:
                            logger.warning("Embedding failed for %s: %s", file_path.name, exc)

                # Auto-detect genre by majority vote across all tracks
                genres: list[str] = []
                for feat in embedding_dict.values():
                    g = emb_extractor.infer_genre(feat)
                    if g:
                        genres.append(g)
                if genres:
                    auto_genre = max(set(genres), key=genres.count)
                    click.echo(f"Auto-detected genre: {auto_genre}")
            except RuntimeError as exc:
                click.echo(f"Warning: {exc}", err=True)
                click.echo("Falling back to intensity-only clustering.", err=True)
                embedding_dict = None

        # Resolve per-track genre for genre-aware modes
        if cluster_mode in ("per-genre", "mixed-genre"):
            from playchitect.core.genre_resolver import (  # noqa: PLC0415
                load_genre_map,
                resolve_genres,
            )

            gm = load_genre_map(genre_map) if genre_map else {}
            infer_fn = None
            if embedding_dict and emb_extractor:  # Use the existing emb_extractor
                infer_fn = emb_extractor.infer_genre
            genre_dict_resolved = resolve_genres(
                metadata_dict,
                embedding_dict,
                gm,
                music_root=music_path,
                infer_genre_fn=infer_fn,
            )
            known_count = sum(1 for g in genre_dict_resolved.values() if g != "unknown")
            click.echo(
                f"Genre resolution: {known_count}/{len(genre_dict_resolved)} tracks assigned"
            )

    # Perform clustering
    click.echo("\nClustering tracks...")
    clusterer = PlaylistClusterer(
        target_tracks_per_playlist=target_tracks,
        target_duration_per_playlist=target_duration,
    )

    if intensity_dict:
        clusters = clusterer.cluster_by_features(
            metadata_dict,
            intensity_dict,
            embedding_dict=embedding_dict,
            genre=auto_genre,
            cluster_mode=cluster_mode,
            genre_dict=genre_dict_resolved,
        )
    else:
        clusters = clusterer.cluster_by_bpm(metadata_dict)

    if not clusters:
        click.echo("Error: Clustering failed", err=True)
        sys.exit(1)

    # Split any cluster that exceeds the target size
    split_clusters: list = []
    for cluster in clusters:
        if target_tracks and cluster.track_count > target_tracks:
            split_clusters.extend(clusterer.split_cluster(cluster, target_tracks))
        elif target_duration and cluster.total_duration > target_duration * 60:
            avg_secs = cluster.total_duration / cluster.track_count
            target_size = max(1, int(target_duration * 60 / avg_secs))
            split_clusters.extend(clusterer.split_cluster(cluster, target_size))
        else:
            split_clusters.append(cluster)
    if len(split_clusters) != len(clusters):
        click.echo(
            f"  (split {len(clusters)} clusters → {len(split_clusters)} playlists"
            f" to meet target size)"
        )
    clusters = split_clusters

    # Perform sequencing
    from playchitect.core.sequencer import Sequencer  # noqa: PLC0415

    sequencer = Sequencer()
    if sequence_mode != "fixed":
        click.echo(f"Sequencing tracks (mode: {sequence_mode})...")
        for cluster in clusters:
            cluster.tracks = sequencer.sequence(
                cluster, metadata_dict, intensity_dict, mode=sequence_mode
            )

    click.echo(f"\nCreated {len(clusters)} playlists:")
    if cluster_mode == "mixed-genre":
        click.echo(
            "  (Mixed-genre mode: cross-genre playlists, genre labels not shown per cluster)"
        )
    for i, cluster in enumerate(clusters):
        duration_min = cluster.total_duration / 60 if cluster.total_duration else 0
        genre_label = f" [{cluster.genre}]" if cluster.genre else ""
        click.echo(
            f"  Cluster {i + 1}{genre_label}: {cluster.track_count} tracks, "
            f"BPM: {cluster.bpm_mean:.1f} ± {cluster.bpm_std:.1f}, "
            f"Duration: {duration_min:.1f} min"
        )

    from playchitect.core.clustering import _EMBEDDING_PCA_COMPONENTS  # noqa: PLC0415

    # Report embedding PCA variance when applicable
    if clusters and clusters[0].embedding_variance_explained is not None:
        var = clusters[0].embedding_variance_explained
        click.echo(
            f"\nEmbedding PCA ({_EMBEDDING_PCA_COMPONENTS} components): "
            f"{var:.1%} variance explained"
        )

    # Track selection — opener/closer recommendations per cluster
    override_first_path = Path(first_override) if first_override else None
    override_last_path = Path(last_override) if last_override else None

    # Load saved overrides when CLI flags are absent
    saved_overrides = config.get_track_override(music_path)
    applied_first = override_first_path or saved_overrides.get("first")
    applied_last = override_last_path or saved_overrides.get("last")

    selector = TrackSelector()
    if intensity_dict:
        click.echo("\nOpener / Closer recommendations:")
    else:
        click.echo(
            "\nOpener / Closer recommendations: (use --cluster-mode or"
            " --use-embeddings to enable intensity-based scoring)"
        )
    for cluster in clusters:
        if not intensity_dict:
            break
        try:
            selection = selector.select(
                cluster,
                metadata_dict,
                intensity_dict,
                user_override_first=(applied_first if applied_first in cluster.tracks else None),
                user_override_last=(applied_last if applied_last in cluster.tracks else None),
            )
            opener = selection.first_tracks[0] if selection.first_tracks else None
            closer = selection.last_tracks[0] if selection.last_tracks else None
            header = (
                f"  Cluster {cluster.cluster_id} "
                f"(BPM {cluster.bpm_mean:.1f} \u00b1 {cluster.bpm_std:.1f}):"
            )
            click.echo(header)
            if opener:
                click.echo(
                    f"    Opener: {opener.path.name}"
                    f"  (score: {opener.score:.2f} \u2014 {opener.reason})"
                )
            if closer:
                click.echo(
                    f"    Closer: {closer.path.name}"
                    f"  (score: {closer.score:.2f} \u2014 {closer.reason})"
                )
        except ValueError as exc:
            logger.warning("Track selection skipped for cluster %s: %s", cluster.cluster_id, exc)

    # Persist overrides if requested
    if save_overrides and (override_first_path or override_last_path):
        config.set_track_override(music_path, first=override_first_path, last=override_last_path)
        click.echo("\nOverrides saved to config.")

    # Export playlists (unless dry-run)
    if dry_run:
        click.echo(f"\n✓ DRY RUN: Would create {len(clusters)} playlists in {output_dir}")
        if export_cue:
            click.echo("  (CUE sheets would also be written alongside M3U playlists)")
        click.echo("\nPlaylist preview:")
        for i, cluster in enumerate(clusters):
            bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
            genre_label = f" {cluster.genre}" if cluster.genre else ""
            filename = f"{playlist_name} {i + 1} [{bpm_label}{genre_label}].m3u"
            click.echo(f"  {filename} ({cluster.track_count} tracks)")
        click.echo("\n💡 Remove --dry-run to create actual playlist files")
    else:
        click.echo(f"\nExporting playlists to {output_dir}...")
        exporter = M3UExporter(output_dir, playlist_prefix=playlist_name)
        playlist_paths = exporter.export_clusters(clusters, metadata_dict=metadata_dict)

        click.echo(f"\n✓ Successfully created {len(playlist_paths)} playlists:")
        for path in playlist_paths:
            click.echo(f"  {path.name}")

        if export_cue:
            click.echo(f"\nWriting CUE sheets to {output_dir}...")
            cue_exporter = CUEExporter(output_dir, playlist_prefix=playlist_name)
            cue_paths = cue_exporter.export_clusters(clusters, metadata_dict=metadata_dict)
            click.echo(f"✓ Successfully created {len(cue_paths)} CUE sheets:")
            for path in cue_paths:
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
