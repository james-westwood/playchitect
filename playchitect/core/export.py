"""
Playlist export functionality.

Supports M3U and CUE sheet export formats.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from playchitect.core.clustering import ClusterResult
from playchitect.core.cue_generator import CueGenerator

if TYPE_CHECKING:
    from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)


class M3UExporter:
    """Exports playlists to M3U format."""

    def __init__(self, output_dir: Path, playlist_prefix: str = "Playlist"):
        """
        Initialize M3U exporter.

        Args:
            output_dir: Directory to save playlists
            playlist_prefix: Prefix for playlist filenames
        """
        self.output_dir = output_dir
        self.playlist_prefix = playlist_prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_cluster(
        self,
        cluster: ClusterResult,
        cluster_index: int = 0,
        metadata_dict: "dict[Path, TrackMetadata] | None" = None,
    ) -> Path:
        """
        Export a single cluster to M3U playlist.

        Args:
            cluster: ClusterResult to export
            cluster_index: Index for numbering (1-based in filename)
            metadata_dict: Optional path -> TrackMetadata for #EXTINF artist/title

        Returns:
            Path to created playlist file
        """
        # Format BPM range in filename
        bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
        genre_label = f" {cluster.genre}" if cluster.genre else ""

        # Create filename
        filename = f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label}{genre_label}].m3u"
        playlist_path = self.output_dir / filename

        logger.info(f"Exporting cluster {cluster_index} to {filename}")

        # Write M3U file
        with open(playlist_path, "w", encoding="utf-8") as f:
            # Write M3U header
            f.write("#EXTM3U\n")

            for track in cluster.tracks:
                # Try to make path relative to output directory
                try:
                    rel_path = track.relative_to(self.output_dir)
                except ValueError:
                    # If not relative, use absolute path
                    rel_path = track

                # Optional #EXTINF with artist, title, genre
                if metadata_dict and track in metadata_dict:
                    meta = metadata_dict[track]
                    duration = int(meta.duration or 0)
                    artist = meta.artist or "Unknown"
                    title = meta.title or track.stem
                    genre_part = f" [{cluster.genre}]" if cluster.genre else ""
                    f.write(f"#EXTINF:{duration},{artist} - {title}{genre_part}\n")
                f.write(f"{rel_path}\n")

        logger.debug(f"Wrote {len(cluster.tracks)} tracks to {playlist_path}")

        return playlist_path

    def export_clusters(
        self,
        clusters: list[ClusterResult],
        metadata_dict: "dict[Path, TrackMetadata] | None" = None,
    ) -> list[Path]:
        """
        Export multiple clusters to M3U playlists.

        Args:
            clusters: List of ClusterResult objects
            metadata_dict: Optional path -> TrackMetadata for #EXTINF lines

        Returns:
            List of paths to created playlist files
        """
        logger.info(f"Exporting {len(clusters)} clusters to {self.output_dir}")

        playlist_paths = []
        for i, cluster in enumerate(clusters):
            playlist_path = self.export_cluster(
                cluster, cluster_index=i, metadata_dict=metadata_dict
            )
            playlist_paths.append(playlist_path)

        logger.info(f"Successfully exported {len(playlist_paths)} playlists")

        return playlist_paths

    def export_as_playlist(
        self,
        tracks: list[Path],
        metadata_dict: "dict[Path, TrackMetadata] | None" = None,
        chapter_boundaries: "list[tuple[int, str]] | None" = None,
        filename: str = "set.m3u",
    ) -> Path:
        """
        Export all tracks as a single flat M3U file with optional chapter markers.

        Args:
            tracks: List of track paths in order
            metadata_dict: Optional path -> TrackMetadata for #EXTINF lines
            chapter_boundaries: Optional list of (track_index, chapter_name) tuples
                for chapter boundary comments
            filename: Output filename

        Returns:
            Path to created playlist file
        """
        playlist_path = self.output_dir / filename

        logger.info(f"Exporting playlist with {len(tracks)} tracks to {filename}")

        with open(playlist_path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")

            chapter_idx = 0
            for i, track in enumerate(tracks):
                # Check if this is a chapter boundary
                if chapter_boundaries and chapter_idx < len(chapter_boundaries):
                    if i == chapter_boundaries[chapter_idx][0]:
                        chapter_name = chapter_boundaries[chapter_idx][1]
                        f.write(f"# --- {chapter_name} ---\n")
                        chapter_idx += 1

                # Make path relative or absolute
                try:
                    rel_path = track.relative_to(self.output_dir)
                except ValueError:
                    rel_path = track

                # Write #EXTINF with metadata
                if metadata_dict and track in metadata_dict:
                    meta = metadata_dict[track]
                    duration = int(meta.duration or 0)
                    artist = meta.artist or "Unknown"
                    title = meta.title or track.stem
                    f.write(f"#EXTINF:{duration},{artist} - {title}\n")
                f.write(f"{rel_path}\n")

        logger.debug(f"Wrote {len(tracks)} tracks to {playlist_path}")

        return playlist_path

    def export_as_chapters(
        self,
        tracks: list[Path],
        metadata_dict: "dict[Path, TrackMetadata] | None" = None,
        chapter_boundaries: "list[tuple[int, str]] | None" = None,
    ) -> list[Path]:
        """
        Export each chapter as a separate M3U file.

        Args:
            tracks: List of track paths in order
            metadata_dict: Optional path -> TrackMetadata for #EXTINF lines
            chapter_boundaries: List of (track_index, chapter_name) tuples defining chapters.
                Each tuple marks the start of a chapter.

        Returns:
            List of paths to created playlist files
        """
        if not chapter_boundaries:
            logger.warning("No chapter boundaries provided, exporting as single file")
            return [self.export_as_playlist(tracks, metadata_dict, filename="set.m3u")]

        logger.info(f"Exporting {len(chapter_boundaries)} chapters to {self.output_dir}")

        playlist_paths = []

        for chapter_num, (start_idx, chapter_name) in enumerate(chapter_boundaries, 1):
            # Determine end index for this chapter
            if chapter_num < len(chapter_boundaries):
                end_idx = chapter_boundaries[chapter_num][0]
            else:
                end_idx = len(tracks)

            chapter_tracks = tracks[start_idx:end_idx]

            # Sanitize chapter name for filename
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in chapter_name)
            filename = f"{chapter_num:02d}_{safe_name}.m3u"
            chapter_path = self.output_dir / filename

            with open(chapter_path, "w", encoding="utf-8") as f:
                f.write("#EXTM3U\n")
                f.write(f"# {chapter_name}\n")

                for track in chapter_tracks:
                    try:
                        rel_path = track.relative_to(self.output_dir)
                    except ValueError:
                        rel_path = track

                    if metadata_dict and track in metadata_dict:
                        meta = metadata_dict[track]
                        duration = int(meta.duration or 0)
                        artist = meta.artist or "Unknown"
                        title = meta.title or track.stem
                        f.write(f"#EXTINF:{duration},{artist} - {title}\n")
                    f.write(f"{rel_path}\n")

            logger.debug(f"Wrote {len(chapter_tracks)} tracks to {filename}")
            playlist_paths.append(chapter_path)

        logger.info(f"Successfully exported {len(playlist_paths)} chapter playlists")

        return playlist_paths


class CUEExporter:
    """Exports playlists to CUE sheet format."""

    def __init__(
        self,
        output_dir: Path,
        playlist_prefix: str = "Playlist",
        per_track: bool = False,
    ):
        """
        Initialize CUE exporter.

        Args:
            output_dir: Directory to save CUE sheets.
            playlist_prefix: Prefix for CUE sheet filenames.
            per_track: When True, write one CUE file per track instead of
                one combined sheet per cluster.
        """
        self.output_dir = output_dir
        self.playlist_prefix = playlist_prefix
        self.per_track = per_track
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._generator = CueGenerator()

    def export_cluster(
        self,
        cluster: ClusterResult,
        cluster_index: int = 0,
        metadata_dict: "dict[Path, TrackMetadata] | None" = None,
    ) -> Path:
        """
        Export a single cluster to a CUE sheet (single-file format).

        Args:
            cluster: ClusterResult to export.
            cluster_index: Index for numbering (1-based in filename).
            metadata_dict: Optional path → TrackMetadata for titles and timing.

        Returns:
            Path to the created CUE file.
        """
        bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
        genre_label = f" {cluster.genre}" if cluster.genre else ""
        filename = f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label}{genre_label}].cue"
        cue_path = self.output_dir / filename
        m3u_ref = filename.replace(".cue", ".m3u")

        sheet = self._generator.generate(
            cluster,
            metadata_dict or {},
            title=f"{self.playlist_prefix} {cluster_index + 1}{genre_label}",
            file_ref=m3u_ref,
        )

        logger.info("Writing CUE sheet: %s (%d tracks)", filename, len(sheet.tracks))
        cue_path.write_text(sheet.render(), encoding="utf-8")
        return cue_path

    def export_clusters(
        self,
        clusters: list[ClusterResult],
        metadata_dict: "dict[Path, TrackMetadata] | None" = None,
    ) -> list[Path]:
        """
        Export multiple clusters to CUE sheets.

        Args:
            clusters: List of ClusterResult objects.
            metadata_dict: Optional path → TrackMetadata for titles and timing.

        Returns:
            List of paths to created CUE files.
        """
        logger.info("Exporting %d clusters to %s", len(clusters), self.output_dir)
        paths = []
        for i, cluster in enumerate(clusters):
            path = self.export_cluster(cluster, cluster_index=i, metadata_dict=metadata_dict)
            paths.append(path)
        logger.info("Exported %d CUE sheets", len(paths))
        return paths
