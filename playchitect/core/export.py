"""
Playlist export functionality.

MVP: M3U export with relative paths.
Future: CUE sheet generation, absolute path option.
"""

import logging
from pathlib import Path
from typing import List

from playchitect.core.clustering import ClusterResult

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

    def export_cluster(self, cluster: ClusterResult, cluster_index: int = 0) -> Path:
        """
        Export a single cluster to M3U playlist.

        Args:
            cluster: ClusterResult to export
            cluster_index: Index for numbering (1-based in filename)

        Returns:
            Path to created playlist file
        """
        # Format BPM range in filename
        bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"

        # Create filename
        filename = f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label}].m3u"
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

                f.write(f"{rel_path}\n")

        logger.debug(f"Wrote {len(cluster.tracks)} tracks to {playlist_path}")

        return playlist_path

    def export_clusters(self, clusters: List[ClusterResult]) -> List[Path]:
        """
        Export multiple clusters to M3U playlists.

        Args:
            clusters: List of ClusterResult objects

        Returns:
            List of paths to created playlist files
        """
        logger.info(f"Exporting {len(clusters)} clusters to {self.output_dir}")

        playlist_paths = []
        for i, cluster in enumerate(clusters):
            playlist_path = self.export_cluster(cluster, cluster_index=i)
            playlist_paths.append(playlist_path)

        logger.info(f"Successfully exported {len(playlist_paths)} playlists")

        return playlist_paths


class CUEExporter:
    """Exports playlists to CUE sheet format."""

    def __init__(self, output_dir: Path):
        """
        Initialize CUE exporter.

        Args:
            output_dir: Directory to save CUE sheets
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_cluster(self, cluster: ClusterResult) -> Path:
        """
        Export cluster to CUE sheet.

        TODO: Implement in Milestone 4

        Args:
            cluster: ClusterResult to export

        Returns:
            Path to created CUE file
        """
        raise NotImplementedError("CUE export will be implemented in Milestone 4")
