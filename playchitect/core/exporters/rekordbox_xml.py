"""Rekordbox XML export functionality.

Exports playlists to Pioneer Rekordbox 6-compatible XML format.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from playchitect.core.clustering import ClusterResult

if TYPE_CHECKING:
    from playchitect.core.intensity_analyzer import IntensityFeatures
    from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)


class RekordboxXMLExporter:
    """Exports playlists to Rekordbox 6-compatible XML format."""

    def __init__(self, output_dir: Path, playlist_prefix: str = "Playlist"):
        """
        Initialize Rekordbox XML exporter.

        Args:
            output_dir: Directory to save XML files
            playlist_prefix: Prefix for playlist filenames
        """
        self.output_dir = output_dir
        self.playlist_prefix = playlist_prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_cluster(
        self,
        cluster: ClusterResult,
        cluster_index: int = 0,
        metadata_dict: dict[Path, TrackMetadata] | None = None,
        features_dict: dict[Path, IntensityFeatures] | None = None,
    ) -> Path:
        """
        Export a single cluster to Rekordbox XML playlist.

        Args:
            cluster: ClusterResult to export
            cluster_index: Index for numbering (1-based in filename)
            metadata_dict: Optional path -> TrackMetadata for track details
            features_dict: Optional path -> IntensityFeatures for Camelot key

        Returns:
            Path to created XML file
        """
        # Format BPM range in filename
        bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
        genre_label = f" {cluster.genre}" if cluster.genre else ""

        # Create filename
        filename = f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label}{genre_label}].xml"
        xml_path = self.output_dir / filename

        logger.info(f"Exporting cluster {cluster_index} to Rekordbox XML: {filename}")

        # Build XML structure
        root = ET.Element("DJ_PLAYLISTS", {"Version": "1.0.0"})

        # COLLECTION section - all tracks
        collection = ET.SubElement(root, "COLLECTION")

        # Track ID counter (must be unique within the file)
        track_id = 1
        track_id_map: dict[Path, int] = {}

        for track in cluster.tracks:
            track_id_map[track] = track_id

            # Get metadata for this track
            meta = metadata_dict.get(track) if metadata_dict else None
            features = features_dict.get(track) if features_dict else None

            # Build TRACK element
            track_elem = ET.SubElement(collection, "TRACK")
            track_elem.set("TrackID", str(track_id))

            # Name (title)
            title = meta.title if meta and meta.title else track.stem
            track_elem.set("Name", title)

            # Artist
            artist = meta.artist if meta and meta.artist else ""
            track_elem.set("Artist", artist)

            # TotalTime (duration in seconds)
            duration = int(meta.duration) if meta and meta.duration else 0
            track_elem.set("TotalTime", str(duration))

            # AverageBpm (2 decimal places)
            bpm = meta.bpm if meta and meta.bpm else cluster.bpm_mean
            track_elem.set("AverageBpm", f"{bpm:.2f}")

            # Tonality (Camelot key or empty string)
            camelot_key = ""
            if features and hasattr(features, "camelot_key"):
                camelot_key = features.camelot_key or ""
            track_elem.set("Tonality", camelot_key)

            # Location (file://localhost + absolute path)
            abs_path = track.resolve()
            location = f"file://localhost{abs_path}"
            track_elem.set("Location", location)

            track_id += 1

        # PLAYLISTS section - the actual playlist structure
        playlists = ET.SubElement(root, "PLAYLISTS")

        # ROOT node
        root_node = ET.SubElement(playlists, "NODE", {"Type": "0", "Name": "ROOT"})

        # Cluster playlist node (Type 1 = playlist)
        cluster_name = self._generate_cluster_name(cluster, cluster_index)
        playlist_node = ET.SubElement(root_node, "NODE", {"Type": "1", "Name": cluster_name})

        # Add tracks to playlist
        for track in cluster.tracks:
            track_key = str(track_id_map[track])
            ET.SubElement(playlist_node, "TRACK", {"Key": track_key})

        # Write XML file with proper declaration
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)

        logger.debug(f"Wrote {len(cluster.tracks)} tracks to {xml_path}")

        return xml_path

    def export_clusters(
        self,
        clusters: list[ClusterResult],
        metadata_dict: dict[Path, TrackMetadata] | None = None,
        features_dict: dict[Path, IntensityFeatures] | None = None,
    ) -> list[Path]:
        """
        Export multiple clusters to Rekordbox XML playlists.

        Args:
            clusters: List of ClusterResult objects
            metadata_dict: Optional path -> TrackMetadata for track details
            features_dict: Optional path -> IntensityFeatures for Camelot keys

        Returns:
            List of paths to created XML files
        """
        logger.info(f"Exporting {len(clusters)} clusters to Rekordbox XML in {self.output_dir}")

        xml_paths = []
        for i, cluster in enumerate(clusters):
            xml_path = self.export_cluster(
                cluster,
                cluster_index=i,
                metadata_dict=metadata_dict,
                features_dict=features_dict,
            )
            xml_paths.append(xml_path)

        logger.info(f"Successfully exported {len(xml_paths)} Rekordbox XML playlists")

        return xml_paths

    def _generate_cluster_name(self, cluster: ClusterResult, cluster_index: int) -> str:
        """Generate a display name for the cluster playlist."""
        bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
        if cluster.genre:
            return f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label} {cluster.genre}]"
        return f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label}]"
