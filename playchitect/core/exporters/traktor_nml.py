"""Traktor NML export functionality.

Exports playlists to Native Instruments Traktor 3-compatible NML format.
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


class TraktorNMLExporter:
    """Exports playlists to Traktor 3-compatible NML format."""

    def __init__(self, output_dir: Path, playlist_prefix: str = "Playlist"):
        """
        Initialize Traktor NML exporter.

        Args:
            output_dir: Directory to save NML files
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
        Export a single cluster to Traktor NML playlist.

        Args:
            cluster: ClusterResult to export
            cluster_index: Index for numbering (1-based in filename)
            metadata_dict: Optional path -> TrackMetadata for track details
            features_dict: Optional path -> IntensityFeatures for Camelot key

        Returns:
            Path to created NML file
        """
        # Format BPM range in filename
        bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
        genre_label = f" {cluster.genre}" if cluster.genre else ""

        # Create filename
        filename = f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label}{genre_label}].nml"
        nml_path = self.output_dir / filename

        logger.info(f"Exporting cluster {cluster_index} to Traktor NML: {filename}")

        # Build NML structure
        root = ET.Element("NML", {"Version": "19"})

        # COLLECTION section - all tracks in this cluster
        n_tracks = len(cluster.tracks)
        collection = ET.SubElement(root, "COLLECTION", {"ENTRIES": str(n_tracks)})

        for track in cluster.tracks:
            # Get metadata for this track
            meta = metadata_dict.get(track) if metadata_dict else None
            features = features_dict.get(track) if features_dict else None

            # Build ENTRY element
            entry = ET.SubElement(collection, "ENTRY")

            # Primary key is the file path
            abs_path = track.resolve()
            entry.set("PRIMARYKEY", str(abs_path))

            # Title
            if meta and meta.title:
                title_elem = ET.SubElement(entry, "TITLE")
                title_elem.text = meta.title

            # Artist
            if meta and meta.artist:
                artist_elem = ET.SubElement(entry, "ARTIST")
                artist_elem.text = meta.artist

            # Location info
            location_elem = ET.SubElement(entry, "LOCATION")
            location_elem.set("DIR", str(abs_path.parent) + "/")
            location_elem.set("FILE", abs_path.name)
            location_elem.set("VOLUME", abs_path.anchor.rstrip("/"))

            # Tempo/BPM
            bpm = meta.bpm if meta and meta.bpm else cluster.bpm_mean
            tempo_elem = ET.SubElement(entry, "TEMPO")
            tempo_elem.set("BPM", f"{bpm:.2f}")

            # Key/Camelot notation
            if features and hasattr(features, "camelot_key") and features.camelot_key:
                key_elem = ET.SubElement(entry, "KEY")
                key_elem.text = features.camelot_key

            # Duration
            if meta and meta.duration:
                info_elem = ET.SubElement(entry, "INFO")
                info_elem.set("PLAYTIME", str(int(meta.duration)))

        # PLAYLISTS section
        playlists = ET.SubElement(root, "PLAYLISTS")

        # Root folder node
        root_node = ET.SubElement(playlists, "NODE", {"TYPE": "FOLDER", "NAME": "$ROOT"})

        # Subnodes count (1 for this single playlist)
        subnodes = ET.SubElement(root_node, "SUBNODES", {"COUNT": "1"})

        # Playlist node
        playlist_name = self._generate_cluster_name(cluster, cluster_index)
        playlist_node = ET.SubElement(subnodes, "NODE", {"TYPE": "PLAYLIST", "NAME": playlist_name})

        # Playlist entries
        playlist_elem = ET.SubElement(
            playlist_node, "PLAYLIST", {"ENTRIES": str(n_tracks), "TYPE": "LIST"}
        )

        # Add track entries to playlist
        for track in cluster.tracks:
            abs_path = track.resolve()
            entry = ET.SubElement(playlist_elem, "ENTRY")
            primary_key = ET.SubElement(entry, "PRIMARYKEY")
            primary_key.set("TYPE", "TRACK")
            primary_key.set("KEY", str(abs_path))

        # Write NML file with proper declaration
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(nml_path, encoding="UTF-8", xml_declaration=True)

        logger.debug(f"Wrote {n_tracks} tracks to {nml_path}")

        return nml_path

    def export_clusters(
        self,
        clusters: list[ClusterResult],
        metadata_dict: dict[Path, TrackMetadata] | None = None,
        features_dict: dict[Path, IntensityFeatures] | None = None,
    ) -> list[Path]:
        """
        Export multiple clusters to Traktor NML playlists.

        Args:
            clusters: List of ClusterResult objects
            metadata_dict: Optional path -> TrackMetadata for track details
            features_dict: Optional path -> IntensityFeatures for Camelot keys

        Returns:
            List of paths to created NML files
        """
        logger.info(f"Exporting {len(clusters)} clusters to Traktor NML in {self.output_dir}")

        nml_paths = []
        for i, cluster in enumerate(clusters):
            nml_path = self.export_cluster(
                cluster,
                cluster_index=i,
                metadata_dict=metadata_dict,
                features_dict=features_dict,
            )
            nml_paths.append(nml_path)

        logger.info(f"Successfully exported {len(nml_paths)} Traktor NML playlists")

        return nml_paths

    def _generate_cluster_name(self, cluster: ClusterResult, cluster_index: int) -> str:
        """Generate a display name for the cluster playlist."""
        bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
        if cluster.genre:
            return f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label} {cluster.genre}]"
        return f"{self.playlist_prefix} {cluster_index + 1} [{bpm_label}]"
