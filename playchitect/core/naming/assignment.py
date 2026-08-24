"""Cluster-to-playlist-name assignment helpers.

Wires the naming package into the scan/export pipeline so that clusters with
intensity features receive character names, while BPM-only runs keep the legacy
BPM-range labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from playchitect.core.naming.playlist_namer import PlaylistNamer
from playchitect.core.naming.vibe_profiler import (
    VibeProfile,
    compute_vibe_profile,
    score_salience,
)

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult
    from playchitect.core.intensity_analyzer import IntensityFeatures
    from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)


def _legacy_name(cluster: ClusterResult, index: int, playlist_name: str) -> str:
    """Return the legacy BPM-range label used by the exporters."""
    bpm_label = f"{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm"
    genre_label = f" {cluster.genre}" if cluster.genre else ""
    return f"{playlist_name} {index + 1} [{bpm_label}{genre_label}]"


def _bpm_range_suffix(cluster: ClusterResult) -> str:
    """Return the BPM-range suffix in the existing exporter style."""
    return f"[{int(cluster.bpm_mean)}-{int(cluster.bpm_mean + cluster.bpm_std)}bpm]"


def _raw_character_name(
    cluster: ClusterResult,
    features: dict[Path, IntensityFeatures],
    library_profiles: list[VibeProfile],
) -> str | None:
    """Generate a raw, un-suffixed character name for a cluster.

    Args:
        cluster: Cluster to name.
        features: Mapping of track path to intensity features.
        library_profiles: Vibe profiles of all clusters that have features.

    Returns:
        A character name, or None if the cluster has no computable profile.
    """
    try:
        profile = compute_vibe_profile(cluster, features)
    except ValueError:
        logger.debug("No vibe profile for cluster %s (missing features)", cluster.cluster_id)
        return None

    salience = score_salience(profile, library_profiles) if len(library_profiles) > 1 else {}

    # Use a fresh namer per cluster so duplicate raw names are visible to the
    # caller and can be disambiguated with BPM-range suffixes.
    namer = PlaylistNamer()
    return namer.name_cluster(profile, salience, library_profiles)


def assign_cluster_names(
    clusters: list[ClusterResult],
    features: dict[Path, IntensityFeatures],
    metadata: dict[Path, TrackMetadata],
    playlist_name: str,
) -> dict[int | str, str]:
    """Assign a display name to every cluster.

    When ``features`` is non-empty, each cluster is named from its character via
    the naming package. Clusters that produce the same raw character name are
    disambiguated by appending the cluster's BPM range (e.g.
    ``"Dark Hypnotic [128-131bpm]"``). Clusters with no computable vibe profile
    fall back to ``"Cluster {cluster_id}"``.

    When ``features`` is empty, the legacy BPM-range label used by the exporters
    is returned for every cluster.

    Args:
        clusters: Final list of clusters to name.
        features: Mapping of track path to intensity features. May be empty for
            BPM-only runs.
        metadata: Mapping of track path to track metadata. Kept for API symmetry
            with the naming package; not used by the current implementation.
        playlist_name: Base playlist name used by the legacy label.

    Returns:
        Mapping of ``cluster_id`` to display name.
    """
    if not clusters:
        return {}

    if not features:
        logger.debug("No intensity features; using legacy BPM-range labels")
        return {
            cluster.cluster_id: _legacy_name(cluster, i, playlist_name)
            for i, cluster in enumerate(clusters)
        }

    # Compute profiles for every cluster that has feature data.
    profiles: dict[int | str, VibeProfile] = {}
    for cluster in clusters:
        try:
            profiles[cluster.cluster_id] = compute_vibe_profile(cluster, features)
        except ValueError:
            logger.debug(
                "Skipping vibe profile for cluster %s (no feature rows)",
                cluster.cluster_id,
            )

    library_profiles = list(profiles.values())

    raw_names: dict[int | str, str] = {}
    for cluster in clusters:
        if cluster.cluster_id not in profiles:
            raw_names[cluster.cluster_id] = f"Cluster {cluster.cluster_id}"
            continue

        name = _raw_character_name(cluster, features, library_profiles)
        if name is None:
            raw_names[cluster.cluster_id] = f"Cluster {cluster.cluster_id}"
            continue

        raw_names[cluster.cluster_id] = name

    # Detect duplicate raw character names and suffix with BPM ranges.
    base_counts: dict[str, list[int | str]] = {}
    for cluster_id, name in raw_names.items():
        base_counts.setdefault(name, []).append(cluster_id)

    result: dict[int | str, str] = {}
    for cluster in clusters:
        cluster_id = cluster.cluster_id
        base_name = raw_names[cluster_id]
        if len(base_counts.get(base_name, [])) > 1:
            result[cluster_id] = f"{base_name} {_bpm_range_suffix(cluster)}"
        else:
            result[cluster_id] = base_name

    return result
