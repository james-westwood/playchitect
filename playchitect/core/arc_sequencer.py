"""Energy arc sequencer for playlist ordering.

Provides preset energy curves that can be applied to clusters to control
the overall energy flow of a playlist (warmup, peak hour, closing, etc.).
"""

from dataclasses import dataclass

from playchitect.core.clustering import ClusterResult


@dataclass
class EnergyArcPreset:
    """A named energy arc preset with a normalized curve.

    The arc_curve is a list of float values in the range [0.0, 1.0],
    representing the desired energy level at each cluster position.
    """

    name: str
    description: str
    arc_curve: list[float]

    def __post_init__(self) -> None:
        """Validate arc_curve values are in range [0.0, 1.0]."""
        if not self.arc_curve:
            raise ValueError("arc_curve must not be empty")
        for i, value in enumerate(self.arc_curve):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"arc_curve value at index {i} must be in range [0.0, 1.0], got {value}"
                )


# Built-in energy arc presets for common DJ set structures
BUILTIN_PRESETS: list[EnergyArcPreset] = [
    EnergyArcPreset(
        name="Warmup Ramp",
        description="Gradual energy build from low to high",
        arc_curve=[0.2, 0.4, 0.6, 0.8, 1.0],
    ),
    EnergyArcPreset(
        name="Peak Hour",
        description="Quick rise to peak, sustained, then gentle decline",
        arc_curve=[0.4, 0.7, 1.0, 1.0, 0.8],
    ),
    EnergyArcPreset(
        name="Sunrise",
        description="Starts medium, dips, then rises to climax",
        arc_curve=[0.6, 0.4, 0.2, 0.4, 0.8],
    ),
    EnergyArcPreset(
        name="Deep Journey",
        description="Low variance arc with subtle dynamic shifts",
        arc_curve=[0.5, 0.3, 0.4, 0.6, 0.5],
    ),
    EnergyArcPreset(
        name="Closing Down",
        description="Starts at peak energy and gradually winds down",
        arc_curve=[1.0, 0.9, 0.7, 0.5, 0.3],
    ),
]


def _get_cluster_rms_energy(cluster: ClusterResult) -> float:
    """Extract mean RMS energy from cluster, or 0.5 as fallback."""
    if cluster.feature_means is not None and "rms_energy" in cluster.feature_means:
        return float(cluster.feature_means["rms_energy"])
    # Fallback: return middle value if no energy data available
    return 0.5


def apply_arc(
    clusters: list[ClusterResult],
    preset: EnergyArcPreset,
) -> list[ClusterResult]:
    """Apply an energy arc preset to reorder clusters.

    Sorts clusters by their mean RMS energy (ascending), then assigns each
    cluster to the arc position whose target value is closest to the cluster's
    normalized energy (greedy closest-value matching).

    Args:
        clusters: List of ClusterResult objects to reorder
        preset: EnergyArcPreset defining the desired energy curve

    Returns:
        Reordered list of ClusterResult objects matching the arc curve

    Raises:
        ValueError: If clusters is empty or arc_curve is empty
    """
    if not clusters:
        raise ValueError("clusters must not be empty")

    if not preset.arc_curve:
        raise ValueError("preset.arc_curve must not be empty")

    # Get energy for each cluster and sort by energy (ascending)
    clusters_with_energy = [(c, _get_cluster_rms_energy(c)) for c in clusters]
    clusters_with_energy.sort(key=lambda x: x[1])

    # If fewer clusters than arc positions, use only needed positions
    # If more clusters than arc positions, some clusters will share positions
    arc_curve = preset.arc_curve
    num_positions = len(arc_curve)

    # Build result by matching each cluster to best arc position
    assigned: list[tuple[ClusterResult, int]] = []
    available_positions = list(range(num_positions))

    for cluster, energy in clusters_with_energy:
        # Find the arc position with target value closest to cluster energy
        best_pos = min(
            available_positions,
            key=lambda pos: abs(arc_curve[pos] - energy),
        )
        assigned.append((cluster, best_pos))

    # Sort by assigned arc position to get final order
    assigned.sort(key=lambda x: x[1])

    # Return clusters in the new order
    return [cluster for cluster, _ in assigned]
