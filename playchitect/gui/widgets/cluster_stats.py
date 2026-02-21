"""Display-ready statistics derived from ClusterResult.

This module is intentionally GTK-free so it can be unit-tested without a
display server and reused by both the GUI and any future CLI summary commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult

# Intensity thresholds (normalised RMS energy, [0, 1]).
_LOW_THRESHOLD: float = 0.33
_HIGH_THRESHOLD: float = 0.66
_VERY_HIGH_THRESHOLD: float = 0.85

# Bar characters used for visual intensity / BPM-fill displays.
_BAR_FULL: str = "█"
_BAR_EMPTY: str = "░"
_BAR_WIDTH: int = 10


@dataclass(frozen=True)
class ClusterStats:
    """Computed display values for a single cluster.

    All string properties are ready to embed directly into UI labels.
    """

    cluster_id: int | str
    track_count: int
    bpm_min: float
    bpm_max: float
    bpm_mean: float
    intensity_mean: float  # normalised [0, 1]; 0 if no feature data
    hardness_mean: float  # combined hardness score [0, 1]
    total_duration: float  # seconds
    opener_name: str | None = None
    closer_name: str | None = None
    feature_importance: list[tuple[str, float]] = field(default_factory=list)
    # Sorted descending by importance; empty when clustering was BPM-only.

    # ── Class constructor ─────────────────────────────────────────────────────

    @classmethod
    def from_result(cls, result: ClusterResult) -> ClusterStats:
        """Build a ``ClusterStats`` from a ``ClusterResult``."""
        # Estimate BPM range from mean ± std (1σ covers ~68% of tracks).
        # Clamp so min is always ≥ 1 and min < max.
        std = result.bpm_std if result.bpm_std > 0 else 0.0
        bpm_min = max(1.0, result.bpm_mean - std)
        bpm_max = max(bpm_min + 1.0, result.bpm_mean + std)

        # Intensity: prefer RMS energy from feature_means, fall back to 0.
        intensity_mean = 0.0
        hardness_mean = 0.0
        if result.feature_means:
            intensity_mean = result.feature_means.get("rms_energy", 0.0)
            hardness_mean = result.feature_means.get("hardness", intensity_mean)

        opener_name = result.opener.name if result.opener else None
        closer_name = result.closer.name if result.closer else None

        # Feature importance: sort descending, exclude zero-weight entries.
        importance: list[tuple[str, float]] = []
        if result.feature_importance:
            importance = sorted(
                ((k, v) for k, v in result.feature_importance.items() if v > 0),
                key=lambda kv: kv[1],
                reverse=True,
            )

        return cls(
            cluster_id=result.cluster_id,
            track_count=result.track_count,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            bpm_mean=result.bpm_mean,
            intensity_mean=intensity_mean,
            hardness_mean=hardness_mean,
            total_duration=result.total_duration,
            opener_name=opener_name,
            closer_name=closer_name,
            feature_importance=importance,
        )

    # ── Display properties ────────────────────────────────────────────────────

    @property
    def bpm_range_str(self) -> str:
        """Human-readable BPM range, e.g. ``'120–125 BPM'``."""
        return f"{self.bpm_min:.0f}–{self.bpm_max:.0f} BPM"

    @property
    def bpm_mean_str(self) -> str:
        """Mean BPM as a plain integer string, e.g. ``'122'``."""
        return str(int(round(self.bpm_mean)))

    @property
    def intensity_label(self) -> str:
        """Categorical label for the cluster's intensity level."""
        if self.intensity_mean >= _VERY_HIGH_THRESHOLD:
            return "Very High"
        if self.intensity_mean >= _HIGH_THRESHOLD:
            return "High"
        if self.intensity_mean >= _LOW_THRESHOLD:
            return "Medium"
        return "Low"

    @property
    def intensity_bars(self) -> str:
        """Ten-character unicode bar proportional to intensity, e.g. ``'███░░░░░░░'``."""
        filled = round(max(0.0, min(1.0, self.intensity_mean)) * _BAR_WIDTH)
        return _BAR_FULL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)

    @property
    def hardness_bars(self) -> str:
        """Ten-character unicode bar proportional to hardness, e.g. ``'███░░░░░░░'``."""
        filled = round(max(0.0, min(1.0, self.hardness_mean)) * _BAR_WIDTH)
        return _BAR_FULL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)

    @property
    def bpm_fill_bars(self) -> str:
        """Ten-character bar showing BPM fill relative to a 200 BPM ceiling."""
        _BPM_CEILING = 200.0
        fraction = min(1.0, self.bpm_mean / _BPM_CEILING)
        filled = round(fraction * _BAR_WIDTH)
        return _BAR_FULL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)

    @property
    def duration_str(self) -> str:
        """Total duration as ``'Xh Ym'`` or ``'Ym'``, e.g. ``'1h 35m'``."""
        total_secs = max(0, int(self.total_duration))
        hours, remainder = divmod(total_secs, 3600)
        minutes = remainder // 60
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    @property
    def track_count_str(self) -> str:
        """Track count with pluralised label, e.g. ``'23 tracks'``."""
        noun = "track" if self.track_count == 1 else "tracks"
        return f"{self.track_count} {noun}"

    @property
    def top_features(self) -> list[tuple[str, float]]:
        """Top-3 most important features (may be fewer if data is sparse)."""
        return self.feature_importance[:3]

    @property
    def cluster_label(self) -> str:
        """Short display label, e.g. ``'Cluster 1'`` or ``'1a'`` for split clusters."""
        return f"Cluster {self.cluster_id}"

    def bpm_range_fraction(self, global_min: float, global_max: float) -> float:
        """Position of this cluster's mean BPM within the global BPM range [0, 1].

        Used for rendering a relative position indicator across all cluster cards.
        Returns 0.0 if ``global_min == global_max``.
        """
        span = global_max - global_min
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (self.bpm_mean - global_min) / span))

    def intensity_fraction(self) -> float:
        """Intensity as a float in [0, 1], clamped."""
        return max(0.0, min(1.0, self.intensity_mean))

    @staticmethod
    def from_results(results: list[ClusterResult]) -> list[ClusterStats]:
        """Convert a list of ``ClusterResult`` objects to ``ClusterStats`` in one call."""
        return [ClusterStats.from_result(r) for r in results]

    @staticmethod
    def global_bpm_range(stats: list[ClusterStats]) -> tuple[float, float]:
        """Return ``(global_min_bpm, global_max_bpm)`` across all cluster stats.

        Useful for normalising BPM bars so all cards share the same scale.
        Returns ``(0.0, 200.0)`` when the list is empty.
        """
        if not stats:
            return (0.0, 200.0)
        return (
            min(s.bpm_min for s in stats),
            max(s.bpm_max for s in stats),
        )

    def __str__(self) -> str:
        return (
            f"[{self.cluster_label}] {self.bpm_range_str} | "
            f"{self.intensity_label} | {self.track_count_str} | {self.duration_str}"
        )
