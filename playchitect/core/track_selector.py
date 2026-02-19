"""
Smart first/last track selector for DJ playlists.

Scores tracks within a cluster for opener and closer suitability using
intensity features. Supports user overrides for both positions.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)

# Duration thresholds for scoring bonuses
_OPENER_DURATION_BONUS_THRESHOLD: float = 300.0  # 5 minutes
_CLOSER_FADE_DURATION_THRESHOLD: float = 360.0  # 6 minutes

# Opener scoring weights
_OPENER_WEIGHT_ONSET: float = 0.40
_OPENER_WEIGHT_PERC: float = 0.30
_OPENER_WEIGHT_KICK: float = 0.20
_OPENER_WEIGHT_RMS: float = 0.10
_OPENER_DURATION_BONUS: float = 0.10

# Closer energetic scoring weights
_CLOSER_ENERGETIC_WEIGHT_RMS: float = 0.40
_CLOSER_ENERGETIC_WEIGHT_ONSET: float = 0.35
_CLOSER_ENERGETIC_WEIGHT_KICK: float = 0.25

# Closer fade scoring weights
_CLOSER_FADE_WEIGHT_ONSET: float = 0.40
_CLOSER_FADE_WEIGHT_RMS: float = 0.30
_CLOSER_FADE_DURATION_BONUS: float = 0.15


@dataclass
class TrackScore:
    """Score for a single track as opener or closer."""

    path: Path
    score: float  # 0.0–1.0
    reason: str


@dataclass
class TrackSelection:
    """Opener/closer selection result for a single cluster."""

    cluster_id: int | str
    first_tracks: list[TrackScore]  # top-N ranked opener candidates
    last_tracks: list[TrackScore]  # top-N ranked closer candidates
    user_override_first: Path | None = field(default=None)
    user_override_last: Path | None = field(default=None)

    @property
    def selected_first(self) -> Path:
        """Return override if set, else the top-scored opener."""
        if self.user_override_first is not None:
            return self.user_override_first
        return self.first_tracks[0].path

    @property
    def selected_last(self) -> Path:
        """Return override if set, else the top-scored closer."""
        if self.user_override_last is not None:
            return self.user_override_last
        return self.last_tracks[0].path


class TrackSelector:
    """Selects opener and closer tracks for each cluster."""

    def __init__(self, top_n: int = 5):
        """
        Initialise selector.

        Args:
            top_n: Number of candidate tracks to return for each position.
        """
        self.top_n = top_n

    def select(
        self,
        cluster: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        user_override_first: Path | None = None,
        user_override_last: Path | None = None,
    ) -> TrackSelection:
        """
        Score all tracks in a cluster and return opener/closer candidates.

        Args:
            cluster: Cluster to analyse.
            metadata_dict: Mapping of file path → TrackMetadata (for duration).
            intensity_dict: Mapping of file path → IntensityFeatures.
            user_override_first: Optional user-specified opener path.
            user_override_last: Optional user-specified closer path.

        Returns:
            TrackSelection with ranked candidates and applied overrides.

        Raises:
            ValueError: If an override path is not a member of the cluster.
        """
        if user_override_first is not None and user_override_first not in cluster.tracks:
            raise ValueError(
                f"Override track not in cluster {cluster.cluster_id}: {user_override_first}"
            )
        if user_override_last is not None and user_override_last not in cluster.tracks:
            raise ValueError(
                f"Override track not in cluster {cluster.cluster_id}: {user_override_last}"
            )

        # Single-track cluster — return the only track for both positions.
        if len(cluster.tracks) == 1:
            single = cluster.tracks[0]
            score = TrackScore(path=single, score=1.0, reason="only track in cluster")
            return TrackSelection(
                cluster_id=cluster.cluster_id,
                first_tracks=[score],
                last_tracks=[score],
                user_override_first=user_override_first,
                user_override_last=user_override_last,
            )

        opener_scores: list[TrackScore] = []
        closer_scores: list[TrackScore] = []

        for track in cluster.tracks:
            if track not in intensity_dict:
                logger.warning("Track missing from intensity_dict, skipping: %s", track.name)
                opener_scores.append(
                    TrackScore(path=track, score=-1.0, reason="missing intensity data")
                )
                closer_scores.append(
                    TrackScore(path=track, score=-1.0, reason="missing intensity data")
                )
                continue

            features = intensity_dict[track]
            duration = (metadata_dict[track].duration if track in metadata_dict else None) or 0.0

            opener_scores.append(self._score_opener(track, features, duration))
            closer_scores.append(self._score_closer(track, features, duration))

        opener_scores.sort(key=lambda s: s.score, reverse=True)
        closer_scores.sort(key=lambda s: s.score, reverse=True)

        return TrackSelection(
            cluster_id=cluster.cluster_id,
            first_tracks=opener_scores[: self.top_n],
            last_tracks=closer_scores[: self.top_n],
            user_override_first=user_override_first,
            user_override_last=user_override_last,
        )

    # ── Scoring helpers ────────────────────────────────────────────────────────

    def _score_opener(self, path: Path, features: IntensityFeatures, duration: float) -> TrackScore:
        """
        Score a track as a set opener (lower intensity = better).

        Formula:
            score = 0.40*(1-onset) + 0.30*(1-perc) + 0.20*(1-kick) + 0.10*(1-rms)
            Bonus: +0.10 if duration >= 300 s
        """
        base = (
            _OPENER_WEIGHT_ONSET * (1.0 - features.onset_strength)
            + _OPENER_WEIGHT_PERC * (1.0 - features.percussiveness)
            + _OPENER_WEIGHT_KICK * (1.0 - features.kick_energy)
            + _OPENER_WEIGHT_RMS * (1.0 - features.rms_energy)
        )

        has_duration_bonus = duration >= _OPENER_DURATION_BONUS_THRESHOLD
        score = min(1.0, base + (_OPENER_DURATION_BONUS if has_duration_bonus else 0.0))

        reasons: list[str] = []
        if features.onset_strength < 0.4:
            reasons.append("low onset")
        if features.rms_energy < 0.4:
            reasons.append("quiet start")
        if features.kick_energy < 0.4:
            reasons.append("low kick energy")
        if has_duration_bonus:
            reasons.append("long track (likely intro)")

        reason = " + ".join(reasons) if reasons else "moderate opener qualities"
        return TrackScore(path=path, score=round(score, 4), reason=reason)

    def _score_closer(self, path: Path, features: IntensityFeatures, duration: float) -> TrackScore:
        """
        Score a track as a set closer (two modes — take the best).

        Energetic mode: 0.40*rms + 0.35*onset + 0.25*kick
        Fade mode:      0.40*(1-onset) + 0.30*(1-rms) [+ 0.15 if duration >= 360 s]
        """
        energetic = (
            _CLOSER_ENERGETIC_WEIGHT_RMS * features.rms_energy
            + _CLOSER_ENERGETIC_WEIGHT_ONSET * features.onset_strength
            + _CLOSER_ENERGETIC_WEIGHT_KICK * features.kick_energy
        )

        fade_base = _CLOSER_FADE_WEIGHT_ONSET * (
            1.0 - features.onset_strength
        ) + _CLOSER_FADE_WEIGHT_RMS * (1.0 - features.rms_energy)
        has_fade_bonus = duration >= _CLOSER_FADE_DURATION_THRESHOLD
        fade = fade_base + (_CLOSER_FADE_DURATION_BONUS if has_fade_bonus else 0.0)

        if energetic >= fade:
            score = min(1.0, energetic)
            mode = "energetic closer"
        else:
            score = min(1.0, fade)
            mode = "smooth fade-out"

        return TrackScore(path=path, score=round(score, 4), reason=mode)
