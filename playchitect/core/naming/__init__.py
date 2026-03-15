"""Playlist naming support: vibe profiling and salience scoring."""

from playchitect.core.naming.vibe_profiler import (
    VibeProfile,
    bucket_bpm,
    bucket_energy,
    compute_vibe_profile,
    score_salience,
)

__all__ = [
    "VibeProfile",
    "compute_vibe_profile",
    "score_salience",
    "bucket_bpm",
    "bucket_energy",
]
