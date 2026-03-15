"""Playlist naming support: vibe profiling, grammar engine, and intelligent naming."""

from playchitect.core.naming.grammar_engine import generate_name
from playchitect.core.naming.playlist_namer import TAG_TO_DESCRIPTORS, PlaylistNamer
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
    "generate_name",
    "PlaylistNamer",
    "TAG_TO_DESCRIPTORS",
]
