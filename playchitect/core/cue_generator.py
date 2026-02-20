"""CUE sheet generator for playlist clusters.

Produces standard CUE sheets with frame-accurate timing derived from track
duration metadata.  Two formats are supported:

- **Single-file CUE**: one CUE sheet referencing a shared file (e.g. an M3U
  playlist or a hypothetical concatenated mix).  This is the most common DJ
  format and is what most media players expect.
- **Per-track CUE** (``per_track=True``): one CUE sheet per audio file, each
  containing a single TRACK 01 entry at INDEX 01 00:00:00.  Useful for
  standalone file tagging workflows.

Depends on :mod:`playchitect.core.cue_timing` for frame-accurate conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from playchitect.core.cue_timing import cumulative_offsets, seconds_to_cue_time

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult
    from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)

# Default file type tag used in the FILE directive.
_DEFAULT_FILE_TYPE: str = "MP3"

# Map audio extensions to CUE FILE type strings.
_EXT_TO_FILE_TYPE: dict[str, str] = {
    ".mp3": "MP3",
    ".flac": "WAVE",
    ".wav": "WAVE",
    ".aiff": "AIFF",
    ".aif": "AIFF",
    ".ogg": "OGG",
}


@dataclass
class CueTrack:
    """Data for one TRACK entry in a CUE sheet."""

    number: int  # 1-based
    filepath: Path
    title: str
    performer: str
    bpm: float
    index_time: str  # "MM:SS:FF"


@dataclass
class CueSheet:
    """An in-memory CUE sheet ready for rendering.

    Build instances via :class:`CueGenerator` rather than directly.
    """

    performer: str
    title: str
    file_ref: str  # e.g. "playlist.m3u"
    file_type: str  # e.g. "MP3", "WAVE"
    tracks: list[CueTrack] = field(default_factory=list)

    def render(self) -> str:
        """Render to a CUE sheet string.

        Returns:
            Newline-terminated string in standard CUE format.
        """
        lines: list[str] = []
        lines.append(f'PERFORMER "{self.performer}"')
        lines.append(f'TITLE "{self.title}"')
        lines.append(f'FILE "{self.file_ref}" {self.file_type}')
        for track in self.tracks:
            lines.append(f"  TRACK {track.number:02d} AUDIO")
            lines.append(f'    TITLE "{track.title}"')
            lines.append(f'    PERFORMER "{track.performer}"')
            if track.bpm > 0:
                lines.append(f"    REM BPM {int(round(track.bpm))}")
            lines.append(f"    INDEX 01 {track.index_time}")
        return "\n".join(lines) + "\n"


class CueGenerator:
    """Generates :class:`CueSheet` objects from clustering results.

    Usage::

        gen = CueGenerator()
        sheet = gen.generate(cluster, metadata_dict, title="Techno Mix 1")
        print(sheet.render())
    """

    def generate(
        self,
        cluster: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
        *,
        title: str = "",
        performer: str = "Various Artists",
        file_ref: str = "playlist.m3u",
        file_type: str = _DEFAULT_FILE_TYPE,
    ) -> CueSheet:
        """Build a single-file :class:`CueSheet` for one cluster.

        Track INDEX 01 timestamps are derived from cumulative track durations
        in the order they appear in ``cluster.tracks``.  Tracks absent from
        *metadata_dict* use placeholder title/performer and 0.0 duration.

        Args:
            cluster: The cluster whose tracks to include.
            metadata_dict: Mapping of ``Path → TrackMetadata`` for titles,
                performers, BPM, and durations.
            title: CUE sheet TITLE field.  Defaults to
                ``"Cluster {cluster_id}"``.
            performer: CUE sheet PERFORMER field.
            file_ref: FILE directive path (e.g. the M3U playlist filename).
            file_type: FILE directive type tag (e.g. ``"MP3"``).

        Returns:
            A populated :class:`CueSheet`.
        """
        resolved_title = title or f"Cluster {cluster.cluster_id}"

        # Build ordered list of (path, metadata | None)
        ordered: list[tuple[Path, TrackMetadata | None]] = [
            (p, metadata_dict.get(p)) for p in cluster.tracks
        ]

        durations = [(meta.duration or 0.0) if meta is not None else 0.0 for _, meta in ordered]
        offsets = cumulative_offsets(durations)

        cue_tracks: list[CueTrack] = []
        for i, ((path, meta), offset) in enumerate(zip(ordered, offsets), start=1):
            if meta is not None:
                t_title = meta.title or path.stem
                t_performer = meta.artist or "Unknown"
                t_bpm = meta.bpm or 0.0
            else:
                t_title = path.stem
                t_performer = "Unknown"
                t_bpm = 0.0
                logger.warning("No metadata for %s; using filename stem", path.name)

            cue_tracks.append(
                CueTrack(
                    number=i,
                    filepath=path,
                    title=t_title,
                    performer=t_performer,
                    bpm=t_bpm,
                    index_time=seconds_to_cue_time(offset),
                )
            )

        return CueSheet(
            performer=performer,
            title=resolved_title,
            file_ref=file_ref,
            file_type=file_type,
            tracks=cue_tracks,
        )

    def generate_per_track(
        self,
        cluster: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
        *,
        performer: str = "Various Artists",
    ) -> list[CueSheet]:
        """Build one per-track :class:`CueSheet` for every track in a cluster.

        Each sheet has a single TRACK 01 entry at INDEX 01 00:00:00, with the
        FILE directive pointing to the actual audio file.

        Args:
            cluster: The cluster whose tracks to process.
            metadata_dict: Mapping of ``Path → TrackMetadata``.
            performer: Fallback PERFORMER when no artist metadata is present.

        Returns:
            List of :class:`CueSheet`, one per track, in cluster order.
        """
        sheets: list[CueSheet] = []
        for path in cluster.tracks:
            meta = metadata_dict.get(path)
            if meta is not None:
                t_title = meta.title or path.stem
                t_performer = meta.artist or performer
                t_bpm = meta.bpm or 0.0
            else:
                t_title = path.stem
                t_performer = performer
                t_bpm = 0.0

            ext = path.suffix.lower()
            file_type = _EXT_TO_FILE_TYPE.get(ext, _DEFAULT_FILE_TYPE)

            track = CueTrack(
                number=1,
                filepath=path,
                title=t_title,
                performer=t_performer,
                bpm=t_bpm,
                index_time="00:00:00",
            )
            sheet = CueSheet(
                performer=t_performer,
                title=t_title,
                file_ref=path.name,
                file_type=file_type,
                tracks=[track],
            )
            sheets.append(sheet)

        return sheets
