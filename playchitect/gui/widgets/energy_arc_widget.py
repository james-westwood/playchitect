"""Energy arc sparkline widget for visualizing cluster energy distribution.

Provides a visual arc/sparkline above the cluster sidebar showing the
mean RMS energy across clusters, helping users understand energy flow
before diving into individual cluster details.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    Gtk,
)

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult

logger = logging.getLogger(__name__)

# Visual styling constants
_ARC_COLOR_HEX: str = "#6B9BDF"
_ARC_FILL_ALPHA: float = 0.30
_ARC_STROKE_WIDTH: float = 2.0
_DOT_RADIUS: float = 3.0
_MIN_HEIGHT: int = 80
_PADDING_PX: int = 8


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert hex color to RGBA tuple with cairo-compatible float values."""
    hex_clean = hex_color.lstrip("#")
    r = int(hex_clean[0:2], 16) / 255.0
    g = int(hex_clean[2:4], 16) / 255.0
    b = int(hex_clean[4:6], 16) / 255.0
    return (r, g, b, alpha)


class EnergyArcWidget(Gtk.DrawingArea):
    """Sparkline widget showing cluster mean RMS energy as an arc.

    Draws a filled polygon under the curve (30% alpha), a stroke over the top,
    and small circle dots at each data point. Positioned above the cluster
    sidebar to provide immediate visual context for energy distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        self._clusters: list[tuple[str, float]] = []
        self.set_size_request(-1, _MIN_HEIGHT)
        self.set_draw_func(self._on_draw)

    def update_clusters(self, clusters: list[ClusterResult]) -> None:
        """Update the widget with new cluster data.

        Extracts cluster name and mean RMS energy from each ClusterResult,
        then triggers a redraw of the sparkline.

        Args:
            clusters: List of ClusterResult objects from clustering operation.
        """
        self._clusters = []
        for cluster in clusters:
            name = getattr(cluster, "cluster_id", "Unknown")
            feature_means = getattr(cluster, "feature_means", None) or {}
            mean_rms = feature_means.get("rms_energy", 0.0)
            self._clusters.append((str(name), float(mean_rms)))

        self.queue_draw()

    def _on_draw(
        self,
        _drawing_area: Gtk.DrawingArea,
        cr: object,  # cairo.Context
        width: int,
        height: int,
    ) -> None:
        """Draw the energy arc sparkline.

        Renders:
        - Filled polygon under the curve (RGBA with 30% alpha)
        - Stroke line over the top
        - Small circle dots at each data point
        """

        if not self._clusters:
            return

        n_points = len(self._clusters)
        if n_points < 2:
            # Single point: just draw a dot in the center
            center_x = width / 2.0
            center_y = height / 2.0
            color = _hex_to_rgba(_ARC_COLOR_HEX, 1.0)
            cr.set_source_rgba(*color)
            cr.arc(center_x, center_y, _DOT_RADIUS, 0, 2 * math.pi)
            cr.fill()
            return

        # Compute min/max for normalization
        values = [v for _, v in self._clusters]
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val > min_val else 1.0

        # Compute point positions
        effective_width = width - 2 * _PADDING_PX
        effective_height = height - 2 * _PADDING_PX
        x_step = effective_width / (n_points - 1) if n_points > 1 else 0

        points: list[tuple[float, float]] = []
        for i, (_, value) in enumerate(self._clusters):
            x = _PADDING_PX + i * x_step
            # Normalize value to 0-1, then invert y (0 at bottom, 1 at top)
            normalized = (value - min_val) / val_range if val_range > 0 else 0.5
            y = _PADDING_PX + effective_height * (1.0 - normalized)
            points.append((x, y))

        # Build polygon path (filled area under curve)
        cr.move_to(points[0][0], height - _PADDING_PX)
        for x, y in points:
            cr.line_to(x, y)
        cr.line_to(points[-1][0], height - _PADDING_PX)
        cr.close_path()

        # Fill with alpha
        fill_color = _hex_to_rgba(_ARC_COLOR_HEX, _ARC_FILL_ALPHA)
        cr.set_source_rgba(*fill_color)
        cr.fill()

        # Draw stroke over the top
        cr.move_to(points[0][0], points[0][1])
        for x, y in points[1:]:
            cr.line_to(x, y)

        stroke_color = _hex_to_rgba(_ARC_COLOR_HEX, 1.0)
        cr.set_source_rgba(*stroke_color)
        cr.set_line_width(_ARC_STROKE_WIDTH)
        cr.stroke()

        # Draw dots at each data point
        cr.set_source_rgba(*stroke_color)
        for x, y in points:
            cr.arc(x, y, _DOT_RADIUS, 0, 2 * math.pi)
            cr.fill()
