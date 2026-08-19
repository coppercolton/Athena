"""Terminal plotting, so the demos need nothing but numpy to show their work."""

from __future__ import annotations

import math
from typing import Sequence

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], width: int = 60) -> str:
    vals = _resample(list(values), width)
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return _BLOCKS[0] * len(vals)
    return "".join(_BLOCKS[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def _resample(values: list[float], width: int) -> list[float]:
    if len(values) <= width:
        return values
    step = len(values) / width
    out = []
    for i in range(width):
        chunk = values[int(i * step) : max(int((i + 1) * step), int(i * step) + 1)]
        out.append(sum(chunk) / len(chunk))
    return out


def chart(
    series: dict[str, Sequence[float]],
    height: int = 14,
    width: int = 74,
    log: bool = False,
    marks: Sequence[int] = (),
    title: str = "",
) -> str:
    """A small multi-series ASCII line chart.

    ``marks`` are x positions (in original series coordinates) to flag on the
    axis -- used by the demos to show where the world changed.
    """
    glyphs = "*o+x#"
    names = list(series)
    n_raw = max(len(series[k]) for k in names)
    data = {k: _resample(list(series[k]), width) for k in names}
    plot_w = max(len(v) for v in data.values())

    flat = [v for k in names for v in data[k] if math.isfinite(v)]
    if not flat:
        return "(no data)"
    if log:
        floor = min(v for v in flat if v > 0) if any(v > 0 for v in flat) else 1e-12
        data = {k: [math.log10(max(v, floor)) for v in data[k]] for k in names}
        flat = [v for k in names for v in data[k]]
    lo, hi = min(flat), max(flat)
    if hi - lo < 1e-12:
        hi = lo + 1.0

    grid = [[" "] * plot_w for _ in range(height)]
    for gi, name in enumerate(names):
        g = glyphs[gi % len(glyphs)]
        for x, v in enumerate(data[name]):
            y = int((hi - v) / (hi - lo) * (height - 1))
            grid[min(height - 1, max(0, y))][x] = g

    def label(v: float) -> str:
        if log:
            v = 10**v
        return f"{v:>9.3g}"

    lines = []
    if title:
        lines.append(title)
    for row in range(height):
        v = hi - (hi - lo) * row / (height - 1)
        lines.append(f"{label(v)} |" + "".join(grid[row]))
    lines.append(" " * 9 + " +" + "-" * plot_w)

    if marks and n_raw > 1:
        axis = [" "] * plot_w
        for m in marks:
            x = int(m / n_raw * plot_w)
            if 0 <= x < plot_w:
                axis[x] = "^"
        lines.append(" " * 9 + "  " + "".join(axis) + "   world changed")
    legend = "   ".join(f"{glyphs[i % len(glyphs)]} {n}" for i, n in enumerate(names))
    lines.append(" " * 11 + legend)
    return "\n".join(lines)
