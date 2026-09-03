"""Inline-SVG chart builders for the static reports.

**Deviation from SPEC.md §3, stated deliberately.** The spec names `matplotlib` for reporting.
These charts are hand-built SVG instead, because a rasterised chart cannot do either of the two
things these reports need: colours that follow the reader's light/dark theme, and a hover layer.
An embedded PNG bakes one theme's colours into pixels. The §12 requirement is on the *content* of
a report, and that is unchanged; only the plotting mechanism differs. matplotlib remains a
dependency for anyone exporting figures separately.

Every mark takes its colour from a CSS custom property, so no hex appears here.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass

# Mark specs from the visualization guidance: 2px lines, ≥8px hover targets, 4px rounded
# data-ends, recessive grid, 2px surface gap between adjacent fills.
LINE_WIDTH = 2.0
BAR_RADIUS = 4.0
BAR_GAP = 2.0


@dataclass(frozen=True, slots=True)
class Box:
    """Plot geometry in user units."""

    width: float = 720.0
    height: float = 260.0
    left: float = 56.0
    right: float = 16.0
    top: float = 16.0
    bottom: float = 34.0

    @property
    def inner_w(self) -> float:
        return self.width - self.left - self.right

    @property
    def inner_h(self) -> float:
        return self.height - self.top - self.bottom


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value: float, *, unit: str = "", decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}{unit}"


def _nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Round tick values spanning [lo, hi], snapped to a 1-2-5 step.

    The naive `round(range / count)` approach collapses to a single repeated label whenever the
    span is large relative to its magnitude, which is exactly the equity-curve case.
    """
    if not (hi > lo) or not all(map(math.isfinite, (lo, hi))):
        return [lo]
    raw = (hi - lo) / max(count, 1)
    power = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10.0**power
    step = next((m * base for m in (1, 2, 2.5, 5, 10) if raw <= m * base), 10 * base)
    start = math.ceil(lo / step) * step
    ticks: list[float] = []
    value = start
    while value <= hi + step * 1e-9 and len(ticks) <= count + 2:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks or [lo, hi]


def _compact(value: float, unit: str = "") -> str:
    """Axis-label formatting that fits the gutter: 25k, 1.2M, -0.75."""
    magnitude = abs(value)
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if magnitude >= cutoff:
            scaled = value / cutoff
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}{unit}"
    if magnitude >= 100:
        return f"{value:,.0f}{unit}"
    if magnitude >= 1:
        return f"{value:,.2f}".rstrip("0").rstrip(".") + unit
    return f"{value:.2f}{unit}"


def _svg_open(box: Box, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {box.width:.0f} {box.height:.0f}" class="chart" '
        f'role="img" aria-label="{_esc(label)}" preserveAspectRatio="xMidYMid meet">'
    )


def _y_axis(box: Box, lo: float, hi: float, *, unit: str = "", log: bool = False) -> str:
    """Recessive gridlines with muted labels. Never a heavy grid.

    On a log axis `lo`/`hi` are exponents; ticks are placed on decade boundaries and labelled in
    the original units, so the reader never has to mentally exponentiate.
    """
    parts = []
    ticks = [e for e in range(math.floor(lo), math.ceil(hi) + 1)] if log else _nice_ticks(lo, hi)
    for tick in ticks:
        y = box.top + box.inner_h * (1 - (tick - lo) / (hi - lo or 1))
        shown = 10.0**tick if log else tick
        parts.append(
            f'<line x1="{box.left}" y1="{y:.1f}" x2="{box.width - box.right}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{box.left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{_compact(shown, unit)}</text>"
        )
    return "".join(parts)


def line_chart(
    series: Sequence[tuple[str, Sequence[float], str]],
    x_labels: Sequence[str],
    *,
    label: str,
    box: Box | None = None,
    unit: str = "",
    log: bool = False,
) -> str:
    """Multi-series line chart with a hover layer.

    `series` is (name, values, css-var-name). Two series is the intended case — net and gross
    equity — and both are direct-labelled at their right end so identity never rests on colour.

    `log` puts the value axis on a log scale, which is the correct form for a compounding equity
    curve: on a linear axis an early multiple flattens every later move into a straight line, and
    the reader cannot see what happened after it. Only positive values are plotted on a log axis;
    a bankrupt path falls back to linear.
    """
    box = box or Box()
    flat = [v for _, values, _ in series for v in values]
    if not flat:
        return f'<svg class="chart" role="img" aria-label="{_esc(label)}"></svg>'
    log = log and min(flat) > 0
    n = max(len(x_labels), 1)

    if log:
        lo_raw, hi_raw = math.log10(min(flat)), math.log10(max(flat))
        pad = (hi_raw - lo_raw) * 0.08 or 0.1
        lo, hi = lo_raw - pad, hi_raw + pad
    else:
        lo, hi = min(flat), max(flat)
        pad = (hi - lo) * 0.08 or abs(hi) * 0.08 or 1.0
        lo, hi = lo - pad, hi + pad

    def x_at(i: int) -> float:
        return box.left + (box.inner_w * i / max(n - 1, 1))

    def y_at(v: float) -> float:
        scaled = math.log10(v) if log and v > 0 else v
        return box.top + box.inner_h * (1 - (scaled - lo) / (hi - lo or 1))

    parts = [_svg_open(box, label), _y_axis(box, lo, hi, unit=unit, log=log)]

    # Direct-label each line at its right end. Where the endpoints are close enough that the
    # labels would overlap, push them apart vertically — identity must stay readable.
    ends = sorted(((values[-1], name) for name, values, _ in series if values), key=lambda t: -t[0])
    label_y: dict[str, float] = {}
    last = -1e9
    for value, name in ends:
        y = max(y_at(value) - 8, box.top + 10)
        if abs(y - last) < 14:
            y = last + 14
        label_y[name] = y
        last = y

    for name, values, var in series:
        pts = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="var(--{var})" '
            f'stroke-width="{LINE_WIDTH}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if values:
            parts.append(
                f'<text x="{x_at(len(values) - 1) - 4:.1f}" y="{label_y[name]:.1f}" '
                f'class="series-label" text-anchor="end">{_esc(name)}</text>'
            )

    # Hover layer: one full-height band per sample, ≥8px wide, carrying a native tooltip.
    band = max(box.inner_w / n, 8.0)
    for i, xl in enumerate(x_labels):
        readout = " · ".join(
            f"{name} {_fmt(values[i], unit=unit)}" for name, values, _ in series if i < len(values)
        )
        parts.append(
            f'<rect x="{x_at(i) - band / 2:.1f}" y="{box.top}" width="{band:.1f}" '
            f'height="{box.inner_h}" class="hover-band"><title>{_esc(xl)} — {_esc(readout)}</title>'
            f"</rect>"
        )

    if log:
        parts.append(f'<text x="{box.left - 8}" y="{box.top - 4}" class="tick" text-anchor="end">log</text>')
    parts.append(
        f'<line x1="{box.left}" y1="{box.top + box.inner_h}" x2="{box.width - box.right}" '
        f'y2="{box.top + box.inner_h}" class="axis"/>'
    )
    if x_labels:
        parts.append(
            f'<text x="{box.left}" y="{box.height - 8}" class="tick">{_esc(x_labels[0])}</text>'
            f'<text x="{box.width - box.right}" y="{box.height - 8}" class="tick" '
            f'text-anchor="end">{_esc(x_labels[-1])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def diverging_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    label: str,
    box: Box | None = None,
    unit: str = "",
    reference: float | None = None,
) -> str:
    """Bars around a zero baseline — the form for polarity (per-fold Sharpe).

    Positive and negative take the diverging pair, with a neutral zero line. Values are direct-
    labelled, so the sign is never carried by hue alone.
    """
    box = box or Box(height=240.0)
    if not values:
        return f'<svg class="chart" role="img" aria-label="{_esc(label)}"></svg>'
    lo, hi = min(*values, 0.0), max(*values, 0.0)
    pad = (hi - lo) * 0.15 or 1.0
    lo, hi = lo - pad, hi + pad
    n = len(values)
    slot = box.inner_w / n
    bar_w = max(slot - BAR_GAP * 2, 3.0)

    def y_at(v: float) -> float:
        return box.top + box.inner_h * (1 - (v - lo) / (hi - lo or 1))

    zero_y = y_at(0.0)
    parts = [_svg_open(box, label), _y_axis(box, lo, hi, unit=unit)]

    for i, (lbl, value) in enumerate(zip(labels, values, strict=True)):
        x = box.left + slot * i + BAR_GAP
        top = min(y_at(value), zero_y)
        height = max(abs(y_at(value) - zero_y), 1.0)
        var = "pos" if value >= 0 else "neg"
        parts.append(
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{height:.1f}" '
            f'rx="{min(BAR_RADIUS, bar_w / 2):.1f}" fill="var(--{var})">'
            f"<title>{_esc(lbl)} — {_fmt(value, unit=unit)}</title></rect>"
        )
        anchor_y = top - 6 if value >= 0 else top + height + 14
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{anchor_y:.1f}" class="bar-value" '
            f'text-anchor="middle">{_fmt(value, unit=unit, decimals=1)}</text>'
        )

    if reference is not None and lo <= reference <= hi:
        ry = y_at(reference)
        parts.append(
            f'<line x1="{box.left}" y1="{ry:.1f}" x2="{box.width - box.right}" y2="{ry:.1f}" '
            f'class="reference"/>'
            f'<text x="{box.width - box.right}" y="{ry - 6:.1f}" class="reference-label" '
            f'text-anchor="end">gate {_fmt(reference, decimals=2)}</text>'
        )

    parts.append(
        f'<line x1="{box.left}" y1="{zero_y:.1f}" x2="{box.width - box.right}" '
        f'y2="{zero_y:.1f}" class="axis"/></svg>'
    )
    return "".join(parts)


def ordinal_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    label: str,
    box: Box | None = None,
    unit: str = "",
    highlight: str | None = None,
) -> str:
    """Horizontal bars for magnitude across an ordered set (the four cost regimes).

    One hue; the headline regime is marked with a ring rather than a second colour, so the
    emphasis does not read as a different series.
    """
    # `right` reserves the gutter the trailing value labels are drawn into; without it they
    # render past the viewBox and get clipped.
    box = box or Box(height=40.0 + 30.0 * len(values), left=104.0, right=64.0, bottom=26.0)
    if not values:
        return f'<svg class="chart" role="img" aria-label="{_esc(label)}"></svg>'
    lo, hi = min(*values, 0.0), max(*values, 0.0)
    span = hi - lo or 1.0
    row = box.inner_h / len(values)
    bar_h = max(row - BAR_GAP * 2, 8.0)

    def x_at(v: float) -> float:
        return box.left + box.inner_w * (v - lo) / span

    zero_x = x_at(0.0)
    parts = [_svg_open(box, label)]
    for i, (lbl, value) in enumerate(zip(labels, values, strict=True)):
        y = box.top + row * i + BAR_GAP
        x = min(x_at(value), zero_x)
        width = max(abs(x_at(value) - zero_x), 1.0)
        cls = "bar-fill emphasis" if lbl == highlight else "bar-fill"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{bar_h:.1f}" '
            f'rx="{min(BAR_RADIUS, bar_h / 2):.1f}" class="{cls}">'
            f"<title>{_esc(lbl)} — {_fmt(value, unit=unit)}</title></rect>"
            f'<text x="{box.left - 8}" y="{y + bar_h / 2 + 4:.1f}" class="tick" '
            f'text-anchor="end">{_esc(lbl)}</text>'
            f'<text x="{x + width + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}" class="bar-value">'
            f"{_fmt(value, unit=unit)}</text>"
        )
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{box.top}" x2="{zero_x:.1f}" '
        f'y2="{box.top + box.inner_h:.1f}" class="axis"/></svg>'
    )
    return "".join(parts)


def gate_bar(value: float, threshold: float, *, passed: bool, higher_is_better: bool) -> str:
    """A one-line meter showing where a metric sits against its §11 gate."""
    scale = max(abs(value), abs(threshold)) * 1.35 or 1.0
    pos = min(max(abs(value) / scale, 0.0), 1.0) * 100
    gate = min(max(abs(threshold) / scale, 0.0), 1.0) * 100
    state = "good" if passed else "critical"
    direction = "≥" if higher_is_better else "≤"
    return (
        f'<span class="meter" role="img" aria-label="{_fmt(value)} vs gate {direction} '
        f'{_fmt(threshold)}">'
        f'<span class="meter-fill" data-state="{state}" style="width:{pos:.1f}%"></span>'
        f'<span class="meter-gate" style="left:{gate:.1f}%"></span></span>'
    )
