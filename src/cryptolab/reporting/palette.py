"""Colour tokens for the static reports.

Values come from the validated reference palette. The categorical pair used for the two-series
charts (net vs gross) passes every gate in both modes: worst adjacent CVD ΔE 24.7 light / 26.8 dark
against an ≥8 target, normal-vision ΔE 33.6 / 31.8 against an ≥15 floor.

Colours are emitted as CSS custom properties so light and dark are a token swap rather than two
sets of hard-coded marks. Nothing in `charts.py` names a hex directly.
"""

from __future__ import annotations

from typing import Final

# Chart surfaces and ink.
LIGHT: Final[dict[str, str]] = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink-2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series-1": "#2a78d6",
    "series-2": "#eb6834",
    "pos": "#2a78d6",
    "neg": "#d03b3b",
    "mid": "#f0efec",
}

DARK: Final[dict[str, str]] = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "ink": "#ffffff",
    "ink-2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series-1": "#3987e5",
    "series-2": "#d95926",
    "pos": "#3987e5",
    "neg": "#e66767",
    "mid": "#383835",
}

# Status colours are fixed, never themed, and never reused as a series colour.
# Each is paired with an icon and a label in the templates: hue never carries meaning alone.
STATUS: Final[dict[str, str]] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Verdict → (status role, icon, human label). `killed` and `candidate` are both failures;
# they are distinguished because §11.1 says a killed candidate must not be tuned further.
VERDICT_STATUS: Final[dict[str, tuple[str, str, str]]] = {
    "validated": ("good", "✓", "PASS"),
    "candidate": ("warning", "!", "FAIL"),
    "killed": ("critical", "✕", "KILLED"),
}


def css_variables() -> str:
    """Render the light/dark token blocks, with the theme toggle winning in both directions."""
    light = "\n".join(f"    --{k}: {v};" for k, v in LIGHT.items())
    dark = "\n".join(f"    --{k}: {v};" for k, v in DARK.items())
    status = "\n".join(f"    --status-{k}: {v};" for k, v in STATUS.items())
    return f""":root {{
    color-scheme: light;
{light}
{status}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
{dark}
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
{dark}
  }}"""
