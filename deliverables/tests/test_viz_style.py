"""Tests for shared chart styling (src/viz_style.py)."""

import matplotlib as mpl

import viz_style
from viz_style import SERIES, apply_style


def test_palette_are_distinct_hex_colors():
    colors = [viz_style.BLUE, viz_style.AQUA, viz_style.YELLOW, viz_style.RED]
    assert all(c.startswith("#") and len(c) == 7 for c in colors)
    assert len(set(colors)) == 4  # no accidental duplicates


def test_series_is_first_three_categorical_slots():
    assert SERIES == [viz_style.BLUE, viz_style.AQUA, viz_style.YELLOW]


def test_apply_style_sets_rcparams():
    apply_style()
    assert mpl.rcParams["axes.grid"] is True
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["figure.facecolor"] == viz_style.SURFACE
