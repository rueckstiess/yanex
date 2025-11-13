"""Tests for visualization styles."""

import pytest

from yanex.results.viz.styles import get_color_palette, get_plot_style


class TestColorPalette:
    """Tests for color palette generation."""

    def test_get_palette_small(self):
        """Test getting small color palette."""
        colors = get_color_palette(5, colorblind_safe=True)
        assert len(colors) == 5
        assert all(c.startswith("#") for c in colors)

    def test_get_palette_medium(self):
        """Test getting medium color palette."""
        colors = get_color_palette(15, colorblind_safe=False)
        assert len(colors) == 15

    def test_get_palette_large_with_warning(self):
        """Test getting large palette issues warning."""
        with pytest.warns(UserWarning, match="25 unique labels"):
            colors = get_color_palette(25)
        assert len(colors) == 25

    def test_get_palette_zero(self):
        """Test edge case with zero colors."""
        colors = get_color_palette(0)
        assert colors == []

    def test_colorblind_safe_palette(self):
        """Test colorblind-safe palette is different."""
        safe = get_color_palette(5, colorblind_safe=True)
        unsafe = get_color_palette(5, colorblind_safe=False)
        # Should use different palettes
        assert safe != unsafe


class TestPlotStyle:
    """Tests for plot styling."""

    def test_get_plot_style(self):
        """Test getting plot style dictionary."""
        style = get_plot_style()
        assert isinstance(style, dict)
        assert "figure.facecolor" in style
        assert style["figure.facecolor"] == "white"
        assert "axes.grid" in style
        assert style["axes.grid"] is True

    def test_style_has_required_params(self):
        """Test style has all required matplotlib params."""
        style = get_plot_style()
        required = [
            "figure.facecolor",
            "axes.facecolor",
            "axes.grid",
            "grid.color",
            "font.size",
        ]
        for param in required:
            assert param in style
