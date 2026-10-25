"""Straightforward tests for point generation and depth fitting."""

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pytest
import tensorflow as tf


# Prevent the application model from being loaded while the test module imports.
with patch.object(tf.keras.models, "load_model", return_value=Mock()):
    import points


def test_update_depth_handles_an_empty_point_list():
    depth = np.ones((10, 10), dtype=float)

    assert points.update_depth([], depth) == []


@pytest.mark.parametrize(
    ("normalised_depth", "max_depth", "expected_depth"),
    [
        (0.25, 4.0, 1.0),
        (0.50, 4.0, 2.0),
        (1.00, 3.0, 3.0),
    ],
)
def test_update_depth_scales_normalised_depth(
    normalised_depth, max_depth, expected_depth
):
    source_points = [
        (5, 5, 0, 1),
        (5, 10, 0, 1),
        (5, 15, 0, 1),
    ]
    depth = np.full((21, 21), normalised_depth, dtype=float)

    result = points.update_depth(
        source_points,
        depth,
        max_depth=max_depth,
        sections=3,
    )

    np.testing.assert_allclose([point[2] for point in result], expected_depth)


def test_update_depth_does_not_modify_the_input_points():
    source_points = [
        [5, 5, 0, 1],
        [5, 10, 0, 2],
        [5, 15, 0, 3],
    ]
    original = [point.copy() for point in source_points]
    depth = np.full((21, 21), 0.5, dtype=float)

    result = points.update_depth(source_points, depth)

    assert source_points == original
    assert all(isinstance(point, tuple) for point in result)


@pytest.mark.parametrize("invalid_depth", [0.0, np.nan])
def test_update_depth_keeps_existing_values_when_depth_is_invalid(invalid_depth):
    source_points = [(5, 5, 0.5, 1), (5, 10, 0.75, 1)]
    depth = np.full((15, 15), invalid_depth, dtype=float)

    result = points.update_depth(source_points, depth)

    assert result == source_points


def test_update_depth_keeps_a_single_point_unchanged():
    source_points = [(5, 5, 1.25, 2)]
    depth = np.ones((10, 10), dtype=float)

    assert points.update_depth(source_points, depth) == source_points


def test_update_depth_ignores_points_outside_the_depth_image():
    source_points = [(-10, 2, 0.5, 1), (-10, 7, 0.75, 1)]
    depth = np.ones((10, 10), dtype=float)

    assert points.update_depth(source_points, depth) == source_points


def test_update_depth_preserves_xy_coordinates_and_radii():
    source_points = [(4, 4, 0, 1.5), (4, 9, 0, 2.0), (4, 14, 0, 2.5)]
    depth = np.full((20, 20), 0.5, dtype=float)

    result = points.update_depth(source_points, depth)

    assert [(x, y, radius) for x, y, _, radius in result] == [
        (4, 4, 1.5),
        (4, 9, 2.0),
        (4, 14, 2.5),
    ]


def make_tree_mask():
    """Create a simple two-branch mask that joins into one trunk."""
    mask = np.zeros((256, 256), dtype=np.float32)

    # Two branch rows establish the split section.
    for row in (10, 11):
        mask[row, 20:30] = 1.0
        mask[row, 60:74] = 1.0

    # One trunk segment marks the junction and continues to the image bottom.
    mask[12:, 120:140] = 1.0
    return mask


if __name__ == "__main__":
    report_path = Path(__file__).with_name("points_test_report.txt")

    with report_path.open("w", encoding="utf-8") as report:
        with redirect_stdout(report), redirect_stderr(report):
            exit_code = pytest.main([__file__, "-v"])

    print(f"Test report saved to: {report_path}")
    raise SystemExit(exit_code)