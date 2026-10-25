"""Straightforward tests for the main RGB-D processing program."""

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pytest
import tensorflow as tf
import main as application

@pytest.mark.parametrize(
    ("image_shape", "crop_height", "crop_width", "expected_shape"),
    [
        ((6, 8, 3), 4, 4, (4, 4, 3)),
        ((10, 12, 1), 6, 8, (6, 8, 1)),
        ((5, 5, 3), 5, 5, (5, 5, 3)),
    ],
)
def test_center_crop_returns_the_requested_shape(
    image_shape, crop_height, crop_width, expected_shape
):
    image = tf.zeros(image_shape, dtype=tf.uint8)

    result = application.center_crop(image, crop_height, crop_width)

    assert tuple(result.shape) == expected_shape


def test_center_crop_selects_the_middle_of_the_image():
    image = tf.reshape(tf.range(24), (4, 6, 1))

    result = application.center_crop(image, 2, 2)

    expected = image[1:3, 2:4, :]
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_center_crop_rejects_a_crop_larger_than_the_image():
    image = tf.zeros((4, 4, 1), dtype=tf.uint8)

    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        application.center_crop(image, 6, 6)


def write_png(path, values):
    """Write an array as a PNG for a preprocessing test."""
    encoded = tf.image.encode_png(tf.convert_to_tensor(values, dtype=tf.uint8))
    path.write_bytes(encoded.numpy())

def test_load_sample_rejects_incorrect_image_sizes(tmp_path):
    rgb_path = tmp_path / "incorrect_rgb.png"
    depth_path = tmp_path / "correct_depth.png"

    # RGB width is incorrect; depth has the required 640x480 dimensions.
    write_png(rgb_path, np.zeros((480, 639, 3), dtype=np.uint8))
    write_png(depth_path, np.zeros((480, 640, 1), dtype=np.uint8))

    with pytest.raises(
        ValueError,
        match="RGB and depth images must both be 640x480 pixels",
    ):
        application.load_sample(str(rgb_path), str(depth_path))


def test_load_sample_raises_for_missing_files(tmp_path):
    with pytest.raises(tf.errors.NotFoundError):
        application.load_sample(
            str(tmp_path / "missing_rgb.png"),
            str(tmp_path / "missing_depth.png"),
        )


def test_predict_sample_runs_generation_export_and_display(tmp_path, monkeypatch):
    sample_id = "R01N01"
    rgb_path = tmp_path / "R01N01_Summer_RGB_001.png"
    depth_path = tmp_path / "R01N01_Summer_D_001.png"
    rgb_path.touch()
    depth_path.touch()

    rgb = tf.zeros((256, 256, 3), dtype=tf.float32)
    depth = tf.fill((256, 256, 1), 0.25)
    image = tf.concat([rgb, depth], axis=-1)[None, ...]

    load_sample = Mock(return_value=(image, rgb, depth))
    generated_points = [(10, 20, 1.5, 2.0)]
    generate_points = Mock(return_value=generated_points)
    model = Mock()
    create_model = Mock(return_value=model)

    monkeypatch.setattr(application, "load_sample", load_sample)
    monkeypatch.setattr(application, "generate_points", generate_points)
    monkeypatch.setattr(application, "create_model", create_model)
    monkeypatch.setattr(application.plt, "imshow", Mock())
    monkeypatch.setattr(application.plt, "show", Mock())
    monkeypatch.setattr(application.plt, "pause", Mock())

    result = application.predict_sample(tmp_path, sample_id)

    assert result is None
    load_sample.assert_called_once_with(str(rgb_path), str(depth_path))
    generate_points.assert_called_once()
    assert generate_points.call_args.args[0] is image
    np.testing.assert_allclose(generate_points.call_args.args[1], 0.25)
    create_model.assert_called_once_with(generated_points)
    model.export.assert_called_once_with("R01N01_tree_model.stl")
    model.show.assert_called_once_with()
    application.plt.imshow.assert_called_once_with(rgb)
    application.plt.show.assert_called_once_with(block=False)
    application.plt.pause.assert_called_once_with(0.5)


def test_predict_sample_raises_when_no_matching_sample_exists(tmp_path):
    with pytest.raises(IndexError):
        application.predict_sample(tmp_path, "UNKNOWN")


def test_main_uses_the_default_dataset_and_sample(monkeypatch):
    predict_sample = Mock()
    monkeypatch.setattr(application, "predict_sample", predict_sample)

    result = application.main()

    assert result is None
    predict_sample.assert_called_once_with(Path("Dataset"), "R01N01")


if __name__ == "__main__":
    report_path = Path(__file__).with_name("main_test_report.txt")

    with report_path.open("w", encoding="utf-8") as report:
        with redirect_stdout(report), redirect_stderr(report):
            exit_code = pytest.main([__file__, "-v"])

    print(f"Test report saved to: {report_path}")
    raise SystemExit(exit_code)