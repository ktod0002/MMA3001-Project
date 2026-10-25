import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from points import generate_points
from modelling import create_model

def center_crop(image, crop_height, crop_width):
    """Crop an image around its center to the specified dimensions.

    Args:
        image: The input image tensor with shape
            "(height, width, channels)".
        crop_height: The height of the cropped image in pixels.
        crop_width: The width of the cropped image in pixels.

    Returns:
        The center-cropped image tensor with shape
        "(crop_height, crop_width, channels)".
    """
    shape = tf.shape(image)

    height = shape[0]
    width = shape[1]

    offset_height = (height - crop_height) // 2
    offset_width = (width - crop_width) // 2

    return tf.image.crop_to_bounding_box(
        image,
        offset_height,
        offset_width,
        crop_height,
        crop_width
    )

def load_sample(rgb_path, depth_path):
    """Load and preprocess paired RGB and depth images.

    Args:
        rgb_path: Path to the RGB PNG image. "640 x 480 px"
        depth_path: Path to the depth PNG image. "640 x 480 px"

    Returns:
        A tuple containing:
            image: A batched four-channel tensor combining the RGB and depth
                images, with shape "(1, 256, 256, 4)".
            rgb: The normalized RGB tensor with shape "(256, 256, 3)".
            depth: The normalized depth tensor with shape "(256, 256, 1)".
    """
    # Read RGB
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_png(rgb, channels=3)

    # Read depth
    depth = tf.io.read_file(depth_path)
    depth = tf.image.decode_png(depth, channels=1)

    # Reject wrong shape size
    if tuple(rgb.shape[:2]) != (480, 640) or tuple(depth.shape[:2]) != (480, 640):
        raise ValueError("RGB and depth images must both be 640x480 pixels")
    
    # Preprocess the data
    rgb = center_crop(rgb, 480, 480)
    depth = center_crop(depth, 480, 480)

    rgb = tf.image.resize(rgb, (256, 256))
    depth = tf.image.resize(depth, (256, 256))

    rgb = tf.cast(rgb, tf.float32) / 255.0
    depth = tf.cast(depth, tf.float32) / 255.0

    # RGB = 3 channels, depth = 1 channel, Result = 4 channels
    image = tf.concat([rgb, depth], axis=-1)
    image = tf.expand_dims(image, axis=0)

    return image, rgb, depth

def predict_sample(folder, id):
    """Generate and display a 3D tree model from one RGB-D sample.

    Finds the RGB and depth images matching the given sample ID, loads them,
    displays the RGB image, generates the tree points, and shows the resulting
    3D model.

    Args:
        folder: Path to the folder containing the RGB and depth images.
        id: Sample ID used to find the matching image files.

    Returns:
        None.
    """
    samples = []

    for rgb in folder.glob(f"{id}*Summer_RGB*.png"):
        # Extract the sample code from the filename.
        index = rgb.name.split("_")[0]
        d = list(folder.glob(f"{index}*Summer_D*.png"))

        # Ensure correct number of arguments.
        if len(d) == 1:
            samples.append((str(rgb), str(d[0])))

    image, rgb, depth = load_sample(samples[0][0], samples[0][1])

    # Show original image for validation
    plt.imshow(rgb)
    plt.show(block=False)
    plt.pause(0.5)

    # Generate and output the tree
    depth = image[0, :, :, 3].numpy()
    points_list = generate_points(image, depth)
    model = create_model(points_list)
    model.export(f"{id}_tree_model.stl")
    model.show()

def main():
    """Runs the main program.
        Args:
            None
        
        Returns:
            None
    """

    # Example use case. The robot should contain it's images following the databases naming code.
    folder = Path("Dataset")
    predict_sample(folder, "R01N01")

if __name__ == '__main__':
    main()
