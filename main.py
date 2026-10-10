import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from points import generate_points
from modelling import create_model

def center_crop(image, crop_height, crop_width):
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
    # Read RGB
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_png(rgb, channels=3)

    # Read depth
    depth = tf.io.read_file(depth_path)
    depth = tf.image.decode_png(depth, channels=1)

    # Center crop
    rgb = center_crop(rgb, 480, 480)
    depth = center_crop(depth, 480, 480)

    # Resize
    rgb = tf.image.resize(rgb, (256, 256))
    depth = tf.image.resize(depth, (256, 256))

    # Normalise
    rgb = tf.cast(rgb, tf.float32) / 255.0
    depth = tf.cast(depth, tf.float32) / 255.0

    # RGB = 3 channels, depth = 1 channel, Result = 4 channels
    image = tf.concat([rgb, depth], axis=-1)
    image = tf.expand_dims(image, axis=0)

    return image, rgb, depth



folder = Path("Dataset")
samples = []

for rgb in folder.glob("R01N01*Summer_RGB*.png"):
    # R01N01_rgb_182736.png -> R01N01
    index = rgb.name.split("_")[0]

    d = list(folder.glob(f"{index}*Summer_D*.png"))

    if len(d) == 1:
        samples.append((str(rgb), str(d[0])))


image, rgb, depth = load_sample(samples[0][0], samples[0][1])

plt.imshow(rgb)
plt.show(block = False)
plt.pause(0.1)

depth = image[0, :, :, 3].numpy()
points_list = generate_points(image, depth)
model = create_model(points_list)
model.show()
