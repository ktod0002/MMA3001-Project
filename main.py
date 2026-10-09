import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from points import generate_points
from modelling import create_model

folder = Path("Dataset")
samples = []

for rgb in folder.glob("R01N01*Summer_RGB*.png"):
    # R01N01_rgb_182736.png -> R01N01
    index = rgb.name.split("_")[0]

    d = list(folder.glob(f"{index}*Summer_D*.png"))

    if len(d) == 1:
        samples.append((str(rgb), str(d[0])))

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

image, rgb, depth = load_sample(samples[0][0], samples[0][1])

plt.imshow(rgb)
plt.show(block = False)
plt.pause(0.1)

depth = image[0, :, :, 3].numpy()
points_list = generate_points(image, depth)
model = create_model(points_list)
model.show()






















# model = tf.keras.models.load_model(
#     "best_model.keras",
#     compile=False
# )

# prediction = model.predict(image)

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(rgb), ax[0].set_title("RGB"), ax[0].axis("off")
# ax[1].imshow(prediction["mask"].squeeze() > 0.95), ax[1].set_title("Prediction"), ax[1].axis("off")
# plt.imshow, plt.show()

# mask = (prediction["mask"].squeeze() > 0.95).astype(int)

# points_list = []

# tolerance = 7
# two_branches = False

# for row in range(255):
#     start = "NaN"
#     new_points = []

#     for column in range(255):
#         # Check for white pixel, if white wait until black then find the middle.
#         if (mask[row, column] == 1) & (start == "NaN"):
#             start = column
#         elif (mask[row, column] == 0) & (start != "NaN"):
#             middle = (start + column) / 2
#             radius = (column - start) / 2 
#             new_points.append((column, row, 0, radius))
#             # points_list.append((column, row, 0, radius))
#             start = "NaN"
#     if (len(new_points) == 2):
#         two_branches = True

#     if (two_branches == True) & (len(new_points) == 1):
#         end_double_row = row
#         final_double_row = (points_list[-2], points_list[-1])
#         break

#     points_list = points_list + new_points
#     # mask[row, middle] = 0
#     previous_radius = radius

# radius_multiplier = 1

# for row in range(255, end_double_row, -1):
#     start = "NaN"
#     new_points = []

#     for column in range(255):
#         # Check for white pixel, if white wait until black then find the middle.
#         if (mask[row, column] == 1) & (start == "NaN"):
#             start = column

#         elif (mask[row, column] == 0) & (start != "NaN"):
#             middle = (start + column) / 2
#             radius = (column - start) / 2 * radius_multiplier
#             new_points.append((column, row, 0, radius))
#             start = "NaN"

#     points_list = points_list + new_points
#     previous_radius = radius

#     if (row - radius < end_double_row):
#         final_single_row = points_list[-1]
#         break

# print(final_double_row)
# print(final_single_row)

# # Linear interpolation between the ends of the single and ends of the double branches.

# num_interpolations = 4
# for branch in final_double_row:
#     interpolations = np.linspace(branch, final_single_row, num_interpolations + 2)[1:-1]
#     points_list.extend(map(tuple, interpolations.tolist()))

# print(points_list)