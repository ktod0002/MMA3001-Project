"""Train a small RGB-D HRNet for predicting a 2D mask."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Global Variables
DATA_DIR = Path("Dataset")
OUTPUT_DIR = Path("training_run")
IMAGE_SIZE = 256
CROP_SIZE = 480
BATCH_SIZE = 2
EPOCHS = 200
LEARNING_RATE = 2e-4
SEED = 42
MAX_DX = 40.0
MAX_DY = 4.0

def find_samples(data_dir: Path) -> list[tuple[str, str, str]]:
    """Match RGB, depth, and mask files by sample ID.

    Args:
        data_dir: Folder containing the dataset PNG files.

    Returns:
        Tuples containing matching RGB, depth, and mask paths.
    """
    def index(pattern: str) -> dict[str, Path]:
        return {path.name.split("_")[0]: path for path in data_dir.glob(pattern)}

    # Filters data to ensure that all required images are available
    rgb = index("*Summer_RGB*.png")
    depth = index("*Summer_D*.png")
    mask = index("*Summer_GT*.png")
    if not rgb or set(rgb) != set(depth) or set(rgb) != set(mask):
        raise ValueError("Each sample must have matching RGB, depth, and mask files.")
    return [(str(rgb[key]), str(depth[key]), str(mask[key])) for key in sorted(rgb)]

def split_samples(samples: list[tuple[str, str, str]]) -> tuple[list, list]:
    """Randomly divide samples into 80% training and 20% validation.

    Args:
        samples: Complete dataset samples.

    Returns:
        Training and validation sample lists.
    """
    if len(samples) < 2:
        raise ValueError("At least two samples are required.")
    samples = [samples[i] for i in np.random.default_rng(SEED).permutation(len(samples))]
    validation_size = max(1, round(len(samples) * 0.2))
    return samples[validation_size:], samples[:validation_size]

def center_crop(image: tf.Tensor) -> tf.Tensor:
    """Take a square crop from the centre of an image.

    Args:
        image: Input image.

    Returns:
        Image cropped to ``CROP_SIZE``.
    """
    shape = tf.shape(image)
    y, x = (shape[0] - CROP_SIZE) // 2, (shape[1] - CROP_SIZE) // 2
    return tf.image.crop_to_bounding_box(image, y, x, CROP_SIZE, CROP_SIZE)

def load_sample(rgb_path: tf.Tensor, depth_path: tf.Tensor, mask_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Load and prepare one RGB-D image and binary mask.

    Args:
        rgb_path: RGB image path.
        depth_path: Depth image path.
        mask_path: Ground-truth mask path.

    Returns:
        Four-channel RGB-D image and one-channel mask.
    """
    # Center crop the RGB and Depth images and the normalise the image.
    rgb = tf.image.decode_png(tf.io.read_file(rgb_path), channels=3)
    depth = tf.image.decode_png(tf.io.read_file(depth_path), channels=1)
    mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
    rgb = tf.image.resize(center_crop(rgb), (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    depth = tf.image.resize(center_crop(depth), (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    image = tf.concat([tf.cast(rgb, tf.float32) / 255.0, tf.cast(depth, tf.float32) / 255.0], axis=-1)

    return image, tf.cast(mask > 127, tf.float32)

def make_dataset(samples: list[tuple[str, str, str]], training: bool) -> tf.data.Dataset:
    """Create a batched TensorFlow dataset.

    Args:
        samples: RGB, depth, and mask path tuples.
        training: Whether to shuffle the samples.

    Returns:
        Batched and prefetched dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(tuple(zip(*samples)))
    if training:
        dataset = dataset.shuffle(len(samples), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def normalise(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Apply group normalisation.

    Args:
        x: Input feature map.
        filters: Number of feature channels.

    Returns:
        Normalised feature map.
    """
    groups = next(group for group in (8, 4, 2, 1) if filters % group == 0)
    return layers.GroupNormalization(groups=groups, epsilon=1e-5)(x)

def convolution(x: tf.Tensor, filters: int, kernel: int = 3, stride: int = 1, activate: bool = True) -> tf.Tensor:
    """Apply convolution and group normalisation.

    Args:
        x: Input feature map.
        filters: Output channels.
        kernel: Convolution kernel size.
        stride: Convolution stride.
        activate: Whether to apply a Swish activation.

    Returns:
        Processed feature map.
    """
    x = layers.Conv2D(filters, kernel, stride, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = normalise(x, filters)
    return layers.Activation("swish")(x) if activate else x

def residual_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Apply a two-convolution residual block.

    Args:
        x: Input feature map.
        filters: Output channels.

    Returns:
        Residual feature map.
    """
    shortcut = x
    x = convolution(x, filters)
    x = convolution(x, filters, activate=False)
    if shortcut.shape[-1] != filters:
        shortcut = convolution(shortcut, filters, kernel=1, activate=False)
    return layers.Activation("swish")(layers.Add()([shortcut, x]))

def change_resolution(x: tf.Tensor, source: int, target: int, filters: int) -> tf.Tensor:
    """Convert one HRNet branch to another branch's resolution.

    Args:
        x: Source feature map.
        source: Source resolution level.
        target: Target resolution level.
        filters: Target channel count.

    Returns:
        Feature map at the requested resolution.
    """
    if source == target:
        return x
    if source > target:
        x = convolution(x, filters, kernel=1, activate=False)
        return layers.UpSampling2D(2 ** (source - target), interpolation="bilinear")(x)

    steps = target - source
    for step in range(steps):
        x = convolution(x, filters, stride=2, activate=step < steps - 1)
    return x

def hr_module(branches: list[tf.Tensor], channels: list[int]) -> list[tf.Tensor]:
    """Process each HRNet branch and exchange information between them.

    Args:
        branches: Feature maps from highest to lowest resolution.
        channels: Channel count for each branch.

    Returns:
        Processed and fused feature maps.
    """
    # Apply two residual blocks to each resolution branch separately.
    processed = []
    for branch, filters in zip(branches, channels):
        branch = residual_block(branch, filters)
        processed.append(residual_block(branch, filters))

    # Build one fused output for each target resolution.
    fused = []
    for target, filters in enumerate(channels):
        # Resize every processed branch to match the current target branch.
        inputs = [
            change_resolution(branch, source, target, filters)
            for source, branch in enumerate(processed)
        ]

        # Add the resized features together, then apply the activation.
        x = inputs[0] if len(inputs) == 1 else layers.Add()(inputs)
        fused.append(layers.Activation("swish")(x))
    return fused


def add_branch(branches: list[tf.Tensor], filters: int) -> list[tf.Tensor]:
    """Add a new lower-resolution branch.

    Args:
        branches: Existing HRNet branches.
        filters: Channels in the new branch.

    Returns:
        Existing branches followed by the new branch.
    """
    return [*branches, convolution(branches[-1], filters, stride=2)]

def build_model() -> tf.keras.Model:
    """Build the four-stage RGB-D HRNet.

    Args:
        None.

    Returns:
        HRNet segmentation model.
    """
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 4))
    rgb = convolution(inputs[..., :3], 18)
    depth = convolution(inputs[..., 3:], 6)
    high = layers.Concatenate()([rgb, depth])
    high = residual_block(residual_block(high, 24), 24)

    # Stage 2: full and half resolution.
    branches = hr_module(add_branch([high], 48), [24, 48])

    # Stage 3: add quarter resolution.
    branches = add_branch(branches, 96)
    branches = hr_module(hr_module(branches, [24, 48, 96]), [24, 48, 96])

    # Stage 4: add eighth resolution.
    branches = add_branch(branches, 192)
    branches = hr_module(hr_module(branches, [24, 48, 96, 192]), [24, 48, 96, 192])

    # Combie all of the resolutions.
    full_resolution = []
    for level, branch in enumerate(branches):
        branch = convolution(branch, 24, kernel=1)
        if level:
            branch = layers.UpSampling2D(2**level, interpolation="bilinear")(branch)
        full_resolution.append(branch)

    # Output the model architecture
    x = layers.Concatenate()(full_resolution)
    x = residual_block(residual_block(x, 48), 32)
    output = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, output, name="rgbd_hrnet")

def centroid(mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Calculate the centre of mass of each binary mask. This is required for the aligned binary dice.

    Args:
        mask: Binary mask batch.

    Returns:
        Vertical and horizontal centroid coordinates.
    """
    mask = tf.cast(mask, tf.float32)
    height, width = tf.shape(mask)[1], tf.shape(mask)[2]
    y = tf.cast(tf.range(height), tf.float32)[None, :, None, None]
    x = tf.cast(tf.range(width), tf.float32)[None, None, :, None]
    mass = tf.reduce_sum(mask, axis=(1, 2, 3))
    # Finds the center by summing up and dividing by the total.
    center_y = tf.math.divide_no_nan(tf.reduce_sum(mask * y, axis=(1, 2, 3)), mass)
    center_x = tf.math.divide_no_nan(tf.reduce_sum(mask * x, axis=(1, 2, 3)), mass)
    center_y = tf.where(mass > 0, center_y, tf.cast(height - 1, tf.float32) / 2.0)
    center_x = tf.where(mass > 0, center_x, tf.cast(width - 1, tf.float32) / 2.0)
    return center_y, center_x

def estimate_shift(y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Estimate mask alignment from the difference between their centroids.

    Args:
        y_true: Ground-truth masks.
        y_pred: Predicted probabilities.

    Returns:
        Bounded vertical and horizontal shifts.
    """
    prediction = tf.cast(tf.stop_gradient(y_pred) >= 0.5, tf.float32)
    true_y, true_x = centroid(y_true)
    pred_y, pred_x = centroid(prediction)
    dy = tf.clip_by_value(true_y - pred_y, -MAX_DY, MAX_DY)
    dx = tf.clip_by_value(true_x - pred_x, -MAX_DX, MAX_DX)
    return tf.stop_gradient(dy), tf.stop_gradient(dx)

def translate(images: tf.Tensor, dy: tf.Tensor, dx: tf.Tensor) -> tf.Tensor:
    """Translate every image in a batch by a different offset.

    Args:
        images: Image batch.
        dy: Vertical shifts.
        dx: Horizontal shifts.

    Returns:
        Translated images.
    """
    zero, one = tf.zeros_like(dx), tf.ones_like(dx)
    transforms = tf.stack([one, zero, -dx, zero, one, -dy, zero, zero], axis=1)
    return tf.raw_ops.ImageProjectiveTransformV3(images=images, transforms=transforms, output_shape=tf.shape(images)[1:3], interpolation="BILINEAR", fill_mode="CONSTANT", fill_value=0.0)

@tf.custom_gradient
def hard_threshold(probability: tf.Tensor):
    """Create a binary mask while allowing training gradients to pass.

    Args:
        probability: Predicted probabilities.

    Returns:
        Binary mask and its straight-through gradient.
    """
    binary = tf.cast(probability >= 0.5, probability.dtype)

    def gradient(upstream: tf.Tensor) -> tf.Tensor:
        return upstream

    return binary, gradient

def aligned_binary_dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Calculate binary Dice loss after centroid alignment.

    Args:
        y_true: Ground-truth masks.
        y_pred: Predicted probabilities.

    Returns:
        Mean aligned binary Dice loss.
    """
    y_true = tf.cast(y_true > 0.5, tf.float32)
    dy, dx = estimate_shift(y_true, y_pred)
    y_pred = hard_threshold(translate(y_pred, dy, dx))
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    total = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return tf.reduce_mean(1.0 - (2.0 * intersection + 1.0) / (total + 1.0))

def main() -> None:
    """Load the data, train HRNet, and evaluate its best validation weights.

    Args:
        None.

    Returns:
        None.
    """
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preprocess the data
    training_samples, validation_samples = split_samples(find_samples(DATA_DIR))
    training_data = make_dataset(training_samples, training=True)
    validation_data = make_dataset(validation_samples, training=False)

    # Construct the model architecture, loss function, and optimiser.
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=aligned_binary_dice)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(OUTPUT_DIR / "best.weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True)
    model.fit(training_data, validation_data=validation_data, epochs=EPOCHS, callbacks=[checkpoint])

    # Save the best model
    model.load_weights(OUTPUT_DIR / "best.weights.h5")
    model.save(OUTPUT_DIR / "complete_model.keras")
    print("Validation loss:", model.evaluate(validation_data))

if __name__ == "__main__":
    main()
