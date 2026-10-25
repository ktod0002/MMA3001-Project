"""
Compare probability-centroid alignment against predefined shift searching.
Ground-truth masks are softened into probability masks and translated by known
amounts. Both alignment methods then attempt to recover the inverse translation.
"""
from pathlib import Path
import statistics
import timeit
import numpy as np
import tensorflow as tf

# Fix the workload so both methods are compared under repeatable conditions.
DATASET = Path("Dataset")
MASK_PATTERN = "*_Summer_GT*.png"
NUM_IMAGES = 5
TRIALS_PER_IMAGE = 4
APPLIED_MAX_X = 12
APPLIED_MAX_Y = 8
SEARCH_MAX_X = 40
SEARCH_MAX_Y = 16
# Separate warm-up and timed settings to reduce startup bias.
REPEATS = 7
ITERATIONS = 10
WARMUP = 3
SEED = 42

def translate_batch(images, dy, dx):
    """Translate each image without wrapping pixels around its boundaries.

    Args:
        images: A batch of images with shape ``(batch, height, width, channels)``.
        dy: The vertical translation for each image in pixels.
        dx: The horizontal translation for each image in pixels.

    Returns:
        A float32 tensor containing the translated images with the same shape
        as ``images``.
    """
    # Use one numeric type so transformation parameters behave consistently.
    images = tf.cast(images, tf.float32)
    dy, dx = tf.cast(dy, tf.float32), tf.cast(dx, tf.float32)
    zeros, ones = tf.zeros_like(dx), tf.ones_like(dx)

    # Encode inverse offsets because the projective operator maps output to input.
    transforms = tf.stack([ones, zeros, -dx, zeros, ones, -dy, zeros, zeros], axis=1)

    # Fill exposed boundaries with background rather than wrapping image content.
    return tf.raw_ops.ImageProjectiveTransformV3(images=images, transforms=transforms, output_shape=tf.shape(images)[1:3], interpolation="BILINEAR", fill_mode="CONSTANT", fill_value=0.0)

def load_mask(path):
    """Load, centre-crop, and resize one ground-truth mask.

    Args:
        path: Path to the PNG mask image.

    Returns:
        A float32 binary mask tensor with shape ``(256, 256, 1)``.
    """
    # Reduce every source mask to the same single-channel representation.
    mask = tf.image.decode_png(tf.io.read_file(str(path)), channels=1)
    height, width = int(tf.shape(mask)[0].numpy()), int(tf.shape(mask)[1].numpy())

    # Preserve ready-to-use masks while normalising all other source dimensions.
    if height != 256 or width != 256:
        crop_size = min(480, height, width)
        offset_y, offset_x = (height - crop_size) // 2, (width - crop_size) // 2

        mask = tf.image.crop_to_bounding_box(mask, offset_y, offset_x, crop_size, crop_size)
        mask = tf.image.resize(mask, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Enforce binary labels so overlap scores remain comparable across files.
    return tf.cast(mask > 127, tf.float32)

def create_test_cases(paths):
    """Create probability predictions with known two-pixel translations.

    Args:
        paths: Paths to the ground-truth mask images used to create test cases.

    Returns:
        A tuple containing the stacked ground-truth masks, stacked probability
        predictions, an array of applied ``(dy, dx)`` translations, and the
        source-mask labels.
    """
    # Keep case generation reproducible without sharing the selection RNG stream.
    generator = np.random.default_rng(SEED + 1)

    # Exclude the zero offset so every case requires a genuine correction.
    possible_shifts = np.array([
        (dy, dx)
        for dy in range(-APPLIED_MAX_Y, APPLIED_MAX_Y + 1, 2)
        for dx in range(-APPLIED_MAX_X, APPLIED_MAX_X + 1, 2)
        if dy != 0 or dx != 0
    ], dtype=np.float32)

    # Accumulate paired data and provenance in matching order.
    truths, predictions, applied_shifts, labels = [], [], [], []

    # Draw several distinct perturbations from each source mask.
    for path in paths:
        truth = load_mask(path)

        # Soft edges make the test mask resemble a probability prediction.
        probability = tf.nn.avg_pool2d(truth[None, ...], ksize=3, strides=1, padding="SAME")
        selected = generator.choice(len(possible_shifts), size=TRIALS_PER_IMAGE, replace=False)

        # Preserve the exact applied offset for later qualitative validation.
        for index in selected:
            dy, dx = possible_shifts[index]
            prediction = translate_batch(probability, tf.constant([dy], tf.float32), tf.constant([dx], tf.float32))[0]
            truths.append(truth)
            predictions.append(prediction)
            applied_shifts.append((dy, dx))
            labels.append(path.stem)

    # Batch the cases once so both methods receive identical inputs.
    return tf.stack(truths), tf.stack(predictions), np.array(applied_shifts), labels

def probability_centroid(mask):
    """Calculate the probability-weighted centre of every mask.

    Args:
        mask: A batch of probability masks with shape
            ``(batch, height, width, 1)``.

    Returns:
        A tuple containing one-dimensional tensors of vertical and horizontal
        centroid coordinates, respectively.
    """
    # Build broadcastable coordinates for a fully vectorised centroid calculation.
    height, width = tf.shape(mask)[1], tf.shape(mask)[2]

    rows = tf.reshape(tf.cast(tf.range(height), tf.float32), (1, -1, 1, 1))
    columns = tf.reshape(tf.cast(tf.range(width), tf.float32), (1, 1, -1, 1))

    # Guard empty masks to prevent undefined centroid values.
    mass = tf.reduce_sum(mask, axis=(1, 2, 3))
    mass = tf.maximum(mass, tf.keras.backend.epsilon())

    # Convert probability mass into subpixel spatial coordinates.
    centre_y = tf.reduce_sum(mask * rows, axis=(1, 2, 3)) / mass
    centre_x = tf.reduce_sum(mask * columns, axis=(1, 2, 3)) / mass

    return centre_y, centre_x

@tf.function
def centroid_alignment(truth, prediction):
    """Align masks using the difference between their centroids.

    Args:
        truth: A batch of ground-truth probability masks.
        prediction: A batch of predicted probability masks to align.

    Returns:
        A tuple containing the vertical and horizontal translations required
        to align each prediction with its ground-truth mask.
    """
    # Recover the inverse displacement directly from paired centroid locations.
    truth_y, truth_x = probability_centroid(truth)
    prediction_y, prediction_x = probability_centroid(prediction)

    return truth_y - prediction_y, truth_x - prediction_x

def candidate_dice(truth, prediction, candidates):
    """Calculate Dice scores for every candidate translation in one batch.

    Args:
        truth: A batch of ground-truth masks with shape
            ``(batch, height, width, 1)``.
        prediction: A batch of predicted masks with the same shape as ``truth``.
        candidates: Candidate ``(dy, dx)`` translations with shape
            ``(batch, candidate_count, 2)``.

    Returns:
        A tensor of Dice scores with shape ``(batch, candidate_count)``.
    """
    batch_size = tf.shape(prediction)[0]
    # Score every candidate for every image in one vectorised operation.
    candidate_count = tf.shape(candidates)[1]

    # Flatten image-candidate pairs to reuse the batch translation primitive.
    repeated = tf.repeat(prediction, repeats=candidate_count, axis=0)
    flat_candidates = tf.reshape(candidates, (-1, 2))

    shifted = translate_batch(repeated, flat_candidates[:, 0], flat_candidates[:, 1])
    # Restore the candidate axis before comparing shifts independently.
    shifted = tf.reshape(shifted, (batch_size, candidate_count, tf.shape(prediction)[1], tf.shape(prediction)[2], 1))

    # Broadcast each truth mask across all of its candidate translations.
    truth = truth[:, None, ...]
    intersection = tf.reduce_sum(truth * shifted, axis=(2, 3, 4))
    total = tf.reduce_sum(truth + shifted, axis=(2, 3, 4))

    # Smooth the ratio so empty regions do not cause division by zero.
    return (2 * intersection + 1) / (total + 1)

def create_predefined_alignment():
    """Create the original coarse-then-fine predefined shift search.

    Args:
        None.

    Returns:
        A compiled alignment callable that accepts ground-truth and predicted
        mask batches and returns their best vertical and horizontal shifts.
    """
    # Trade a broad inexpensive search for a narrow higher-resolution refinement.
    coarse_step, fine_step = 4, 2

    # Express full-resolution offsets in the coarse image coordinate system.
    coarse_candidates = tf.constant([
        (dy / coarse_step, dx / coarse_step)
        for dy in range(-SEARCH_MAX_Y, SEARCH_MAX_Y + 1, coarse_step)
        for dx in range(-SEARCH_MAX_X, SEARCH_MAX_X + 1, coarse_step)
    ], dtype=tf.float32)

    # Restrict refinement to the neighbourhood around the coarse winner.
    fine_offsets = tf.constant([
        (dy, dx)
        for dy in (-fine_step, 0, fine_step)
        for dx in (-fine_step, 0, fine_step)
    ], dtype=tf.float32)

    @tf.function
    def predefined_alignment(truth, prediction):
        """Search predefined coarse shifts and refine around the best result.

        Args:
            truth: A batch of ground-truth probability masks.
            prediction: A batch of predicted probability masks to align.

        Returns:
            A tuple containing the best vertical and horizontal translation
            for each predicted mask.
        """
        # Apply the same candidate grid independently to every batch member.
        batch_size = tf.shape(prediction)[0]

        # Downsample first to make the wide search computationally affordable.
        truth_64 = tf.image.resize(truth, (64, 64), method="area")
        prediction_64 = tf.image.resize(prediction, (64, 64), method="area")
        coarse = tf.tile(coarse_candidates[None, ...], (batch_size, 1, 1))

        # Promote only the strongest coarse candidate to the refinement stage.
        coarse_scores = candidate_dice(truth_64, prediction_64, coarse)
        coarse_index = tf.argmax(coarse_scores, axis=1, output_type=tf.int32)
        coarse_best = tf.gather(coarse, coarse_index, batch_dims=1) * coarse_step

        # Increase spatial detail before resolving the final local offset.
        truth_128 = tf.image.resize(truth, (128, 128), method="area")
        prediction_128 = tf.image.resize(prediction, (128, 128), method="area")
        fine_candidates = coarse_best[:, None, :] + fine_offsets[None, ...]

        # Keep refinement inside the range supported by the benchmark.
        fine_candidates = tf.stack([tf.clip_by_value(fine_candidates[..., 0], -SEARCH_MAX_Y, SEARCH_MAX_Y), tf.clip_by_value(fine_candidates[..., 1], -SEARCH_MAX_X, SEARCH_MAX_X)], axis=-1)

        # Select the best refined shift and return it in full-resolution pixels.
        fine_scores = candidate_dice(truth_128, prediction_128, fine_candidates / 2)
        fine_index = tf.argmax(fine_scores, axis=1, output_type=tf.int32)
        best = tf.gather(fine_candidates, fine_index, batch_dims=1)

        return best[:, 0], best[:, 1]

    return predefined_alignment

def benchmark(method, truth, prediction):
    """Measure mean, median, and best runtime using repeated timeit trials.

    Args:
        method: An alignment callable that accepts truth and prediction batches
            and returns vertical and horizontal shifts.
        truth: The batch of ground-truth masks passed to ``method``.
        prediction: The batch of predicted masks passed to ``method``.

    Returns:
        A dictionary containing the mean, median, and best runtime per
        iteration in milliseconds.
    """
    def timed_call():
        """Execute and synchronise one call to the alignment method.

        Args:
            None.

        Returns:
            A NumPy scalar formed by summing the predicted shifts. Converting
            the value to NumPy ensures asynchronous TensorFlow work completes.
        """
        dy, dx = method(truth, prediction)

        # Converting to NumPy waits for TensorFlow or GPU work to finish.
        return tf.reduce_sum(dy + dx).numpy()

    # Warm TensorFlow traces and caches before collecting measurements.
    for _ in range(WARMUP):
        timed_call()

    # Average within trials while retaining variation across repeated trials.
    totals = timeit.repeat(timed_call, repeat=REPEATS, number=ITERATIONS)
    milliseconds = [total * 1000 / ITERATIONS for total in totals]

    # Report complementary summaries instead of relying on one noisy timing.
    return {"mean": statistics.mean(milliseconds), "median": statistics.median(milliseconds), "best": min(milliseconds)}

def main():
    """Create test cases, benchmark both methods, and print the results.
    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: If fewer than ``NUM_IMAGES`` matching masks are found.
    """
    # Sort discovery results before seeded sampling for cross-platform repeatability.
    all_paths = sorted(DATASET.glob(MASK_PATTERN))

    # Fail early rather than benchmark an unintended smaller workload.
    if len(all_paths) < NUM_IMAGES:
        raise ValueError(f"Only {len(all_paths)} matching masks were found")

    # Select one stable subset for every alignment method.
    generator = np.random.default_rng(SEED)
    selected_indices = generator.choice(len(all_paths), size=NUM_IMAGES, replace=False)
    selected_paths = [all_paths[index] for index in selected_indices]

    # Reuse the exact same generated cases to keep the comparison fair.
    truth, prediction, applied_shifts, labels = create_test_cases(selected_paths)
    predefined_alignment = create_predefined_alignment()

    # Give both strategies a common interface for timing and reporting.
    methods = {"Probability centroid": centroid_alignment, "Predefined shifts": predefined_alignment}

    # Isolate timing summaries by method for a direct comparison.
    results = {name: benchmark(method, truth, prediction) for name, method in methods.items()}

    case_count = len(truth)

    # Present aggregate performance before inspecting individual predictions.
    print(f"\nRuntime - {case_count} trials each")
    print(f"{'Method':<22}{'Mean ms':>10}{'Median ms':>11}{'Best ms':>10}{'ms/case':>10}")
    print("-" * 63)

    for name, result in results.items():
        print(f"{name:<22}{result['mean']:>10.3f}{result['median']:>11.3f}", f"{result['best']:>10.3f}{result['median'] / case_count:>10.4f}")

    # Re-run both methods outside timing to expose representative outputs.
    centroid_dy, centroid_dx = centroid_alignment(truth, prediction)
    shifted_dy, shifted_dx = predefined_alignment(truth, prediction)

    print("\n\nTesting the two methods\n")
    print(f"{'Mask':<29}{'applied (x,y)':<17}{'Centroid Result':<18}{'Shifted Result'}")
    print("-" * 82)

    # Limit diagnostics so the comparison remains readable for larger batches.
    for index in range(min(10, case_count)):
        applied = f"({applied_shifts[index, 1]:.0f}, {applied_shifts[index, 0]:.0f})"
        centroid = f"({centroid_dx[index].numpy():.2f}, {centroid_dy[index].numpy():.2f})"
        shifted = f"({shifted_dx[index].numpy():.0f}, {shifted_dy[index].numpy():.0f})"

        print(f"{labels[index]:<29}{applied:<17}{centroid:<18}{shifted}")

# Avoid running the benchmark when this module is imported elsewhere.
if __name__ == "__main__":
    main()
