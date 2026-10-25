import gc
import statistics
import timeit
import os
import threading
import time
import psutil
import tensorflow as tf
import sys
from pathlib import Path
# Make project modules importable when the benchmark runs from this folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from points import generate_points
from modelling import create_model, create_sphere_model

def center_crop(image, crop_height, crop_width):
    """Crop an image around its center to the specified dimensions.

    Args:
        image: The input image tensor with shape
            ``(height, width, channels)``.
        crop_height: The height of the cropped image in pixels.
        crop_width: The width of the cropped image in pixels.

    Returns:
        The center-cropped image tensor with shape
        ``(crop_height, crop_width, channels)``.
    """
    # Derive offsets dynamically so the crop works with different source sizes.
    shape = tf.shape(image)

    height = shape[0]
    width = shape[1]

    # Keep discarded pixels balanced around the image centre.
    offset_height = (height - crop_height) // 2
    offset_width = (width - crop_width) // 2

    # Use TensorFlow cropping to preserve tensor shape and dtype handling.
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
        rgb_path: Path to the RGB PNG image.
        depth_path: Path to the depth PNG image.

    Returns:
        A tuple containing the batched four-channel input image, the normalized
        RGB tensor, and the normalized depth tensor. Their respective shapes
        are ``(1, 256, 256, 4)``, ``(256, 256, 3)``, and
        ``(256, 256, 1)``.
    """
    # Read RGB
    # Decode the modalities separately so their channel layouts stay explicit.
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_png(rgb, channels=3)

    # Read depth
    depth = tf.io.read_file(depth_path)
    depth = tf.image.decode_png(depth, channels=1)

    # Center crop
    # Focus both inputs on the same central field of view.
    rgb = center_crop(rgb, 480, 480)
    depth = center_crop(depth, 480, 480)

    # Resize
    # Standardise dimensions to match the model input contract.
    rgb = tf.image.resize(rgb, (256, 256))
    depth = tf.image.resize(depth, (256, 256))

    # Normalise
    # Place both modalities on a consistent scale for point generation.
    rgb = tf.cast(rgb, tf.float32) / 255.0
    depth = tf.cast(depth, tf.float32) / 255.0

    # RGB = 3 channels, depth = 1 channel, Result = 4 channels
    # Keep RGB appearance and geometry available in one model input.
    image = tf.concat([rgb, depth], axis=-1)
    image = tf.expand_dims(image, axis=0)

    # Retain the separate tensors for diagnostics alongside the batched input.
    return image, rgb, depth

def run_complete_pipeline():
    """Run point generation and tapered-model creation for the loaded image.

    Args:
        None.

    Returns:
        The mesh created from points generated using the globally loaded input
        image and its depth channel.
    """
    # Pass the geometric channel separately to the point extractor.
    depth = image[0, :, :, 3].numpy()
    points_list = generate_points(image, depth)
    # Measure the full path from network-ready input to completed mesh.
    model = create_model(points_list)

    return model

def measure_peak_memory(function, sample_interval=0.01):
    """Measure process memory while a function runs.

    Args:
        function: A callable that takes no arguments and performs the operation
            to measure.
        sample_interval: The interval, in seconds, between process-memory
            samples. The default checks memory every 10 milliseconds.

    Returns:
        A dictionary containing the function result, elapsed time in seconds,
        baseline and peak memory in megabytes, and peak and final memory
        increases in megabytes.
    """
    # Reduce unrelated retained objects before establishing the memory baseline.
    gc.collect()

    # Track total process memory so native-library allocations are included.
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = [baseline]
    # Coordinate shutdown without busy-waiting in the sampling thread.
    stop_monitoring = threading.Event()

    def monitor():
        """Sample process memory until monitoring is stopped.

        Args:
            None.

        Returns:
            None.
        """
        # Sample often enough to capture short-lived allocation peaks.
        while not stop_monitoring.wait(sample_interval):
            current = process.memory_info().rss
            peak[0] = max(peak[0], current)

    # Observe memory concurrently so measurement covers the entire function call.
    monitor_thread = threading.Thread(
        target=monitor,
        daemon=True,
    )

    # Start sampling before timing to avoid missing initial allocations.
    monitor_thread.start()
    start_time = time.perf_counter()

    # Guarantee monitor cleanup even if mesh generation raises an exception.
    try:
        result = function()
    finally:
        elapsed = time.perf_counter() - start_time
        stop_monitoring.set()
        monitor_thread.join()

    # Include allocations that remain after the monitored call returns.
    final_memory = process.memory_info().rss
    peak[0] = max(peak[0], final_memory)

    # Return the generated model with timing and memory metrics in one record.
    return {"result": result, "elapsed_seconds": elapsed, "baseline_mb": baseline / MB, "peak_mb": peak[0] / MB, "peak_increase_mb": (peak[0] - baseline) / MB, "final_increase_mb": (final_memory - baseline) / MB}

# Use one known scene to keep both modelling benchmarks directly comparable.
folder = Path("Dataset")
samples = []

# Find the test image R01N01
for rgb in folder.glob("R01N01*Summer_RGB*.png"):
    # R01N01_rgb_182736.png -> R01N01
    index = rgb.name.split("_")[0]

    # Pair RGB only when its corresponding depth input is unambiguous.
    d = list(folder.glob(f"{index}*Summer_D*.png"))

    if len(d) == 1:
        samples.append((str(rgb), str(d[0])))

# Preprocess once so timing measures geometry generation rather than file I/O.
image, rgb, depth = load_sample(samples[0][0], samples[0][1])

def run_complete_pipeline_sphere():
    """Run point generation and sphere-model creation for the loaded image.

    Args:
        None.

    Returns:
        The sphere-based mesh created from points generated using the globally
        loaded input image and its depth channel.
    """
    # Reuse identical extracted points when evaluating the sphere strategy.
    depth = image[0, :, :, 3].numpy()
    points_list = generate_points(image, depth)
    model = create_sphere_model(points_list)
    return model

def run_complete_pipeline_cylinder():
    """Run point generation and cylinder-model creation for the loaded image.

    Args:
        None.

    Returns:
        The cylinder-based mesh created from points generated using the
        globally loaded input image and its depth channel.
    """
    # Reuse the same input path for a fair cylinder-strategy comparison.
    depth = image[0, :, :, 3].numpy()
    points_list = generate_points(image, depth)
    model = create_model(points_list)
    return model

# Run once first
# Prime TensorFlow and modelling caches before collecting sphere timings.
run_complete_pipeline_sphere()
gc.collect()

# Isolate complete pipeline calls while keeping garbage collection enabled.
timer = timeit.Timer(stmt=run_complete_pipeline_sphere, setup=gc.enable)
times = timer.repeat(repeat=5, number=1)

# Show individual trials so runtime variability remains visible.
print("\nComplete pipeline timings (Spheres):")
for run_number, elapsed in enumerate(times, start=1):
    print(f"Run {run_number}: {elapsed:.4f} seconds")

print(f"\nMinimum: {min(times):.4f} seconds")
print(f"Median:  {statistics.median(times):.4f} seconds")
print(f"Mean:    {statistics.mean(times):.4f} seconds")
print(f"SD:      {statistics.stdev(times):.4f} seconds")

# Convert process byte counts into readable binary megabytes.
MB = 1024 * 1024

# Measure memory separately so sampling does not distort timed trials.
measurement = measure_peak_memory(run_complete_pipeline_sphere)

model = measurement["result"]

# Report retained and transient memory to distinguish allocation behaviour.
print(f"Execution time:       {measurement['elapsed_seconds']:.3f} seconds")
print(f"Memory before:        {measurement['baseline_mb']:.2f} MB")
print(f"Peak process memory:  {measurement['peak_mb']:.2f} MB")
print(f"Peak memory increase: {measurement['peak_increase_mb']:.2f} MB")
print(f"Final memory increase:{measurement['final_increase_mb']:.2f} MB")


# Optional warm-up: excludes initial library/cache overhead.
# Prime the alternative modelling path under the same warm-up conditions.
run_complete_pipeline_cylinder()
gc.collect()

# number=1 is appropriate because mesh generation is expensive.
# Mirror the sphere timing procedure for a like-for-like comparison.
timer = timeit.Timer(stmt=run_complete_pipeline_cylinder, setup=gc.enable)
times = timer.repeat(repeat=5, number=1)

# Preserve per-run results before summarising cylinder performance.
print("\nComplete pipeline timings (cylinder):")
for run_number, elapsed in enumerate(times, start=1):
    print(f"Run {run_number}: {elapsed:.4f} seconds")

print(f"\nMinimum: {min(times):.4f} seconds")
print(f"Median:  {statistics.median(times):.4f} seconds")
print(f"Mean:    {statistics.mean(times):.4f} seconds")
print(f"SD:      {statistics.stdev(times):.4f} seconds")

# Keep memory units consistent between the two modelling strategies.
MB = 1024 * 1024

# Capture the cylinder path peak independently from its timing trials.
measurement = measure_peak_memory(run_complete_pipeline_cylinder)

model = measurement["result"]

print(f"Execution time:       {measurement['elapsed_seconds']:.3f} seconds")
print(f"Memory before:        {measurement['baseline_mb']:.2f} MB")
print(f"Peak process memory:  {measurement['peak_mb']:.2f} MB")
print(f"Peak memory increase: {measurement['peak_increase_mb']:.2f} MB")
print(f"Final memory increase:{measurement['final_increase_mb']:.2f} MB")

# Display the final cylinder mesh for a visual sanity check.
model.show()
