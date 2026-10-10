import numpy as np
import tensorflow as tf
from scipy.interpolate import UnivariateSpline

model = tf.keras.models.load_model(
    "best_model.keras",
    compile=False
)

def update_depth(points, depth_image, max_depth=4.0, sections=8):
    """
    Updates the depth values of 3D points using a depth image and spline interpolation.

    Separates points into the trunk, left branch, and right branch based on
    their positions. Each branch is divided into sections, with the furthest
    valid depth measurement selected from each section to reduce interference
    from leaves. Outliers are removed before fitting a smoothing spline to
    estimate the depth along each branch.

    Args:
        points (list[tuple]): List of points containing (x, y, z, radius).
        depth_image (np.ndarray): Normalised depth image used to estimate
            the depth of each point.
        max_depth (float): Maximum depth used to scale the image values.
            Defaults to 4.0.
        sections (int): Number of sections used to sample depth measurements
            along each branch. Defaults to 8.

    Returns:
        output (list[tuple]): Updated points containing (x, y, z, radius), with
            depth values replaced by the fitted spline predictions where
            sufficient valid measurements are available.
    """
    points = [list(p) for p in points]

    # Add each index of the points into the rows dictionary.
    rows = {}
    for i, (x, y, _, r) in enumerate(points):
        rows.setdefault(y, []).append(i)

    trunk, left, right = [], [], []

    # Seperates the points into either a trunk, left branch or right branch depending on the amount of points in each row.
    for row in rows.values():
        if len(row) == 1:
            trunk.append(row[0])

        elif len(row) == 2:
            row.sort(key=lambda i: points[i][0])  # sort by x
            left.append(row[0])
            right.append(row[1])

    def fit_branch(indices):
        # Filter data
        if len(indices) < 2:
            return

        # Sort from top to bottom
        indices.sort(key=lambda i: points[i][1])

        y = np.array([points[i][1] for i in indices], dtype=float)

        measured = np.array([
            depth_image[int(points[i][1]), int(points[i][0])] * max_depth
            for i in indices
        ])

        # Split into sections and take the furthest point in each section
        section_indices = np.array_split(np.arange(len(indices)), sections)

        fit_y = []
        fit_z = []

        for section in section_indices:
            if len(section) == 0:
                continue

            section = np.asarray(section).flatten()

            section_depths = measured[section]

            valid = section[
                np.isfinite(section_depths) &
                (section_depths > 0)
            ]

            if len(valid) == 0:
                continue

            # Furthest point is most likely the branch rather than a leaf
            j = valid[np.argmax(measured[valid])]

            fit_y.append(y[j])
            fit_z.append(measured[j])

        fit_y = np.array(fit_y)
        fit_z = np.array(fit_z)

        if len(fit_y) < 2:
            return

        # Remove obvious depth outliers
        median = np.median(fit_z)
        mae = np.median(np.abs(fit_z - median))

        if mae > 0:
            valid = np.abs(fit_z - median) < 3 * mae
            fit_y = fit_y[valid]
            fit_z = fit_z[valid]

        if len(fit_y) < 2:
            return

        # Fit spline
        k = min(3, len(fit_y) - 1)

        spline = UnivariateSpline(fit_y, fit_z, k=k, s=0.01 * len(fit_y))
        predicted = spline(y)

        # Update depth entry in original tuples
        for i, z in zip(indices, predicted):
            points[i][2] = float(z)

    # Fit all three parts separately
    fit_branch(trunk)
    fit_branch(left)
    fit_branch(right)

    output = [tuple(p) for p in points]
    return output

def generate_points(image, depth):
    """
    Generates 3D points representing the centre and radius of tree branches
    from a predicted 2D mask and depth image.

    Scans the binary mask horizontally to identify branch boundaries and
    calculate their centre positions and radii. Separates the trunk and
    branches, estimates their depth using spline interpolation, and adds
    intermediate points to create smooth connections between the branches
    and trunk.

    Args:
        image (np.ndarray): Input image used by the neural network to
            predict the tree skeleton mask.
        depth (np.ndarray): Normalised depth image used to estimate the
            depth coordinates of the generated points.

    Returns:
        list[tuple]: List of points containing (x, y, z, radius), defining
            the centre position and radius of each spherical section
            used to construct the 3D tree model.
    """
    # Generate the 2D mask and initialise variables.
    prediction = model.predict(image)
    mask = (prediction["mask"].squeeze() > 0.95).astype(int)
    points_list = []
    two_branches = False

    # Finds the center of every branch in horizontal slices.
    for row in range(255):
        start = "NaN"
        new_points = []

        for column in range(255):
            # Check for white pixel, if white wait until black then find the middle.
            if (mask[row, column] == 1) & (start == "NaN"):
                start = column
            elif (mask[row, column] == 0) & (start != "NaN"):
                middle = (start + column) / 2
                radius = (column - start) / 2
                new_points.append((middle, row, 0, radius))
                start = "NaN"
        if (len(new_points) == 2):
            two_branches = True

        if (two_branches == True) & (len(new_points) == 1):
            end_double_row = row
            final_double_index = len(points_list)
            break
        points_list = points_list + new_points

    radius_multiplier = 1

    for row in range(255, end_double_row, -1):
        start = "NaN"
        new_points = []

        for column in range(255):
            # Check for white pixel, if white wait until black then find the middle.
            if (mask[row, column] == 1) & (start == "NaN"):
                start = column

            elif (mask[row, column] == 0) & (start != "NaN"):
                middle = (start + column) / 2
                radius = (column - start) / 2 * radius_multiplier
                new_points.append((middle, row, 0, radius))
                start = "NaN"

        points_list = points_list + new_points

        if (row - radius < end_double_row):
            final_single_row = points_list[-1]
            break

    # Linear interpolation between the ends of the trunk and ends of the left and right branches.
    num_interpolations = 4
    points_list = update_depth(points_list, depth, 4, 8)
    final_double_row = (points_list[final_double_index - 2], points_list[final_double_index - 1])
    final_single_row = (points_list[-1])
    for branch in final_double_row:
        interpolations = np.linspace(branch, final_single_row, num_interpolations + 2)[1:-1]
        points_list.extend(map(tuple, interpolations.tolist()))

    return points_list
