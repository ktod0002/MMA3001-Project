import numpy as np
import tensorflow as tf
from scipy.interpolate import PchipInterpolator

model = tf.keras.models.load_model("best_model.keras", compile=False)

def update_depth(points, depth_image, max_depth=4.0, sections=8, join_points=4):
    """Estimate and update point depths using a depth image. Points are grouped into a central trunk and side branches. 
    Local depth samples are fitted with shape-preserving interpolation, then blended at branch junctions to reduce discontinuities. 
    The input points are copied and are not modified directly.

    Args:
        points: An iterable of points represented as "(x, y, z, radius)"
            tuples.
        depth_image: A two-dimensional normalized depth array.
        max_depth: The maximum physical depth represented by the normalized
            depth image.
        sections: The maximum number of vertical sections used to fit each
            branch's depth curve.
        join_points: The maximum number of points corrected near each
            branch-to-trunk junction.

    Returns:
        A list of updated "(x, y, z, radius)" tuples with fitted and
        junction-corrected depth values.
    """
    points = [list(p) for p in points]
    h, w = depth_image.shape

    def sample(x, y):
        """Sample a robust depth estimate around an image coordinate.

        Args:
            x: The horizontal image coordinate.
            y: The vertical image coordinate.

        Returns:
            The 80th-percentile depth from the valid local pixels, scaled by
            "max_depth". Returns "numpy.nan" if the coordinate is outside
            the image or the local patch contains no valid depth values.
        """
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < w and 0 <= y < h):
            return np.nan

        # Small patch is more reliable than one pixel.
        patch = depth_image[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)] * max_depth
        patch = patch[np.isfinite(patch) & (patch > 0)]
        return np.percentile(patch, 80) if patch.size else np.nan

    # Group approximate image rows.
    rows = {}
    for i, (x, y, _, _) in enumerate(points):
        rows.setdefault(int(round(y)), []).append(i)

    trunk, left, right = [], [], []

    for row in rows.values():
        row.sort(key=lambda i: points[i][0])

        if len(row) == 1:
            trunk.append(row[0])
        else:
            left.append(row[0])
            right.append(row[-1])

    def fit(indices):
        """Fit a depth curve to a collection of point indices.

        Args:
            indices: Indices of points belonging to one branch or the trunk.

        Returns:
            The point indices sorted by vertical image position. When enough
            valid samples exist, the corresponding point depths are updated
            in place using a shape-preserving interpolated curve.
        """
        indices = sorted(indices, key=lambda i: points[i][1])
        if len(indices) < 2:
            return indices

        y = np.array([points[i][1] for i in indices])
        z = np.array([sample(points[i][0], points[i][1]) for i in indices])

        valid = np.flatnonzero(np.isfinite(z))
        if len(valid) < 2:
            return indices

        fit_y, fit_z = [], []

        for section in np.array_split(valid, min(sections, len(valid))):
            depths = z[section]

            # Favour farther branch pixels, but avoid using a noisy maximum.
            cutoff = np.percentile(depths, 75)
            selected = section[depths >= cutoff]

            fit_y.append(np.median(y[selected]))
            fit_z.append(np.median(z[selected]))

        fit_y = np.asarray(fit_y)
        fit_z = np.asarray(fit_z)

        # Consolidate duplicate vertical positions.
        unique_y = np.unique(fit_y)
        unique_z = np.array([
            np.median(fit_z[fit_y == value])
            for value in unique_y
        ])

        if len(unique_y) < 2:
            return indices

        curve = PchipInterpolator(unique_y, unique_z)
        predicted = curve(np.clip(y, unique_y[0], unique_y[-1]))

        for i, depth in zip(indices, predicted):
            points[i][2] = float(np.clip(depth, 0, max_depth))

        return indices

    # Fit each of the individual branches / trunk
    trunk = fit(trunk)
    left = fit(left)
    right = fit(right)

    def distance(a, b):
        """Calculate the squared image-space distance between two points.

        Args:
            a: The index of the first point.
            b: The index of the second point.

        Returns:
            The squared Euclidean distance between the points' XY coordinates.
        """
        return ((points[a][0] - points[b][0]) ** 2 + (points[a][1] - points[b][1]) ** 2)

    # Find the end of each branch nearest the trunk.
    connections = []

    if trunk:
        for branch in (left, right):
            if not branch:
                continue

            candidates = [
                (distance(endpoint, trunk_i), endpoint, trunk_i)
                for endpoint in (branch[0], branch[-1])
                for trunk_i in trunk
            ]
            connections.append(min(candidates))

    if connections:
        # Both branches receive exactly the same junction depth.
        junction_depth = float(np.median([
            points[trunk_i][2]
            for _, _, trunk_i in connections
        ]))

        for _, endpoint, _ in connections:
            branch = left if endpoint in left else right

            # Order the branch from its junction outward.
            if branch[-1] == endpoint:
                branch = branch[::-1]

            count = min(join_points, len(branch))

            for position, index in enumerate(branch[:count]):
                # Strong correction at the junction, fading outwards.
                weight = 1.0 - position / max(count, 1)
                points[index][2] = (weight * junction_depth + (1.0 - weight) * points[index][2])

    return [tuple(p) for p in points]

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

        # Check if there is only a single value, this indicates the start of the trunk.
        if (two_branches == True) & (len(new_points) == 1):
            end_double_row = row
            final_double_index = len(points_list)
            break
        points_list.extend(new_points)

    radius_multiplier = 1

    old_middle = 128
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

        # Exclude outliers in the data
        min = float('inf')
        for point in new_points:
            if abs(point[0] - old_middle) < min:
                min = abs(point[0] - old_middle)
                new_point = point

        points_list.extend([new_point])
        old_middle = new_point[0]

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
