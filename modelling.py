"""All of the required functions for transforming the list of points into a 3d mesh"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree

def tapered_cylinder_between(start: Sequence[float], end: Sequence[float], start_radius: float, end_radius: float, sections: int = 16) -> trimesh.Trimesh:
    """Create a tapered cylinder between two 3D positions.

    Args:
        start: The three-dimensional coordinates of the frustum's start.
        end: The three-dimensional coordinates of the frustum's end.
        start_radius: The radius of the frustum at "start".
        end_radius: The radius of the frustum at "end".
        sections: The number of radial sections used to construct the mesh.

    Returns:
        A triangular mesh representing the capped frustum, aligned between
        "start" and "end".

    Raises:
        ValueError: If "start" and "end" are identical, or if either
            radius is not positive.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))

    # Filter out bad inputs
    if length < 1e-8:
        raise ValueError("Cannot join identical 3D positions")
    if start_radius <= 0 or end_radius <= 0:
        raise ValueError("Radii must be positive")

    # Generate the tapered cylinder through a revolution
    profile = np.array([[0.0, 0.0], [float(start_radius), 0.0], [float(end_radius), length], [0.0, length]])
    mesh = trimesh.creation.revolve(profile, sections=sections)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis / length)
    transform[:3, 3] = start
    mesh.apply_transform(transform)
    return mesh

def build_minimum_spanning_tree(image_positions: np.ndarray, neighbour_distance: float) -> list[set[int]]:
    """Build a minimum spanning forest from nearby skeleton pixels.

    Connectivity is calculated in image XY, not list order and not depth. This
    matches a skeleton extracted from a 2D probability mask and avoids noisy
    depth values disconnecting otherwise adjacent skeleton pixels.

    Args:
        image_positions: An "(N, 2)" array containing the image-space XY
            coordinates of the skeleton pixels.
        neighbour_distance: The maximum image-space distance at which two
            pixels are considered neighbours.

    Returns:
        An adjacency list for the minimum spanning forest. Each list index is
        a point index and its set contains the indices of adjacent points.
    """
    point_count = len(image_positions)
    adjacency = [set() for _ in range(point_count)]
    if point_count < 2:
        return adjacency

    tree = cKDTree(image_positions)
    pairs = tree.query_pairs(neighbour_distance, output_type="ndarray")
    if len(pairs) == 0:
        return adjacency

    distances = np.linalg.norm(image_positions[pairs[:, 0]] - image_positions[pairs[:, 1]], axis=1)
    
    # Processing the data into a form that can be used for the utilised algorithm.
    weights = np.maximum(distances, 1e-8)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    columns = np.concatenate([pairs[:, 1], pairs[:, 0]])
    values = np.concatenate([weights, weights])
    graph = coo_matrix((values, (rows, columns)), shape=(point_count, point_count)).tocsr()

    # Determining the minimum spanning tree adjacency.
    forest = minimum_spanning_tree(graph).tocoo()
    for first, second in zip(forest.row, forest.col):
        first = int(first)
        second = int(second)
        adjacency[first].add(second)
        adjacency[second].add(first)
    return adjacency

def trace_branch_paths(adjacency: list[set[int]]) -> list[list[int]]:
    """Trace maximal paths between tips, junctions, and isolated points.

    Args:
        adjacency: An undirected graph represented as an adjacency list, with
            one neighbour-index set per point.

    Returns:
        A list of paths, where each path is an ordered list of point indices
        running between graph anchors.
    """
    paths: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()

    def edge_key(first: int, second: int) -> tuple[int, int]:
        """Create an order-independent key for an undirected edge.

        Args:
            first: The index of the first endpoint.
            second: The index of the second endpoint.

        Returns:
            The two endpoint indices in ascending order.
        """
        return (first, second) if first < second else (second, first)

    # Find the endpoints and branch intersections.
    anchors = [
        index for index, neighbours in enumerate(adjacency)
        if len(neighbours) != 2
    ]

    for start in anchors:
        if not adjacency[start]:
            paths.append([start])
            continue

        # Start a path along each unused edge.
        for neighbour in adjacency[start]:
            first_edge = edge_key(start, neighbour)
            if first_edge in visited_edges:
                continue

            path = [start]
            previous = start
            current = neighbour
            visited_edges.add(first_edge)

            # Follow the path until another endpoint or intersection is reached.
            while True:
                path.append(current)
                if len(adjacency[current]) != 2:
                    break

                next_candidates = adjacency[current] - {previous}
                if not next_candidates:
                    break

                next_point = next(iter(next_candidates))
                next_edge = edge_key(current, next_point)
                if next_edge in visited_edges:
                    break

                visited_edges.add(next_edge)
                previous, current = current, next_point

            paths.append(path)

    return paths

def sample_path(path: list[int], points_per_segment: int) -> list[int]:
    """Keep the path start, every Nth point, and the path end.

    Args:
        path: An ordered list of point indices describing a branch path.
        points_per_segment: The interval between retained point indices along
            the path.

    Returns:
        The sampled point indices, always including the final path point.
    """
    selected = path[::points_per_segment]
    if selected[-1] != path[-1]:
        selected.append(path[-1])
    return selected

def create_model(points: list, points_per_segment: int = 5, neighbour_distance: float = 3.0, sections: int = 16, pixels_per_m: float = 65.0, decimate_perc: float = 0.25, add_joint_spheres: bool = True) -> trimesh.Trimesh:
    """Create the model without assuming that "points" is branch-ordered.

    This keeps the original compatible call "create_model(points_list)".
    Input values remain "(x, y, z, radius)". Point order is ignored: nearby
    image-space points are connected into a minimum spanning forest, maximal
    branch paths are traced, and every "points_per_segment" point along each
    traced path becomes a tapered-segment boundary. Every path's actual start
    and end are always retained.

    Args:
        points: Point records in "(x, y, z, radius)" form. Point order is
            ignored when constructing the skeleton graph.
        points_per_segment: The number of path points between sampled segment
            boundaries.
        neighbour_distance: The maximum XY distance between neighbouring
            skeleton points.
        sections: The number of radial sections used for each tapered segment.
        pixels_per_m: The scale factor applied to point depth values.
        decimate_perc: The fraction of mesh faces to retain when decimation is
            enabled.
        add_joint_spheres: Whether to add overlapping spheres at sampled path
            points to reinforce bends and junctions.

    Returns:
        The unified triangular mesh generated from all tapered segments and
        joint or isolated-point spheres.

    Raises:
        ValueError: If an option is outside its valid range, if "points" is
            not an "(N, 4)" array, if no points are supplied, or if point
            values are non-finite or contain a non-positive radius.
        RuntimeError: If no geometry can be generated or the Boolean union
            fails to produce a mesh.
    """
    # Filter out any bad input data.
    if points_per_segment < 1:
        raise ValueError("points_per_segment must be at least 1")
    if neighbour_distance <= 0:
        raise ValueError("neighbour_distance must be positive")
    if sections < 3:
        raise ValueError("sections must be at least 3")
    if pixels_per_m <= 0:
        raise ValueError("pixels_per_m must be positive")
    if not 0.0 < decimate_perc <= 1.0:
        raise ValueError("decimate_perc must be in the range (0, 1]")

    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 4:
        raise ValueError("points must have shape (N, 4): x, y, z, radius")
    if len(point_array) == 0:
        raise ValueError("At least one point is required")
    if np.any(~np.isfinite(point_array)):
        raise ValueError("points contain NaN or infinite values")
    if np.any(point_array[:, 3] <= 0):
        raise ValueError("Every radius must be positive")

    image_positions = point_array[:, :2]
    model_positions = np.column_stack([point_array[:, 0], 255.0 - point_array[:, 1], pixels_per_m * point_array[:, 2]])
    radii = point_array[:, 3]

    adjacency = build_minimum_spanning_tree(image_positions, neighbour_distance=neighbour_distance)
    branch_paths = trace_branch_paths(adjacency)

    meshes: list[trimesh.Trimesh] = []
    joint_indices: set[int] = set()

    for path in branch_paths:
        selected = sample_path(path, points_per_segment)
        joint_indices.update(selected)

        for first, second in zip(selected[:-1], selected[1:]):
            start = model_positions[first]
            end = model_positions[second]
            if np.linalg.norm(end - start) < 1e-8:
                continue
            meshes.append(tapered_cylinder_between(start, end, float(radii[first]), float(radii[second]), sections))

    if add_joint_spheres:
        # This retains the original model's reliable overlap at bends and tree
        # junctions, but only at sampled control points rather than every pixel.
        for index in sorted(joint_indices):
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=float(radii[index]))
            sphere.apply_translation(model_positions[index])
            meshes.append(sphere)

    # Isolated pixels are not part of any segment and remain spheres, matching
    # the behaviour of the original implementation.
    for index, neighbours in enumerate(adjacency):
        if neighbours:
            continue
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=float(radii[index]))
        sphere.apply_translation(model_positions[index])
        meshes.append(sphere)

    if not meshes:
        raise RuntimeError("No geometry could be generated from the points")

    # Merge and decimate to create a single mesh
    merged = trimesh.boolean.union(meshes, engine="manifold")
    if merged is None or len(merged.faces) == 0:
        raise RuntimeError("Boolean union failed to create a mesh")

    if decimate_perc < 1.0:
        target_faces = max(4, int(len(merged.faces) * decimate_perc))
        merged = merged.simplify_quadric_decimation(face_count=target_faces)

    return merged

def create_sphere_model(points: list) -> trimesh.Trimesh:
    """
    Creates a 3D model from a list of points, representing each point as a sphere.

    Args:
        points (list[tuple]): A list of points, where each tuple contains
            (x, y, z, radius), defining the sphere's centre and radius.
            x, y indicate pixels. z indicates the depth in meters.

    Returns:
        optimised (trimesh.Trimesh): The generated 3D model containing all spheres.
    """
    meshes = []
    pixels_per_m = 65
    decimate_perc = 0.25

    # Generate every point into a list
    for x, y, z, r in points:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=r)
        sphere.apply_translation([x, 255-y, pixels_per_m*z])
        meshes.append(sphere)

    # Combine all spheres into one mesh
    merged = trimesh.boolean.union(meshes, engine="manifold")

    # Decimate the unoptimised to reduce the amount of faces, optimising the geometry for easier calculations.
    target_faces = int(len(merged.faces) * decimate_perc)
    optimised = merged.simplify_quadric_decimation(face_count=target_faces)

    return optimised
