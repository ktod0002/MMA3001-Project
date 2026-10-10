import trimesh

def create_model(points: list) -> trimesh.Trimesh:
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
    pixels_per_m = 90
    decimate_perc = 0.25

    # Generate every point into a list
    for x, y, z, r in points:
        sphere = trimesh.creation.icosphere(
            subdivisions=2,
            radius=r
        )
        sphere.apply_translation([x, 255-y, pixels_per_m*z])
        meshes.append(sphere)

    # Combine all spheres into one mesh
    merged = trimesh.boolean.union(meshes, engine="manifold")

    # Decimate the unoptimised to reduce the amount of faces, optimising the geometry for easier calculations.
    target_faces = int(len(merged.faces) * decimate_perc)
    optimised = merged.simplify_quadric_decimation(face_count=target_faces)

    return optimised
