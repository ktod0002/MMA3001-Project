"""Straightforward tests for the mesh-modelling helpers."""

import numpy as np
import pytest
import trimesh
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modelling

def test_tapered_cylinder_creates_mesh_between_vertical_points():
    mesh = modelling.tapered_cylinder_between(
        start=(0, 0, 0),
        end=(0, 0, 2),
        start_radius=1.0,
        end_radius=0.5,
        sections=12,
    )

    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    np.testing.assert_allclose(mesh.bounds[:, 2], [0.0, 2.0])


def test_tapered_cylinder_rejects_identical_points():
    with pytest.raises(ValueError, match="identical"):
        modelling.tapered_cylinder_between((1, 2, 3), (1, 2, 3), 1.0, 1.0)


@pytest.mark.parametrize(
    ("start_radius", "end_radius"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
)
def test_tapered_cylinder_rejects_non_positive_radii(start_radius, end_radius):
    with pytest.raises(ValueError, match="positive"):
        modelling.tapered_cylinder_between(
            (0, 0, 0), (0, 0, 1), start_radius, end_radius
        )


def test_minimum_spanning_tree_handles_no_points():
    positions = np.empty((0, 2), dtype=float)

    assert modelling.build_minimum_spanning_tree(positions, 2.0) == []


def test_minimum_spanning_tree_handles_one_point():
    positions = np.array([[4.0, 7.0]])

    assert modelling.build_minimum_spanning_tree(positions, 2.0) == [set()]


def test_minimum_spanning_tree_connects_nearby_line_points():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    adjacency = modelling.build_minimum_spanning_tree(positions, 1.1)

    assert adjacency == [{1}, {0, 2}, {1}]


def test_minimum_spanning_tree_leaves_distant_points_isolated():
    positions = np.array([[0.0, 0.0], [10.0, 10.0]])

    assert modelling.build_minimum_spanning_tree(positions, 2.0) == [set(), set()]


def test_minimum_spanning_tree_connects_duplicate_positions():
    positions = np.array([[3.0, 4.0], [3.0, 4.0]])

    assert modelling.build_minimum_spanning_tree(positions, 1.0) == [{1}, {0}]


def test_trace_branch_paths_handles_isolated_points():
    adjacency = [set(), set()]

    assert modelling.trace_branch_paths(adjacency) == [[0], [1]]


def test_trace_branch_paths_follows_a_straight_line():
    adjacency = [{1}, {0, 2}, {1, 3}, {2}]

    assert modelling.trace_branch_paths(adjacency) == [[0, 1, 2, 3]]


def test_trace_branch_paths_splits_at_a_junction():
    adjacency = [{1}, {0, 2, 3}, {1}, {1}]

    paths = modelling.trace_branch_paths(adjacency)
    path_edges = {frozenset(path) for path in paths}

    assert path_edges == {
        frozenset((0, 1)),
        frozenset((1, 2)),
        frozenset((1, 3)),
    }


@pytest.mark.parametrize(
    ("path", "interval", "expected"),
    [
        ([0], 3, [0]),
        ([0, 1, 2, 3, 4], 2, [0, 2, 4]),
        ([0, 1, 2, 3, 4, 5], 2, [0, 2, 4, 5]),
        ([0, 1, 2], 10, [0, 2]),
    ],
)
def test_sample_path_keeps_regular_points_and_the_end(path, interval, expected):
    assert modelling.sample_path(path, interval) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points_per_segment": 0},
        {"neighbour_distance": 0},
        {"sections": 2},
        {"pixels_per_m": 0},
        {"decimate_perc": 0},
        {"decimate_perc": 1.1},
    ],
)
def test_create_model_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        modelling.create_model([(0, 0, 1, 1)], **kwargs)


@pytest.mark.parametrize(
    "points",
    [
        [],
        [(0, 0, 1)],
        [(0, 0, np.nan, 1)],
        [(0, 0, 1, 0)],
        [(0, 0, 1, -1)],
    ],
)
def test_create_model_rejects_invalid_points(points):
    with pytest.raises(ValueError):
        modelling.create_model(points)


def test_create_model_places_an_isolated_point_correctly(monkeypatch):
    captured = {}

    def fake_union(meshes, engine):
        captured["mesh_count"] = len(meshes)
        captured["engine"] = engine
        return meshes[0]

    monkeypatch.setattr(modelling.trimesh.boolean, "union", fake_union)

    mesh = modelling.create_model(
        [(10, 20, 2, 1)],
        decimate_perc=1.0,
        add_joint_spheres=False,
    )

    assert captured == {"mesh_count": 1, "engine": "manifold"}
    np.testing.assert_allclose(mesh.centroid, [10, 235, 130], atol=1e-6)


def test_create_model_reports_boolean_union_failure(monkeypatch):
    monkeypatch.setattr(modelling.trimesh.boolean, "union", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="Boolean union failed"):
        modelling.create_model(
            [(10, 20, 2, 1)],
            decimate_perc=1.0,
            add_joint_spheres=False,
        )


def test_create_sphere_model_translates_and_decimates(monkeypatch):
    created_meshes = []

    class FakeMesh:
        def __init__(self, radius=None):
            self.radius = radius
            self.translation = None
            self.faces = np.zeros((40, 3), dtype=int)
            self.simplified_to = None

        def apply_translation(self, translation):
            self.translation = translation

        def simplify_quadric_decimation(self, face_count):
            self.simplified_to = face_count
            return self

    def fake_icosphere(subdivisions, radius):
        mesh = FakeMesh(radius)
        created_meshes.append(mesh)
        assert subdivisions == 2
        return mesh

    merged = FakeMesh()

    def fake_union(meshes, engine):
        assert meshes == created_meshes
        assert engine == "manifold"
        return merged

    monkeypatch.setattr(modelling.trimesh.creation, "icosphere", fake_icosphere)
    monkeypatch.setattr(modelling.trimesh.boolean, "union", fake_union)

    result = modelling.create_sphere_model(
        [(10, 20, 2, 1.5), (30, 40, 1, 0.5)]
    )

    assert result is merged
    assert [mesh.radius for mesh in created_meshes] == [1.5, 0.5]
    assert [mesh.translation for mesh in created_meshes] == [
        [10, 235, 130],
        [30, 215, 65],
    ]
    assert merged.simplified_to == 10

if __name__ == "__main__":
    from contextlib import redirect_stderr, redirect_stdout
    from pathlib import Path

    report_path = Path(__file__).with_name("modelling_test_report.txt")

    with report_path.open("w", encoding="utf-8") as report:
        with redirect_stdout(report), redirect_stderr(report):
            exit_code = pytest.main([__file__, "-v"])

    print(f"Test report saved to: {report_path}")
    raise SystemExit(exit_code)