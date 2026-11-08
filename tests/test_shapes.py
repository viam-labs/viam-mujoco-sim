import pytest
from viam.proto.common import Capsule, Geometry, Mesh, RectangularPrism, Sphere, Vector3

from mujoco_module.shapes import (
    cylinder_ply,
    geometry_shape,
    mj_size,
    prop_shape,
    viewer_geometry,
)


def test_prop_shapes_fill_size_times_scale():
    assert prop_shape("b", "cube", 0.1, (1, 2, 3)) == ("box", pytest.approx((0.05, 0.1, 0.15)))
    assert prop_shape("s", "sphere", 0.1, (2, 2, 2)) == ("sphere", pytest.approx((0.1,)))
    assert prop_shape("c", "cylinder", 0.1, (1, 1, 3)) == ("cylinder", pytest.approx((0.05, 0.15)))
    # capsule 0.1 wide, 0.3 tall: radius 0.05, straight part 0.2 long -> half 0.1
    assert prop_shape("p", "capsule", 0.1, (1, 1, 3)) == ("capsule", pytest.approx((0.05, 0.1)))


@pytest.mark.parametrize("kind,scale,message", [
    ("cone", (1, 1, 1), "type must be one of"),
    ("sphere", (1, 1, 2), "sphere's scale must be uniform"),
    ("cylinder", (1, 2, 1), r"scale\[0\] and scale\[1\] must match"),
    ("capsule", (1, 1, 1), "height must be greater than its diameter"),
])
def test_prop_shape_errors(kind, scale, message):
    with pytest.raises(ValueError, match=message):
        prop_shape("thing", kind, 0.1, scale)


def test_geometry_shape_converts_viam_geometry_mm_to_mujoco_m():
    box = Geometry(box=RectangularPrism(dims_mm=Vector3(x=600, y=800, z=20)))
    assert geometry_shape(box) == ("box", pytest.approx((0.3, 0.4, 0.01)))
    assert geometry_shape(Geometry(sphere=Sphere(radius_mm=50))) == ("sphere", pytest.approx((0.05,)))
    capsule = Geometry(capsule=Capsule(radius_mm=50, length_mm=300))
    assert geometry_shape(capsule) == ("capsule", pytest.approx((0.05, 0.1)))
    # RDK turns this into a sphere before config reaches a module; guard anyway
    round_capsule = Geometry(capsule=Capsule(radius_mm=50, length_mm=100))
    assert geometry_shape(round_capsule) == ("sphere", pytest.approx((0.05,)))


@pytest.mark.parametrize("geometry,message", [
    (Geometry(), "set frame.geometry to a box, sphere or capsule"),
    (Geometry(mesh=Mesh(content_type="stl", mesh=b"x")), "mesh obstacles aren't supported yet"),
    (Geometry(box=RectangularPrism(dims_mm=Vector3(x=1, y=0, z=1))), "must be positive"),
    (Geometry(sphere=Sphere(radius_mm=float("nan"))), "must be positive"),
    (Geometry(capsule=Capsule(radius_mm=50, length_mm=60)), "at least twice its radius"),
])
def test_geometry_shape_errors(geometry, message):
    with pytest.raises(ValueError, match=message):
        geometry_shape(geometry)


def test_mj_size_pads_to_three():
    assert mj_size(("sphere", (0.05,))) == [0.05, 0.0, 0.0]
    assert mj_size(("box", (0.1, 0.2, 0.3))) == [0.1, 0.2, 0.3]


def test_viewer_geometry_is_in_mm_and_labelled():
    box = viewer_geometry("box", (0.05, 0.1, 0.15), "b")
    assert box.label == "b"
    assert (box.box.dims_mm.x, box.box.dims_mm.y, box.box.dims_mm.z) == pytest.approx((100, 200, 300))
    assert box.center.o_z == 1
    assert viewer_geometry("sphere", (0.05,), "s").sphere.radius_mm == pytest.approx(50)
    capsule = viewer_geometry("capsule", (0.05, 0.1), "p").capsule
    assert (capsule.radius_mm, capsule.length_mm) == pytest.approx((50, 300))  # tip to tip
    mesh = viewer_geometry("cylinder", (0.05, 0.15), "c").mesh
    assert mesh.content_type == "ply" and mesh.mesh.startswith(b"ply\nformat ascii 1.0\n")


def test_cylinder_ply_is_meters_centred_on_z():
    text = cylinder_ply(0.05, 0.3).decode()
    header, body = text.split("end_header\n")
    assert "property list uchar int vertex_indices" in header
    n_vertices = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[2])
    n_faces = int(next(l for l in header.splitlines() if l.startswith("element face")).split()[2])
    lines = body.strip().splitlines()
    assert len(lines) == n_vertices + n_faces
    vertices = [tuple(map(float, l.split())) for l in lines[:n_vertices]]
    assert max(abs(v[2]) for v in vertices) == pytest.approx(0.15)  # meters, centred
    assert max(abs(v[0]) for v in vertices) == pytest.approx(0.05)
    assert all(l.split()[0] == "3" for l in lines[n_vertices:])
    assert all(0 <= int(i) < n_vertices for l in lines[n_vertices:] for i in l.split()[1:])
