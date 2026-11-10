"""viam-labs:mujoco-sim:world - owns the MuJoCo scene.

Attributes:
  physics_dt (float)  - physics step in seconds (default 0.002)
  real_time (bool)    - pace sim time to wall time (default true)
  props (list)        - movable objects: {name, type: "cube" | "sphere" |
                        "cylinder" | "capsule" | "mesh", size, scale, position,
                        orientation_wxyz, mass, color, fixed, static_friction,
                        texture, dynamic_friction (accepted, ignored)}. Each shape fills
                        a box of size x scale. A mesh prop takes mesh (path,
                        assets-relative path or https URL), scale and
                        threshold, like the obstacle's; it's split into
                        convex pieces.
  import_geometry (list) - names of components (of any module) whose box,
                        sphere and capsule geometries become fixed geoms, at
                        their frame system poses: a pallet, a fence, an
                        obstacle from another module. GetGeometries first,
                        else the geometry in the component's frame config.
                        Not dependencies: they are looked up after the world
                        starts, every 2 s for up to 60 s.
  floor_texture (string) - a .png or .jpg image (path, assets-relative path or
                        https URL) tiled across the floor. A prop's texture is
                        the same kind of image, on each of its faces. Cameras
                        see textures; the 3D viewer uses color.
  floor_texture_size (float) - one floor tile's edge in meters (default 1)
  lighting (object)   - how cameras see the scene: lights ({type:
                        "directional" | "spot" | "point", position,
                        direction, color, shadows, cutoff_deg}; replaces the
                        default sun), keep_default_light, headlight, skybox
                        ([[r,g,b] top, [r,g,b] bottom]), shadows, shadow_size,
                        reflections, antialias. A scene file with lights
                        drops the default sun. See the README.

DoCommand: status, play, pause, reset, set_prop_pose, get_prop_pose,
park_props, reimport. Shapes match the viam-labs:isaac-sim world.
"""

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

import mujoco
from grpclib import GRPCError, Status
from viam.components.generic import Generic
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, PoseInFrame, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .. import FAMILY, NAMESPACE, robots
from ..meshes import load_pieces, load_texture
from ..shapes import GEOM_TYPES, geometry_shape, mj_size
from ..sim import LOGGER, PROP_CONTACT_PRIORITY, PROP_SOLREF, Contribution, SimManager, WorldConfig
from .obstacle import _pose_quat

# The running module, set by main.py: its parent connection to viam-server
# finds the components to import.
MODULE: Module | None = None
IMPORT = "world:import"  # the import's scene contribution; ":" keeps it apart from component names
IMPORT_RETRY_S = 2.0
IMPORT_TIMEOUT_S = 60.0

# (geom name, shape, position (m), quaternion (w, x, y, z))
_Imported = list[tuple[str, tuple[str, tuple[float, ...]], list[float], list[float]]]


def _import_names(attrs: Mapping[str, Any]) -> list[str]:
    names = attrs.get("import_geometry", [])
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("import_geometry must be a list of component names")
    return names


async def _fetch(parent: Any, name: str) -> tuple[_Imported, str]:
    """One component's box, sphere and capsule geometries at their world
    poses, and where they came from: "geometries" (GetGeometries) or, when
    that has none or isn't implemented, "frame" (the geometry in its frame
    config, what the motion planner sees). ([], "") and a log line for one
    with neither, or one of this module's. Raises while it's not in the
    machine (yet)."""
    if SimManager.get().contributes(name):
        LOGGER.info("import_geometry: %s is a mujoco-sim component, already in the scene; skipped", name)
        return [], ""
    found = [r for r in parent.resource_names if r.name == name and r.type == "component"]
    if not found:
        raise LookupError(f"{name} is not in the machine's resources")
    source = "geometries"
    try:
        geometries: list[Geometry] = list(await parent.get_component(found[0]).get_geometries())
    except GRPCError as error:
        # permanent: a workcell-components model, or an API RDK never asks (motor, sensor)
        if error.status != Status.UNIMPLEMENTED:
            raise
        geometries = []
    if not any(g.WhichOneof("geometry_type") in GEOM_TYPES for g in geometries):
        source = "frame"
        # a part's frame: reference_frame is the component, physical_object its
        # frame.geometry, centred in the component's own frame like GetGeometries'
        geometries = [
            part.frame.physical_object
            for part in await parent.get_frame_system_config()
            if part.frame.reference_frame == name and part.frame.HasField("physical_object")
        ]
    shapes = [(i, g) for i, g in enumerate(geometries) if g.WhichOneof("geometry_type") in GEOM_TYPES]
    if not shapes:
        LOGGER.warning(
            "import_geometry: %s has no box, sphere or capsule geometry, from GetGeometries or its "
            "frame; skipped. Add a box, sphere or capsule geometry to its frame if it should be an "
            "obstacle", name,
        )
        return [], ""
    # the component's frame in world, through any parents: what both sources are relative to
    pose = (await parent.transform_pose(PoseInFrame(reference_frame=name, pose=Pose(o_z=1)), "world")).pose
    frame = robots.transform([pose.x / 1000, pose.y / 1000, pose.z / 1000], _pose_quat(pose))
    labels = [g.label for _, g in shapes]
    out = []
    for i, geometry in shapes:
        center = geometry.center
        world = frame @ robots.transform([center.x / 1000, center.y / 1000, center.z / 1000], _pose_quat(center))
        label = geometry.label if geometry.label and labels.count(geometry.label) == 1 else str(i)
        out.append((f"import/{name}/{label}", geometry_shape(geometry), list(world[:3, 3]), list(robots.quat(world))))
    return out, source


def _contribution(imported: Mapping[str, _Imported]) -> Contribution:
    def build(spec: mujoco.MjSpec) -> None:
        for geoms in imported.values():
            for geom_name, shape, position, quat in geoms:
                spec.worldbody.add_geom(
                    name=geom_name, type=GEOM_TYPES[shape[0]], size=mj_size(shape), pos=position, quat=quat,
                    rgba=[0.55, 0.55, 0.5, 1.0], priority=PROP_CONTACT_PRIORITY, solref=list(PROP_SOLREF),
                )

    return Contribution(build=build)


class MujocoWorld(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "world")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._import_names: list[str] = []
        self._import_task: asyncio.Task | None = None
        self._imported = False  # this world set the import contribution

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        world = cls(config.name)
        world.reconfigure(config, dependencies)
        return world

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        WorldConfig.from_attrs(attrs)
        _import_names(attrs)
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        cfg = WorldConfig.from_attrs(attrs)
        # slow the first time per mesh (CoACD), cached afterwards
        pieces = {
            prop.name: load_pieces(f"prop {prop.name}", prop.mesh)
            for prop in cfg.props
            if prop.mesh is not None
        }
        textures = {
            prop.name: load_texture(f"prop {prop.name}", prop.texture)
            for prop in cfg.props
            if prop.texture is not None
        }
        if cfg.floor_texture is not None:
            textures["floor"] = load_texture("floor_texture", cfg.floor_texture)
        sim = SimManager.get()
        sim.configure_world(cfg, pieces, textures)
        self._import_names = _import_names(attrs)
        if self._import_names:
            # new() runs on the module's event loop; the components come later
            self._start_import()
        else:
            sim.remove_contribution(IMPORT)

    def _start_import(self) -> asyncio.Task:
        task = self._import_task = asyncio.get_running_loop().create_task(self._import())
        # _import logged any error; retrieve it so asyncio doesn't report it again
        task.add_done_callback(lambda task: task.cancelled() or task.exception())
        return task

    async def close(self) -> None:
        task, self._import_task = self._import_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):  # _import logged any error
                await task
        if self._imported:
            await asyncio.to_thread(SimManager.get().remove_contribution, IMPORT)
            self._imported = False

    async def _import(self) -> dict[str, Any]:
        try:
            return await self._import_once()
        except Exception:
            LOGGER.exception("import_geometry: the import failed")
            raise

    async def _import_once(self) -> dict[str, Any]:
        """Import every import_geometry component as one contribution,
        replacing the last import; retry the missing ones every
        IMPORT_RETRY_S until IMPORT_TIMEOUT_S."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + IMPORT_TIMEOUT_S
        while MODULE is None or MODULE.parent is None:  # connected before resources are added, normally
            if loop.time() >= deadline:
                LOGGER.error("import_geometry: no connection to viam-server; nothing imported")
                return {"imported": [], "from_frame": [], "missing": list(self._import_names)}
            await asyncio.sleep(IMPORT_RETRY_S)
        parent = MODULE.parent
        pending, imported, from_frame, errors = list(self._import_names), {}, [], {}
        while True:
            added = False
            try:
                # the parent connected before any component existed and doesn't refresh on its own
                await parent.refresh()
                names = list(pending)
            except Exception as error:  # retried like a missing component
                errors.update(dict.fromkeys(pending, f"refresh failed: {error}"))
                names = []
            for name in names:
                try:
                    geoms, source = await _fetch(parent, name)
                except Exception as error:  # not in the machine or its frame system yet, or a lost connection
                    errors[name] = str(error)
                    continue
                pending.remove(name)
                if geoms:
                    imported[name] = geoms
                    added = True
                    if source == "frame":
                        from_frame.append(name)
            if added:
                await asyncio.to_thread(SimManager.get().set_contribution, IMPORT, _contribution(dict(imported)))
                self._imported = True
            if not pending or loop.time() >= deadline:
                break
            await asyncio.sleep(IMPORT_RETRY_S)
        for name in pending:
            LOGGER.error("import_geometry: could not import %s: %s", name, errors[name])
        if not imported:
            await asyncio.to_thread(SimManager.get().remove_contribution, IMPORT)
            self._imported = False
        return {"imported": sorted(imported), "from_frame": sorted(from_frame), "missing": pending}

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") == "reimport":
            if self._import_task is not None:
                self._import_task.cancel()
            task = self._start_import()
            try:
                # a caller's deadline mustn't abort the import halfway
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if self._import_task is task:  # this call itself was cancelled
                    raise
                return {"superseded": True}
        return await asyncio.to_thread(self._command, command)

    @staticmethod
    def _command(command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        sim = SimManager.get()
        cmd = str(command.get("command", ""))
        if cmd == "status":
            return sim.status()
        if cmd == "play":
            sim.play()
            return {"ok": True}
        if cmd == "pause":
            sim.pause()
            return {"ok": True}
        if cmd == "reset":
            sim.reset()
            return {"ok": True}
        if cmd == "park_props":
            props = command.get("props") or []
            if not isinstance(props, list) or not all(isinstance(p, str) for p in props):
                raise ValueError("park_props requires props: a list of prop names")
            return sim.park_props(props)
        prop = str(command.get("prop", ""))
        if cmd == "get_prop_pose":
            if not prop:
                raise ValueError("get_prop_pose requires prop")
            return {"prop": prop, "pose": sim.get_prop_pose(prop)}
        if cmd == "set_prop_pose":
            if not prop:
                raise ValueError("set_prop_pose requires prop")
            return sim.set_prop_pose(
                prop, command.get("position"), command.get("orientation_wxyz")
            )
        raise ValueError(f"unknown command: {command}")
