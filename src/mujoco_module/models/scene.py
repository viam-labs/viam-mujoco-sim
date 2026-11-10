"""viam-labs:mujoco-sim:scene - an MJCF scene file attached into the world.

The frame (parent "world", no geometry) places the file's world origin. The
file's bodies, worldbody geoms, lights, materials, textures and skybox come
in, every name prefixed "<name>/"; the world keeps its own <option>, <size>
and <visual> (the file's own angle units still apply to it). A colliding plane
directly in the file's worldbody replaces the world's floor. Free bodies
become props named "<name>/<body>" for the world's prop commands; the world
state store publishes them. RDK asks for this
component's geometries (its frame has none) and gets every colliding fixed or
passively jointed geom at the file's starting pose: meshes as their convex
hull, ellipsoids as a box; planes and height fields aren't planned against.
Actuators, tendons, equality constraints, flexes, skins, plugins and mocap
bodies are errors; keyframes are dropped.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
  mjcf (string, required)  - the scene's .xml file: an absolute path or a path
                             relative to the module's assets/
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import ClassVar, Self

import mujoco
from viam.components.generic import Generic
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..scenes import Scene, copy_world_settings, load_scene
from ..sim import Contribution, SimManager
from .utils import frame_pose, get_attrs, validate_sim_component


def _parse(config: ComponentConfig) -> str:
    """The mjcf source; ValueError if the config is invalid."""
    name = config.name
    if not config.HasField("frame"):
        raise ValueError(f'{name}: set a frame with parent "world" to place the scene')
    validate_sim_component(config, needs_asset=False)  # world attr; parent must be world
    if config.frame.HasField("geometry"):
        raise ValueError(
            f"{name}: remove frame.geometry: RDK would plan against it instead of the scene's geometry"
        )
    source = get_attrs(config).get("mjcf")
    if not isinstance(source, str) or not source.lower().endswith(".xml"):
        raise ValueError(f"{name}: mjcf must be a .xml MJCF file path, got {source!r}")
    return source


def scene_contribution(name: str, scene: Scene, position, quat) -> Contribution:
    def build(spec: mujoco.MjSpec) -> None:
        child = scene.spec.copy()
        copy_world_settings(spec, child)
        frame = spec.worldbody.add_frame(pos=list(position), quat=list(quat))
        spec.attach(child, prefix=f"{name}/", frame=frame)

    return Contribution(build=build, replaces_floor=scene.has_plane, props=scene.props)


class MujocoScene(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "scene")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._geometries: list[Geometry] | None = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        scene = cls(config.name)
        scene.reconfigure(config, dependencies)
        return scene

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        _parse(config)
        return [str(get_attrs(config)["world"])], []

    # Called once, from new(): viam-server rebuilds resources (close() then new())
    # on config changes. The geometries must exist before new() returns: RDK
    # builds the frame system then and asks for them.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        source = _parse(config)
        position, quat = frame_pose(config)
        position = position or (0.0, 0.0, 0.0)
        quat = quat or (1.0, 0.0, 0.0, 0.0)
        scene = load_scene(self.name, source, position, quat)
        SimManager.get().set_contribution(
            self.name, scene_contribution(self.name, scene, position, quat)
        )
        self._geometries = list(scene.geometries)

    async def close(self) -> None:
        if self._geometries is not None:
            await asyncio.to_thread(SimManager.get().remove_contribution, self.name)
            self._geometries = None

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return list(self._geometries or [])
