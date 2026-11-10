"""The world's lights, headlight, skybox and render quality, applied to a
fresh spec on every rebuild and to each new renderer. The settings are parsed
by sim.LightingConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from .sim import LightingConfig

LIGHT_TYPES = {
    "directional": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    "spot": mujoco.mjtLightType.mjLIGHT_SPOT,
    "point": mujoco.mjtLightType.mjLIGHT_POINT,
}
# MuJoCo's headlight defaults; `headlight` scales all three
HEADLIGHT = {"ambient": 0.1, "diffuse": 0.4, "specular": 0.5}
SKYBOX_NAME = "world/skybox"
SKYBOX_FACE_PX = 512


def apply(spec: mujoco.MjSpec, cfg: LightingConfig, scene_lights: bool) -> None:
    """The world's lights (or the default sun) and global visual settings.
    The default sun is left out when a scene file brings lights, unless
    keep_default_light."""
    if cfg.lights is None:
        if not scene_lights or cfg.keep_default_light:
            spec.worldbody.add_light(
                name="sun", pos=[0, 0, 3], dir=[0, 0, -1],
                type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            )
    else:
        for index, light in enumerate(cfg.lights):
            added = spec.worldbody.add_light(
                name=f"light_{index}",
                type=LIGHT_TYPES[light.type],
                pos=list(light.position),
                dir=list(light.direction),
                diffuse=list(light.color),
                castshadow=light.shadows,
            )
            if light.type == "spot":
                added.cutoff = light.cutoff_deg
    headlight = spec.visual.headlight
    if cfg.headlight == 0:
        headlight.active = 0
    for key, value in HEADLIGHT.items():
        setattr(headlight, key, [value * cfg.headlight] * 3)
    spec.visual.quality.shadowsize = cfg.shadow_size
    spec.visual.quality.offsamples = cfg.antialias


def add_skybox(spec: mujoco.MjSpec, cfg: LightingConfig) -> list[str]:
    """Call after every contribution has built. Adds the world's gradient
    skybox unless the spec already has one (MuJoCo shows the first skybox
    texture, so this keeps a scene's in front). Returns the names of the
    skyboxes already there when the world's was left out, else []."""
    if cfg.skybox is None:
        return []
    present = [t.name for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX]
    if present:
        return present
    top, bottom = cfg.skybox
    spec.add_texture(
        name=SKYBOX_NAME,
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=list(top),
        rgb2=list(bottom),
        width=SKYBOX_FACE_PX,
        height=SKYBOX_FACE_PX,
    )
    return []


def set_render_flags(scene: mujoco.MjvScene, cfg: LightingConfig) -> None:
    """Shadows and reflections; they persist across update_scene."""
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = cfg.shadows
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = cfg.reflections
