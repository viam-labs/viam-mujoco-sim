"""Sim twin of a real Viam machine, behind `cell.py twin`, `twin-pose` and `twin-check`.

convert() maps a real machine's expanded config to a twin config and a report. The rest compares
the running twin with what `twin` saved about the real machine. Poses are 4x4 numpy arrays in mm.
"""

import copy
import io
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _unit(v):
    return v / np.linalg.norm(v)


def quat_from_orientation(orientation):
    """(w, x, y, z) of a config frame orientation; empty means identity."""
    from viam.spatialmath import AxisAngle, EulerAngles, OrientationVector

    if not orientation:
        return np.array([1.0, 0.0, 0.0, 0.0])
    kind, v = orientation.get("type", "ov_degrees"), orientation.get("value") or {}
    get = lambda key, default=0.0: float(default if v.get(key) is None else v[key])  # None: unresolved variable
    if kind in ("ov_degrees", "ov_radians"):
        th = get("th")
        x, y, z = get("x"), get("y"), get("z", 1.0)
        if x == y == z == 0:
            z = 1.0  # RDK treats a zero vector as z
        q = OrientationVector(x, y, z, math.radians(th) if kind == "ov_degrees" else th)
    elif kind == "euler_angles":
        q = EulerAngles(get("roll"), get("pitch"), get("yaw"))
    elif kind == "axis_angles":
        q = AxisAngle(get("x"), get("y"), get("z", 1.0), get("th"))
    elif kind == "quaternion":
        return _unit(np.array([get("w", 1.0), get("x"), get("y"), get("z")]))
    else:
        raise ValueError(f"orientation type {kind!r} is not supported")
    q = q.to_quaternion()
    return np.array([q.w, q.i, q.j, q.k])


def _mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat(r):
    """(w, x, y, z) of a 3x3 rotation (Shepperd's method, stable at half turns)."""
    t = np.trace(r)
    i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
    if t >= r[i, i]:
        w = math.sqrt(1 + t) / 2
        return np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)])
    j, k = (i + 1) % 3, (i + 2) % 3
    q = np.zeros(4)
    q[1 + i] = math.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) / 2
    s = 4 * q[1 + i]
    q[0] = (r[k, j] - r[j, k]) / s
    q[1 + j] = (r[j, i] + r[i, j]) / s
    q[1 + k] = (r[k, i] + r[i, k]) / s
    return q


def frame_pose(frame):
    """4x4 (mm) of a config frame relative to its parent."""
    t = frame.get("translation") or {}
    m = np.eye(4)
    m[:3, :3] = _mat(quat_from_orientation(frame.get("orientation")))
    m[:3, 3] = [float(t.get(k) or 0.0) for k in "xyz"]
    return m


def pose_frame(parent, m):
    """A config frame for a 4x4 (mm), its orientation as ov_degrees."""
    from viam.spatialmath import Quaternion

    ov = Quaternion(*_quat(m[:3, :3])).to_orientation_vector()
    return {
        "parent": parent,
        "translation": {k: round(float(v), 3) for k, v in zip("xyz", m[:3, 3], strict=True)},
        "orientation": {"type": "ov_degrees", "value": {
            "x": round(ov.o_x, 6), "y": round(ov.o_y, 6), "z": round(ov.o_z, 6), "th": round(math.degrees(ov.theta), 4)}},
    }


def pose_error(a, b):
    """(mm, degrees) between two 4x4s."""
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1.0, 1.0)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), math.degrees(math.acos(cos))


def to_ancestor(name, frames, stop):
    """(ancestor, 4x4): `name`'s pose composed up its frame chain until the parent is in `stop`, is
    world, or has no frame. A moving parent (a gantry) counts at its zero."""
    m, current = np.eye(4), name
    while True:
        frame = frames[current]
        m = frame_pose(frame) @ m
        parent = frame.get("parent", "world")
        if parent == "world" or parent in stop or parent not in frames:
            return parent, m
        current = parent


# ---------- fragments ----------

MERGED = ("components", "services", "modules", "packages")


def expand(part, fetch):
    """(config, unapplied): the config the machine runs, `part`'s own resources plus every fragment's
    (nested too), with `fragment_mods` applied. `fetch(id, version)` returns a fragment's config.
    Only $set and $unset are applied; anything else is returned in `unapplied`, as is each
    {"op": "variable", "name", "path"} left with neither an import value nor a default."""
    seen, unapplied = set(), []

    def fill(config, values):  # a $variable takes its import's value, else its default; the rest is reported
        config, left = resolve_variables(config, values)
        unapplied.extend({"op": "variable", **v} for v in left)
        return config

    part = fill(part, None)
    out = {k: copy.deepcopy(v) for k, v in part.items() if k not in ("fragments", "fragment_mods")}

    def merge(config):
        for ref in config.get("fragments", []):
            fid, version = (ref, None) if isinstance(ref, str) else (ref["id"], ref.get("version"))
            if "@" in fid or (not isinstance(ref, str) and "prefix" in ref) or fid in seen:
                unapplied.append({"op": "fragment_import", "path": fid})
                continue
            seen.add(fid)
            fragment = fill(fetch(fid, version), None if isinstance(ref, str) else ref.get("variables"))
            for key in MERGED:
                out.setdefault(key, []).extend(copy.deepcopy(fragment.get(key, [])))
            merge(fragment)

    merge(part)
    for key in MERGED:
        if key in out:
            names, kept = set(), []
            for item in out[key]:
                if item.get("name") not in names:
                    names.add(item.get("name"))
                    kept.append(item)
            out[key] = kept
    for entry in part.get("fragment_mods", []):
        for mod in entry.get("mods", []):
            for op, fields in mod.items():
                for path, value in fields.items():
                    if op not in ("$set", "$unset") or not _apply(out, op, path, value):
                        unapplied.append({"op": op, "path": path})
    return out, unapplied


def _apply(config, op, path, value):
    keys = path.split(".")
    if len(keys) < 3 or keys[0] not in ("components", "services"):
        return False
    node = next((r for r in config.get(keys[0], []) if r.get("name") == keys[1]), None)
    if node is None:
        return False
    for key in keys[2:-1]:
        node = node.setdefault(key, {}) if op == "$set" else node.get(key)
        if not isinstance(node, dict):
            return op == "$unset" and node is None
    if op == "$set":
        node[keys[-1]] = copy.deepcopy(value)
    else:
        node.pop(keys[-1], None)
    return True


def resolve_variables(config, values=None):
    """(config, unresolved): `config` with every {"$variable": {...}} replaced by values[name], else a
    copy of its default_value, else None (then listed as {"name", "path"})."""
    unresolved, values = [], values or {}

    def walk(node, path):
        if isinstance(node, dict):
            if set(node) == {"$variable"}:
                var = node["$variable"]
                if var.get("name") in values:
                    return copy.deepcopy(values[var["name"]])
                if "default_value" in var:
                    return walk(copy.deepcopy(var["default_value"]), path)
                unresolved.append({"name": var.get("name"), "path": ".".join(path)})
                return None
            return {k: walk(v, [*path, k]) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, [*path, str(v["name"]) if isinstance(v, dict) and "name" in v else str(i)])
                    for i, v in enumerate(node)]
        return node

    return walk(config, []), unresolved


def apply_live_frames(real, frames):
    """(real, names): a copy of `real` whose components named in `frames` take the live parent,
    translation and orientation, keeping any geometry their config frame had."""
    real, names = copy.deepcopy(real), []
    for c in real.get("components", []):
        if c.get("name") in frames:
            geometry = (c.get("frame") or {}).get("geometry")
            c["frame"] = {**copy.deepcopy(frames[c["name"]]), **({"geometry": geometry} if geometry else {})}
            names.append(c["name"])
    return real, names


# ---------- convert ----------

SIM = "viam-labs:mujoco-sim"
FAKE = "rdk:builtin:fake"
# Real arm model -> mujoco-sim arm attributes. twin-check's kinematics comparison is the check on
# each entry; record the first real machine it passed on beside the entry.
ARMS = {
    "viam:universal-robots:ur5e": {"asset": "ur5e"},
    "viam:ufactory:lite6": {"asset": "lite6"},
}
# Real gripper model -> (mujoco-sim gripper attributes, the sim arm asset it needs or None for any,
# frame geometry to add when the real frame has none: the planner's shape for a gripper that serves
# none of its own; every mujoco-sim gripper does now, the Lite6's own included).
GRIPPERS = {
    "viam:ufactory:gripper_lite": ({"asset": "lite6_gripper"}, "lite6", None),
}
COPY_MODULES = {"viam:obstacle-pack", "erh:vmodutils"}  # their components are copied as they are
FAKE_APIS = {f"rdk:component:{a}" for a in (
    "arm", "audio_in", "audio_out", "base", "board", "button", "camera", "encoder", "gantry", "generic", "gripper",
    "input_controller", "motor", "movement_sensor", "power_sensor", "sensor", "servo", "switch")} | {
    f"rdk:service:{a}" for a in ("discovery", "generic", "slam", "video", "vision", "world_state_store")}
SIM_SOURCES = {"color", "depth"}


# ponytail: heuristic lists, extend when a new one turns up. Names match by word (snake, kebab, camelCase), so
# `keyframe` stays and `api_key`/`apiKey`/`HF_TOKEN` go; values match known live-credential shapes.
SECRET_WORDS = {"key", "keys", "token", "tokens", "secret", "secrets", "password", "passwd", "passphrase", "pwd",
                "credential", "credentials", "auth", "authorization", "webhook", "cookie", "dsn", "apikey",
                "accesskey", "secretkey", "authtoken", "privatekey", "clientsecret"}
SECRET_VALUE = re.compile(r"hooks\.slack\.com/|discord(app)?\.com/api/webhooks/|://[^/\s:@]+:[^/\s@]+@|^Bearer "
                          r"|^(sk|hf|ghp|gho|xox[abps])[-_]|-----BEGIN|[?&](token|key|api_key|apikey|access_token|sig"
                          r"|signature|password)=|^AKIA[0-9A-Z]{16}|^eyJ[\w-]+\.[\w-]+\.|^AIza"
                          r"|^(glpat-|github_pat_|gh[pousr]_)")


def secret_name(name):
    flat = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name)).lower()
    return "connection_string" in flat or any(w in SECRET_WORDS for w in re.split(r"[^a-z]+", flat))


def scrub(value, path):
    """(copy of value without secret keys and values, the paths removed). Paths only: never a value."""
    gone = []
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            here = f"{path}.{k}"
            # numbers and bools are never credentials (max_tokens: 512, use_auth: true)
            if (secret_name(k) and (v and isinstance(v, str) or _strings(v))) or (
                    isinstance(v, str) and SECRET_VALUE.search(v)):
                gone.append(here)
            else:
                out[k], g = scrub(v, here)
                gone += g
        return out, gone
    if isinstance(value, list):
        out = []
        for i, v in enumerate(value):
            here = f"{path}[{i}]"
            if isinstance(v, str) and SECRET_VALUE.search(v):
                gone.append(here)
            else:
                v, g = scrub(v, here)
                out.append(v)
                gone += g
        return out, gone
    return copy.deepcopy(value), gone


def api_of(resource, kind):
    """A resource's API triplet; `kind` is "component" or "service" for configs that give only `type`."""
    return resource.get("api") or f"{resource.get('namespace', 'rdk')}:{kind}:{resource['type']}"


def module_of(model):
    """"ns:module" of a "ns:module:model" triplet, or None for builtins and short names."""
    parts = (model or "").split(":")
    return None if len(parts) != 3 or parts[0] == "rdk" else ":".join(parts[:2])


def _unique(name, taken):
    while name in taken:
        name += "-twin"
    taken.add(name)
    return name


class _Twin:
    """Accumulates the twin config and the report while convert() walks the real config."""

    def __init__(self, real, live, module_version):
        self.real, self.live = real, live or {}
        self.comps = copy.deepcopy(real.get("components", []))
        self.frames = {c["name"]: c["frame"] for c in self.comps if c.get("frame")}
        taken = {r["name"] for r in self.comps + real.get("services", [])}
        self.world = _unique("sim-world", taken)
        self.overview = _unique("twin-overview", taken)
        self.module_version = module_version
        self.out_comps, self.out_services, self.imports = [], [], []
        self.keep_modules = set()
        self.report = {"world": self.world, "overview_camera": self.overview, "resources": [], "flags": [],
                       "expected_frames": {}}
        for c in self.comps:
            if c.get("frame") and c["frame"].get("parent") is None:
                c["frame"]["parent"] = "world"
                self.flag(c["name"], "needs_review", "frame parent unknown (unresolved variable)")
        self.frames = {c["name"]: c["frame"] for c in self.comps if c.get("frame")}
        self.arms = {}  # real arm name -> sim asset, or None when faked
        self.grippers = {}  # mapped gripper name -> its arm
        self.faked_mounts = set()  # arms and grippers faked for want of an asset
        self.gripper_arm = {}  # every real gripper name (mapped or faked) -> its frame parent
        names = {c["name"] for c in self.comps}
        self.gone = {c["name"] for c in self.comps if api_of(c, "component") not in FAKE_APIS
                     and module_of(c.get("model")) not in COPY_MODULES}
        self.present = names - self.gone

    def note(self, resource, kind, outcome, twin_model, note=None):
        entry = {"name": resource["name"], "api": api_of(resource, kind), "real_model": resource.get("model"),
                 "outcome": outcome, "twin_model": twin_model}
        if note:
            entry["note"] = note
        self.report["resources"].append(entry)

    def flag(self, name, flag, note):
        self.report["flags"].append({"name": name, "flag": flag, "note": note})

    def scrubbed(self, name, out, field="attributes"):
        """Remove secrets from out[field] in place and flag the removed paths (never values)."""
        if field in out:
            out[field], gone = scrub(out[field], field)
            if not out[field] and gone:
                del out[field]
            if gone:
                self.flag(name, "needs_review", f"removed: looks like a secret; give the twin its own value if it "
                          f"needs one ({', '.join(gone)})")

    def add(self, resource, model, frame=None, attributes=None):
        out = {"name": resource["name"], "api": api_of(resource, "component"), "model": model}
        if frame:
            out["frame"] = frame
            self.report["expected_frames"][resource["name"]] = frame
        if attributes:
            out["attributes"] = attributes
        self.out_comps.append(out)
        return out

    def lost(self, parent):
        return parent != "world" and parent not in self.present

    def fix_frame(self, c):
        """The component's frame, re-framed past a parent the twin doesn't have (and flagged)."""
        frame = copy.deepcopy(c.get("frame"))
        parent = (frame or {}).get("parent", "world")
        if frame and self.lost(parent):
            anchor, m = to_ancestor(c["name"], self.frames, stop=self.present)
            anchor = anchor if anchor in self.present else "world"
            frame = pose_frame(anchor, m)
            self.flag(c["name"], "reframed", f"was framed on {parent}, which the twin doesn't have; placed on {anchor}")
        return frame

    def fake(self, resource, note, outcome="needs_asset"):
        self.add(resource, FAKE, self.fix_frame(resource))
        self.note(resource, "component", outcome, FAKE, note)

    def arm(self, c):
        frame = copy.deepcopy(c.get("frame") or {"parent": "world"})
        outcome, note = "mapped", None
        if frame.get("parent", "world") != "world":
            _, m = to_ancestor(c["name"], self.frames, stop=set())
            note = f"was framed on {frame['parent']}; placed in the world at its zero, so any axis it rides is lost"
            frame, outcome = pose_frame("world", m), "reframed"
        attrs = ARMS.get(c.get("model"))
        if attrs is None:
            self.arms[c["name"]] = None
            self.faked_mounts.add(c["name"])
            self.add(c, FAKE, frame)
            self.note(c, "component", "needs_asset", FAKE, "no sim arm for this model: follow assets.md, then replace the fake")
            return
        self.arms[c["name"]] = attrs.get("asset")
        self.add(c, f"{SIM}:arm", frame, {"world": self.world, **attrs})
        self.note(c, "component", outcome, f"{SIM}:arm", note)

    def gripper(self, c):
        parent = (c.get("frame") or {}).get("parent", "world")
        entry = GRIPPERS.get(c.get("model"))
        asset = self.arms.get(parent)
        self.gripper_arm[c["name"]] = parent
        if entry is None:
            note = "no sim gripper for this model: follow assets.md, then replace the fake"
        elif parent not in self.arms:
            note = f"its frame parent {parent} isn't an arm"
        elif asset is None:
            note = f"its arm {parent} has no sim asset yet"
        elif entry[1] not in (None, asset):
            note = f"this gripper needs the {entry[1]} arm, not {asset}"
        else:
            self.grippers[c["name"]] = parent
            frame = copy.deepcopy(c["frame"])
            if entry[2] and not frame.get("geometry"):
                frame["geometry"] = copy.deepcopy(entry[2])
            self.add(c, f"{SIM}:gripper", frame, {"world": self.world, "mount": parent, **entry[0]})
            self.note(c, "component", "mapped", f"{SIM}:gripper")
            return
        self.faked_mounts.add(c["name"])
        self.fake(c, note)

    def other(self, c):
        if api_of(c, "component") == "rdk:component:camera" and module_of(c.get("model")) not in COPY_MODULES:
            self.camera(c)
        else:
            self.rest(c)

    def camera(self, c):
        name = c["name"]
        frame = copy.deepcopy(c.get("frame") or {"parent": "world"})
        parent = frame.get("parent", "world")
        if not c.get("frame"):
            self.flag(name, "needs_review", "no pose on the real machine; placed at the world origin")
        arm_of = self.gripper_arm.get(parent)
        on_mapped_arm = arm_of is not None and self.arms.get(arm_of) is not None
        if parent in self.faked_mounts and not on_mapped_arm:
            self.fake(c, f"rides {parent}, which has no sim asset yet")
            return
        outcome, note = "mapped", None
        if on_mapped_arm:
            arm, m = to_ancestor(name, self.frames, stop=set(self.arms))
            frame, outcome = pose_frame(arm, m), "reparented"
            note = f"was framed on gripper {parent}; framed on arm {arm} with the gripper's offset folded in"
        elif parent != "world" and parent not in self.arms:
            anchor, m = to_ancestor(name, self.frames, stop=set(self.arms))
            frame, outcome = pose_frame(anchor if anchor in self.arms else "world", m), "reframed"
            note = f"was framed on {parent}; placed at its zero"
            if self.lost(parent):
                self.flag(name, "reframed", f"was framed on {parent}, which the twin doesn't have; placed at its zero")
        live = self.live.get(name, {})
        ip = (c.get("attributes") or {}).get("intrinsic_parameters") or {}
        size = live.get("size") or ([ip["width_px"], ip["height_px"]] if "width_px" in ip and "height_px" in ip else None)
        fov = live.get("fov_deg") or (fov_deg(ip["height_px"], ip["fy"]) if "fy" in ip and "height_px" in ip else None)
        guessed = [k for k, v in (("size", size), ("fov_deg", fov)) if v is None]
        from_config = [k for k, v, lv in (("size", size, live.get("size")), ("fov_deg", fov, live.get("fov_deg")))
                       if v is not None and lv is None]
        if from_config:
            self.flag(name, "config_only", f"{' and '.join(from_config)} from config intrinsics, not a live reading")
        if guessed:
            self.flag(name, "guessed", f"{' and '.join(guessed)} defaulted: no live reading or config intrinsics")
        width, height = size or DEFAULT_SIZE
        attrs = {"world": self.world, "width": int(width), "height": int(height), "fov_deg": fov or DEFAULT_FOV}
        if live.get("depth"):
            attrs["depth"] = True
        if live.get("sources") and not set(live["sources"]) <= SIM_SOURCES:
            self.flag(name, "source_names_differ",
                      f"the real camera serves {live['sources']}; the twin serves color and depth")
        self.add(c, f"{SIM}:camera", frame, attrs)
        self.note(c, "component", outcome, f"{SIM}:camera", note)

    def rest(self, c):
        api, mod = api_of(c, "component"), module_of(c.get("model"))
        has_geometry = bool((c.get("frame") or {}).get("geometry"))
        if mod in COPY_MODULES:
            out = copy.deepcopy(c)
            out.pop("service_configs", None)
            self.scrubbed(c["name"], out)
            if out.get("frame"):
                out["frame"] = self.fix_frame(c)
            self.out_comps.append(out)
            if out.get("frame"):
                self.report["expected_frames"][c["name"]] = out["frame"]
            self.keep_modules.add(mod)
            if "obstacle" in c["model"].split(":")[-1]:  # ponytail: heuristic, name-based
                self.imports.append(c["name"])
            self.note(c, "component", "copied", c["model"])
        elif api in FAKE_APIS:
            self.fake(c, None, outcome="faked")
            if has_geometry:
                self.imports.append(c["name"])
        else:
            self.note(c, "component", "unsupported", None, f"no builtin fake for {api}")

    def service(self, s):
        api, mod = api_of(s, "service"), module_of(s.get("model"))
        component_modules = {module_of(c.get("model")) for c in self.comps} - self.keep_modules - {None}
        if api == "rdk:service:shell":
            self.note(s, "service", "removed", None, "a remote shell on the computer that runs the twin")
        elif api == "rdk:service:data_manager":
            self.note(s, "service", "removed", None, "sim images must not reach the real machine's data")
        elif mod in component_modules:
            if api in FAKE_APIS:
                self.out_services.append({"name": s["name"], "api": api, "model": FAKE})
                self.note(s, "service", "faked", FAKE, f"its module {mod} drives replaced hardware")
            else:
                self.note(s, "service", "unsupported", None, f"its module {mod} drives replaced hardware")
        else:
            out = copy.deepcopy(s)
            out.pop("service_configs", None)
            self.scrubbed(s["name"], out)
            self.out_services.append(out)
            if mod:
                self.keep_modules.add(mod)
            self.note(s, "service", "copied", s.get("model"))
            paths = local_paths(out.get("attributes"))
            if paths:
                self.flag(s["name"], "needs_review", f"local paths on the real host: {paths}")
            gone = {r["name"] for r in self.report["resources"] if r["outcome"] in ("unsupported", "removed")}
            needs = sorted(gone & set(_strings(s.get("attributes")) + _strings(s.get("depends_on"))))
            if needs:
                self.flag(s["name"], "needs_review", f"depends on {needs}, which the twin doesn't have")

    def _module(self, m):
        """A copy of a registry module entry without its reload_* keys. A reload build belongs to the real
        machine's part, so the twin can't fetch it, and one failed fetch stops every module from starting."""
        out = {k: copy.deepcopy(v) for k, v in m.items() if not k.startswith("reload_")}
        if "env" in out:
            out["env"], gone = scrub(out["env"], "env")
            if not out["env"]:
                del out["env"]
            if gone:
                self.flag(m["name"], "needs_review", f"removed: looks like a secret; give the twin its own value "
                          f"if it needs one ({', '.join(gone)})")
        if m.get("reload_enabled"):
            self.flag(m["name"], "needs_review",
                      f"ran a dev reload build on the real machine; the twin uses the registry's {m.get('version', 'latest')}")
        return out

    def finish(self, real):
        world = {"name": self.world, "api": "rdk:component:generic", "model": f"{SIM}:world", "attributes": {}}
        if self.imports:
            world["attributes"]["import_geometry"] = self.imports
        overview = {"name": self.overview, "api": "rdk:component:camera", "model": f"{SIM}:camera",
                    "frame": {"parent": "world", "translation": {"x": 1500, "y": 1500, "z": 1200}},
                    "attributes": {"world": self.world, "target": [0, 0, 0.3], "width": 1280, "height": 720}}
        self.out_comps[:0] = [world, overview]
        self.modules = [{"type": "registry", "name": "viam-labs_mujoco-sim", "module_id": "viam-labs:mujoco-sim",
                         "version": self.module_version}]
        replaced = {module_of(c.get("model")) for c in self.comps if module_of(c.get("model")) not in COPY_MODULES}
        for m in real.get("modules", []):
            if m.get("type") == "local":
                self.flag(m["name"], "needs_review", "a local module on the real host isn't copied")
            elif m.get("module_id") in self.keep_modules:
                self.modules.append(self._module(m))
            elif m.get("module_id") not in replaced:
                self.modules.append(self._module(m))
                self.flag(m["name"], "needs_review", "can't tell which resources this module serves; kept")
        for remote in real.get("remotes", []):
            self.report["resources"].append({"name": remote["name"], "api": "remote", "real_model": None,
                                             "outcome": "unsupported", "twin_model": None,
                                             "note": "remotes and other parts aren't twinned in v1"})
        dropped = [r["name"] for r in self.comps + real.get("services", []) if r.get("service_configs")]
        if dropped:
            self.flag(",".join(dropped), "data_capture_removed", "data capture configs dropped with the hardware")


DEFAULT_SIZE, DEFAULT_FOV = (640, 480), 60.0


def fov_deg(height_px, fy):
    """Vertical field of view (degrees) from intrinsics at their own image height."""
    return math.degrees(2 * math.atan(height_px / (2 * fy)))


def _strings(value):
    """Every string in a config value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _strings(v)]
    if isinstance(value, list):
        return [x for v in value for x in _strings(v)]
    return []


def local_paths(value):
    """String values in a config that look like local file paths."""
    if isinstance(value, str):
        return [value] if value.startswith(("/", "~", "./")) else []
    if isinstance(value, dict):
        return [p for v in value.values() for p in local_paths(v)]
    if isinstance(value, list):
        return [p for v in value for p in local_paths(v)]
    return []


def convert(real, live=None, module_version=None):
    """(twin config, report) for a real machine's expanded config. `live` maps a camera's name to what
    the real camera reported: {"size": [w, h], "fov_deg", "sources", "depth"}."""
    t = _Twin(real, live, module_version)
    # A resource disabled on the real machine is copied as it is and stays disabled: nothing runs it.
    disabled = [r for r in t.comps + real.get("services", []) if r.get("disabled")]
    for r in disabled:
        kind = "component" if r in t.comps else "service"
        if kind == "service" and api_of(r, kind) == "rdk:service:shell":
            t.note(r, kind, "removed", None, "a remote shell on the computer that runs the twin")
            continue
        out = {k: copy.deepcopy(v) for k, v in r.items() if k != "service_configs"}
        t.scrubbed(r["name"], out)
        (t.out_comps if kind == "component" else t.out_services).append(out)
        t.note(r, kind, "disabled", r.get("model"), "disabled on the real machine")
    t.comps = [c for c in t.comps if not c.get("disabled")]
    real = {**real, "services": [s for s in real.get("services", []) if not s.get("disabled")]}

    def by_api(api):  # obstacle-pack and other copy modules can use any API; they're copied, not mapped
        return [c for c in t.comps if api_of(c, "component") == api and module_of(c.get("model")) not in COPY_MODULES]
    for c in by_api("rdk:component:arm"):
        t.arm(c)
    for c in by_api("rdk:component:gripper"):
        t.gripper(c)
    handled = {e["name"] for e in t.report["resources"]}
    for c in t.comps:
        if c["name"] not in handled:
            t.other(c)
    for s in real.get("services", []):
        t.service(s)
    t.finish(real)
    packages = []
    for p in real.get("packages", []):
        p = copy.deepcopy(p)
        for k in [k for k in p if secret_name(k) and _strings(p[k])]:
            del p[k]
            t.flag(p.get("name", "package"), "needs_review", f"removed: looks like a secret; give the twin its "
                   f"own value if it needs one (packages.{k})")
        packages.append(p)
    return {"components": t.out_comps, "services": t.out_services, "modules": t.modules,
            "packages": packages}, t.report


# ---------- kinematics ----------

SVA_FORMAT, URDF_FORMAT = 1, 2
KIN_MM, KIN_DEG = 2.0, 1.0


def _axis_rot(axis, angle):
    x, y, z = _unit(np.asarray(axis, float))
    m = np.eye(4)
    m[:3, :3] = _mat([math.cos(angle / 2), *(math.sin(angle / 2) * np.array([x, y, z]))])
    return m


def _rpy(r, p, y):
    m = np.eye(4)
    m[:3, :3] = _axis_rot((0, 0, 1), y)[:3, :3] @ _axis_rot((0, 1, 0), p)[:3, :3] @ _axis_rot((1, 0, 0), r)[:3, :3]
    return m


def chain(data, fmt):
    """[(offset 4x4 mm, axis or None, (min, max) degrees or None, prismatic)] from the base to the
    last moving joint, from SVA JSON (fmt 1) or URDF (fmt 2)."""
    steps = _sva_chain(json.loads(data)) if fmt == SVA_FORMAT else _urdf_chain(data)
    moving = [i for i, s in enumerate(steps) if s[1] is not None]
    if not moving:
        raise ValueError("kinematics have no moving joints")
    last = moving[-1]
    return steps[: last + 1]


def _lower_keys(value):
    """RDK reads SVA JSON keys case-insensitively (Go), and real drivers write {"X": ...}."""
    if isinstance(value, dict):
        return {k.lower(): _lower_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_lower_keys(v) for v in value]
    return value


def _sva_chain(doc):
    doc = _lower_keys(doc)
    if (doc.get("kinematic_param_type") or "SVA") != "SVA":
        raise ValueError(f"kinematics type {doc['kinematic_param_type']} isn't supported")
    nodes = {}
    for link in doc.get("links", []):
        nodes[link["id"]] = (link.get("parent", "world"), frame_pose(link), None, None, False)
    for j in doc.get("joints", []):
        axis = [float(j["axis"].get(k, 0.0)) for k in "xyz"]
        limits = (float(j["min"]), float(j["max"])) if "min" in j and "max" in j else None
        nodes[j["id"]] = (j["parent"], np.eye(4), axis, limits, j.get("type") == "prismatic")
    children = {}
    for name, node in nodes.items():
        children.setdefault(node[0], []).append(name)
    steps, current = [], "world"
    while children.get(current):
        current = _deepest(children[current], children, lambda n: nodes[n][2] is not None)
        steps.append(nodes[current][1:])
    return steps


def _urdf_chain(data):
    from xml.etree import ElementTree

    root = ElementTree.fromstring(data)
    joints, children, child_links = {}, {}, set()
    for j in root.findall("joint"):
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        origin = j.find("origin")
        xyz = [float(v) * 1000 for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        offset = _rpy(*rpy)
        offset[:3, 3] = xyz
        kind = j.get("type")
        moving = kind in ("revolute", "continuous", "prismatic")
        axis = [float(v) for v in j.find("axis").get("xyz").split()] if moving and j.find("axis") is not None else (
            [1.0, 0.0, 0.0] if moving else None)
        limit = j.find("limit")
        limits = None
        if moving and kind != "continuous" and limit is not None:
            scale = 1000 if kind == "prismatic" else 180 / math.pi
            limits = (float(limit.get("lower", 0)) * scale, float(limit.get("upper", 0)) * scale)
        joints[j.get("name")] = (offset, axis, limits, kind == "prismatic", child)
        children.setdefault(parent, []).append(j.get("name"))
        child_links.add(child)
    roots = [link.get("name") for link in root.findall("link") if link.get("name") not in child_links]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {roots}")
    steps, link = [], roots[0]
    joint_children = {n: children.get(joints[n][4], []) for n in joints}
    while children.get(link):
        name = _deepest(children[link], joint_children, lambda n: joints[n][1] is not None)
        steps.append(joints[name][:4])
        link = joints[name][4]
    return steps


def _deepest(options, children, moving):
    """The option whose subtree holds the most moving joints (first on a tie)."""
    def count(n):
        return int(moving(n)) + sum(count(c) for c in children.get(n, []))
    return max(options, key=count)


def fk(steps, q):
    """Pose after the last step for joint values q (degrees, or mm for prismatic)."""
    m, i = np.eye(4), 0
    for offset, axis, _, prismatic in steps:
        m = m @ offset
        if axis is not None:
            if prismatic:
                move = np.eye(4)
                move[:3, 3] = _unit(np.asarray(axis, float)) * q[i]
                m = m @ move
            else:
                m = m @ _axis_rot(axis, math.radians(q[i]))
            i += 1
    return m


def compare_kinematics(real, twin):
    """{"ok", "joints", "limits", "poses"} for two chains from chain()."""
    n_real = sum(s[1] is not None for s in real)
    n_twin = sum(s[1] is not None for s in twin)
    result = {"joints": [n_real, n_twin], "limits": [], "poses": []}
    if n_real != n_twin:
        return {**result, "ok": False}
    real_limits = [s[2] for s in real if s[1] is not None]
    twin_limits = [s[2] for s in twin if s[1] is not None]
    prismatic = [s[3] for s in real if s[1] is not None]
    for i, (a, b) in enumerate(zip(real_limits, twin_limits, strict=True)):
        if a and b and max(abs(a[0] - b[0]), abs(a[1] - b[1])) > (KIN_MM if prismatic[i] else KIN_DEG):
            result["limits"].append({"joint": i, "real": [round(v, 2) for v in a], "twin": [round(v, 2) for v in b]})
    ranges = [a or (-180.0, 180.0) for a in real_limits]
    configs = [[0.0] * n_real]
    for frac in (0.25, 0.75, 0.4, 0.6):
        configs.append([lo + (hi - lo) * (frac if i % 2 == 0 else 1 - frac) for i, (lo, hi) in enumerate(ranges)])
    for q in configs:
        mm, deg = pose_error(fk(real, q), fk(twin, q))
        result["poses"].append({"q": [round(v, 1) for v in q], "mm": round(mm, 2), "deg": round(deg, 2)})
    ok = not result["limits"] and all(p["mm"] <= KIN_MM and p["deg"] <= KIN_DEG for p in result["poses"])
    return {**result, "ok": ok}


# ---------- checks ----------

FRAME_MM, FRAME_DEG = 1.0, 0.5
EDGE_COLOUR = (255, 0, 255)


def compare_names(report, live):
    """Every real resource the twin should have, by (api, name), against the twin's live set."""
    want = {(r["api"], r["name"]) for r in report["resources"] if r["outcome"] not in ("removed", "unsupported", "disabled")}
    missing = sorted([list(x) for x in want - set(live)])
    return {"ok": not missing, "missing": missing}


def compare_frames(expected, live):
    """Config frames {name: frame} against the twin's frame system {name: (parent, 4x4)}."""
    deltas = []
    for name, frame in sorted(expected.items()):
        if name not in live:
            deltas.append({"name": name, "missing": True})
            continue
        parent, m = live[name]
        mm, deg = pose_error(frame_pose(frame), m)
        if (parent != frame.get("parent", "world") or not (math.isfinite(mm) and math.isfinite(deg))
                or mm > FRAME_MM or deg > FRAME_DEG):
            deltas.append({"name": name, "parent": [frame.get("parent", "world"), parent],
                           "mm": round(mm, 2), "deg": round(deg, 2)})
    return {"ok": not deltas, "deltas": deltas}


def proto_pose(pose):
    """4x4 (mm) of a viam Pose proto (x, y, z mm; orientation vector, theta degrees)."""
    return frame_pose({"translation": {"x": pose.x, "y": pose.y, "z": pose.z},
                       "orientation": {"type": "ov_degrees",
                                       "value": {"x": pose.o_x, "y": pose.o_y, "z": pose.o_z, "th": pose.theta}}})


def edge_overlay(real_path, sim_path, out_path, threshold=40):
    """The real image with the sim image's edges drawn over it in EDGE_COLOUR."""
    real = Image.open(real_path).convert("RGB")
    if min(real.size) < 3:  # a placeholder image from a point-cloud camera: nothing to compare
        raise ValueError(f"the real image is {real.size[0]}x{real.size[1]}: too small to overlay")
    sim = Image.open(sim_path).convert("L").resize(real.size)
    mask = sim.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > threshold else 0)
    mask = ImageOps.expand(ImageOps.crop(mask, 1), 1, fill=0)  # FIND_EDGES leaves the border unfiltered
    out = real.copy()
    out.paste(Image.new("RGB", real.size, EDGE_COLOUR), mask=mask)
    out.save(out_path)


# ---------- I/O ----------


def cli(*args):
    """stdout of a `viam` CLI command, run with the user's login."""
    done = subprocess.run(["viam", *args], capture_output=True, text=True, check=False)
    if done.returncode:
        raise SystemExit(json.dumps({"failed": "viam " + " ".join(args[:3]), "error": done.stderr.strip()[-500:]}))
    return done.stdout


def main_part_id(listing):
    """The main part's ID from `viam machines part list` output."""
    found = re.search(r"ID:\s*(\S+)\s*\n\s*Name:[^\n]*\(main\)", listing)
    if not found:
        raise ValueError("no main part in `viam machines part list` output")
    return found.group(1)


def read_real_config(machine_id):
    """(expanded config, unapplied mods) of a machine's main part, through the CLI."""
    part = main_part_id(cli("machines", "part", "list", "--machine", machine_id))
    raw = json.loads(cli("machines", "part", "config", "--part", part))

    def fetch(fid, version):
        return json.loads(cli("fragment", "get", "--fragment", fid, *(["--version", version] if version else [])))

    return expand(raw, fetch)


async def pull_live(robot, real, out_dir):
    """Read-only live values from the real machine. Saves real_<camera>.jpg, kin_<arm>.<json|urdf>
    and joints.json in out_dir; returns {"cameras": {...}, "files": {...}, "errors": {...}}."""
    from viam.components.arm import Arm
    from viam.components.camera import Camera

    out_dir = Path(out_dir)
    cameras, files, errors, joints, frames = {}, {"images": {}, "kinematics": {}}, {}, {}, {}
    try:
        for f in await robot.get_frame_system_config():
            p = f.frame.pose_in_observer_frame
            pose = p.pose
            frames[f.frame.reference_frame] = {
                "parent": p.reference_frame,
                "translation": {"x": pose.x, "y": pose.y, "z": pose.z},
                "orientation": {"type": "ov_degrees", "value": {
                    "x": pose.o_x, "y": pose.o_y, "z": pose.o_z, "th": pose.theta}}}
    except Exception as e:
        errors["frame_system"] = str(e)[:300]
    for c in real.get("components", []):
        name, api = c["name"], api_of(c, "component")
        try:
            if api == "rdk:component:camera":
                cam = Camera.from_robot(robot, name)
                entry = {}
                ip = (await cam.get_properties()).intrinsic_parameters
                if ip and ip.focal_y_px and ip.height_px:
                    entry["fov_deg"] = fov_deg(ip.height_px, ip.focal_y_px)
                cameras[name] = entry
                images, _ = await cam.get_images()
                entry["sources"] = [i.name for i in images if i.name]
                if any("dep" in str(i.mime_type) for i in images):
                    entry["depth"] = True
                colour = next((i for i in images if "dep" not in str(i.mime_type)), None)
                if colour is not None:
                    img = Image.open(io.BytesIO(colour.data)).convert("RGB")
                    path = out_dir / f"real_{name}.jpg"
                    img.save(path, quality=92)
                    files["images"][name] = path.name
                    entry["size"] = list(img.size)
            elif api == "rdk:component:arm":
                arm = Arm.from_robot(robot, name)
                fmt, data = (await arm.get_kinematics())[:2]
                path = out_dir / f"kin_{name}.{'urdf' if fmt == URDF_FORMAT else 'json'}"
                path.write_bytes(data)
                files["kinematics"][name] = [path.name, int(fmt)]
                joints[name] = list((await arm.get_joint_positions()).values)
        except Exception as e:  # one bad resource mustn't stop the rest
            errors[name] = str(e)[:300]
    (out_dir / "joints.json").write_text(json.dumps(joints))
    files["joints"] = "joints.json"
    return {"cameras": cameras, "files": files, "errors": errors, "frames": frames}
