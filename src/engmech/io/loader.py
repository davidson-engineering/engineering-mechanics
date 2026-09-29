"""Load model files (YAML) into :class:`~engmech.model.Model` objects."""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError

from engmech import joints as jt
from engmech import shapes as sh
from engmech.errors import InputError
from engmech.io import schema as s
from engmech.loads import DistributedLoad, Force, Moment, Motion, UnknownLoad
from engmech.model import Check, Model


@dataclass
class SourceMap:
    """Maps dotted input paths (``supports.A.normal``) to file positions."""

    path: str
    data: Any = None

    def locate(self, where: str | Sequence[Any] | None) -> tuple[int, int] | None:
        if where is None or self.data is None:
            return None
        keys = _split_path(where) if isinstance(where, str) else list(where)
        node, pos = self.data, None
        for key in keys:
            child_pos = _position(node, key)
            if child_pos is None:
                continue  # e.g. a pydantic union tag that is not in the data
            pos = child_pos
            node = node[key]
        return pos

    def describe(self, where) -> str:
        pos = self.locate(where)
        return f"{self.path}:{pos[0]}:{pos[1]}" if pos else self.path


class ModelFileError(InputError):
    """One or more problems in a model file, each with a file position."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


def _split_path(where: str) -> list[Any]:
    keys: list[Any] = []
    for part in re.split(r"\.(?![^\[]*\])", where.split(" ")[0]):
        m = re.fullmatch(r"([^\[]*)((?:\[\d+\])*)", part)
        if not m:
            keys.append(part)
            continue
        if m.group(1):
            keys.append(m.group(1))
        keys += [int(i) for i in re.findall(r"\[(\d+)\]", m.group(2))]
    return keys


def _position(node: Any, key: Any) -> tuple[int, int] | None:
    try:
        if isinstance(node, Mapping) and key in node:
            line, col = node.lc.key(key)
            return line + 1, col + 1
        if isinstance(node, list) and isinstance(key, int) and 0 <= key < len(node):
            line, col = node.lc.item(key)
            return line + 1, col + 1
    except (AttributeError, KeyError, TypeError):
        return None
    return None


# --------------------------------------------------------------------------- parsing


def read_yaml(text: str, path: str = "<input>") -> Any:
    yaml = YAML(typ="rt")
    try:
        return yaml.load(text)
    except MarkedYAMLError as exc:
        mark = exc.problem_mark
        where = f"{path}:{mark.line + 1}:{mark.column + 1}" if mark else path
        hint = ""
        line = text.splitlines()[mark.line] if mark and mark.line < len(text.splitlines()) else ""
        if re.search(r"[\]\)]\s*[A-Za-z]", line):
            hint = ' (a vector with a unit after the brackets must be quoted: "[0, -10] kN")'
        raise ModelFileError([f"{where}: invalid YAML: {exc.problem}{hint}"]) from None


def load_model(path: str | Path) -> Model:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror}") from None
    return loads_model(text, str(path))


def loads_model(text: str, path: str = "<input>") -> Model:
    data = read_yaml(text, path)
    source = SourceMap(path, data)
    if data is None:
        raise ModelFileError([f"{path}: the file is empty"])
    if not isinstance(data, Mapping):
        raise ModelFileError([f"{path}: the top level must be a mapping of sections"])
    try:
        spec = s.ModelFile.model_validate(data)
    except ValidationError as exc:
        raise ModelFileError(_format_validation(exc, source)) from None
    try:
        model = build_model(spec)
    except InputError as exc:
        raise ModelFileError([f"{source.describe(exc.where)}: {exc}"]) from None
    model.source = source
    return model


def _all_field_names() -> set[str]:
    names: set[str] = set()
    stack: list[type[BaseModel]] = [s.ModelFile]
    seen = set()
    for cls in vars(s).values():
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            stack.append(cls)
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        names |= set(cls.model_fields)
    return names


def _format_validation(exc: ValidationError, source: SourceMap) -> list[str]:
    known = _all_field_names()
    problems = []
    for err in exc.errors():
        loc = list(err["loc"])
        dotted = ".".join(str(k) if not isinstance(k, int) else f"[{k}]" for k in loc)
        dotted = dotted.replace(".[", "[")
        kind = err["type"]
        if kind == "extra_forbidden":
            name = str(loc[-1])
            guess = difflib.get_close_matches(name, known, n=1)
            msg = f"unknown field {name!r}" + (f" (did you mean {guess[0]!r}?)" if guess else "")
            parent = dotted.rsplit(".", 1)[0] if "." in dotted else ""
            dotted = parent or dotted
        elif kind == "missing":
            msg = f"missing required field {str(loc[-1])!r}"
            dotted = ".".join(dotted.split(".")[:-1]) or dotted
        elif kind in ("union_tag_invalid", "union_tag_not_found"):
            msg = err["msg"].replace("Input tag", "type").replace(" using 'type'", "")
            if kind == "union_tag_not_found":
                msg = "missing 'type' (e.g. type: pin)"
        else:
            msg = err["msg"]
        problems.append(f"{source.describe(loc)}: {dotted}: {msg}")
    return problems


# --------------------------------------------------------------------------- building


def _fields(spec: BaseModel, *drop: str) -> dict[str, Any]:
    return {
        k: v
        for k, v in spec.model_dump(exclude_none=False).items()
        if k not in drop and k in spec.model_fields_set | _required(spec)
    }


def _required(spec: BaseModel) -> set[str]:
    return {k for k, f in type(spec).model_fields.items() if f.is_required()}


def build_model(spec: s.ModelFile) -> Model:
    gravity, gravity_case = spec.gravity, None
    if isinstance(gravity, Mapping) and "case" in gravity:
        gravity = dict(gravity)
        gravity_case = str(gravity.pop("case"))
        if set(gravity) == {"vector"}:
            gravity = gravity["vector"]
    model = Model(
        spec.name,
        planar=spec.analysis == "planar",
        units=_plain(spec.units),
        output_units=_plain(spec.output_units),
        parameters=dict(spec.parameters),
        description=spec.description,
        gravity=_plain(gravity),
        gravity_case=gravity_case,
    )
    for name, value in spec.points.items():
        try:
            model.point(name, _plain(value))
        except InputError as exc:
            raise exc.at(f"points.{name}") from None

    for name, body in spec.bodies.items():
        body = body or s.BodySpec()
        where = f"bodies.{name}"
        shapes = []
        for shape in body.shapes:
            data = _plain(_fields(shape, "type"))
            cls = sh.Cylinder if shape.type == "disc" else sh.SHAPES[shape.type]
            shapes.append(cls(**data))
        motion = None
        if body.motion is not None:
            motion = Motion(**_plain(body.motion.model_dump()))
        try:
            model.body(
                name,
                mass=_plain(body.mass),
                cog=_plain(body.cog),
                inertia=_plain(body.inertia),
                shapes=shapes,
                particle=body.particle,
                motion=motion,
                outline=_plain(body.outline),
            )
        except InputError as exc:
            raise exc.at(where) from None
        for i, load in enumerate(body.loads):
            if load.body is not None and load.body != name:
                raise InputError(f"this load is listed under body {name!r}", f"{where}.loads[{i}]")
            _add_load(model, load, name, f"{where}.loads[{i}]")

    for section, entries in (("supports", spec.supports), ("joints", spec.joints)):
        for name, joint in entries.items():
            where = f"{section}.{name}"
            data = _plain(_fields(joint, "type", "body", "bodies"))
            cls = jt.JOINT_TYPES[joint.type]
            if section == "supports":
                if joint.bodies is not None:
                    raise InputError(
                        "a support connects one 'body' to the ground; "
                        "use 'joints' to connect two bodies",
                        where,
                    )
                if joint.type in ("link", "strut", "cable") and "ends" in data:
                    raise InputError("a link support takes 'at' and 'anchor'", where)
                model.support(name, cls(**data), body=joint.body)
            else:
                if joint.body is not None or joint.bodies is None:
                    raise InputError("a joint needs 'bodies: [a, b]'", where)
                if joint.type in ("link", "strut", "cable") and ("at" in data or "anchor" in data):
                    raise InputError("a link between bodies takes 'ends: [on a, on b]'", where)
                try:
                    model.joint(name, cls(**data), bodies=joint.bodies)
                except InputError as exc:
                    raise exc.at(where) from None
            model._joints[name].where = where

    for i, load in enumerate(spec.loads):
        _add_load(model, load, load.body, f"loads[{i}]")

    for name, factors in spec.combinations.items():
        model.combination(name, _plain(factors))
    for c in spec.checks:
        model.check(Check(**_plain(c.model_dump())))
    return model


def _add_load(model: Model, load: s.LoadSpec, body: str | None, where: str) -> None:
    given = [
        k for k in ("force", "moment", "distributed", "unknown") if getattr(load, k) is not None
    ]
    if len(given) != 1:
        raise InputError(
            "each load needs exactly one of force, moment, distributed or unknown", where
        )
    kind = given[0]
    if kind != "unknown" and (load.direction is not None or load.axis is not None):
        raise InputError("'direction'/'axis' belong inside the force, or on an unknown load", where)
    if kind == "force":
        if load.at is None:
            raise InputError("a force needs 'at'", where)
        item = Force(_plain(load.force), _plain(load.at), load.name, load.case)
    elif kind == "moment":
        item = Moment(_plain(load.moment), _plain(load.at), load.name, load.case)
    elif kind == "distributed":
        if load.at is not None:
            raise InputError("a distributed load takes 'start' and 'end', not 'at'", where)
        d = load.distributed
        item = DistributedLoad(
            _plain(d.start),
            _plain(d.end),
            _plain(d.intensity),
            _plain(d.direction),
            load.name,
            load.case,
            d.projected,
        )
    else:
        if load.case is not None:
            raise InputError("an unknown load is solved for, so it has no load case", where)
        try:
            item = UnknownLoad(
                load.unknown, _plain(load.at), _plain(load.direction), _plain(load.axis)
            )
        except InputError as exc:
            raise exc.at(where) from None
    model.load(item, body=body, where=where)


def _plain(value: Any) -> Any:
    """Convert ruamel containers and scalars to plain Python types."""
    if isinstance(value, BaseModel):
        return _plain(value.model_dump())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    return value
