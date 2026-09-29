"""Unit-aware parsing and formatting.

Everything inside engmech is stored as plain floats in SI (m, N, kg, s, rad).
Units only exist at the boundaries: this module turns user input such as
``"10 kN"``, ``"[0, -2.5] kN/m"`` or ``"L/2"`` into SI floats, checking the
physical dimension on the way, and formats SI floats back into the user's
preferred units for display.
"""

from __future__ import annotations

import ast
import io
import keyword
import math
import re
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pint

from engmech.errors import InputError

ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"
ureg.formatter.default_sort_func = None  # keep units in the order written: N·m, not m·N
Q_ = ureg.Quantity

STANDARD_GRAVITY = 9.80665  # m/s^2


# --------------------------------------------------------------------------- kinds


@dataclass(frozen=True)
class Kind:
    """A physical quantity kind, e.g. force or length.

    ``si`` is the SI unit engmech stores values in. ``derive`` builds the
    default display unit from a unit system's base units.
    """

    name: str
    si: str
    derive: str  # format string over base units, e.g. "{force}*{length}"

    @property
    def dimensionality(self):
        return ureg.Unit(self.si).dimensionality


KINDS: dict[str, Kind] = {
    k.name: k
    for k in [
        Kind("length", "m", "{length}"),
        Kind("force", "N", "{force}"),
        Kind("moment", "N*m", "{force}*{length}"),
        Kind("mass", "kg", "{mass}"),
        Kind("angle", "rad", "{angle}"),
        Kind("time", "s", "{time}"),
        Kind("velocity", "m/s", "{length}/{time}"),
        Kind("acceleration", "m/s**2", "{length}/{time}**2"),
        Kind("angular_velocity", "rad/s", "rad/{time}"),
        Kind("angular_acceleration", "rad/s**2", "rad/{time}**2"),
        Kind("density", "kg/m**3", "{mass}/{length}**3"),
        Kind("linear_density", "kg/m", "{mass}/{length}"),
        Kind("inertia", "kg*m**2", "{mass}*{length}**2"),
        Kind("force_per_length", "N/m", "{force}/{length}"),
        Kind("stiffness", "N/m", "{force}/{length}"),
        Kind("rotational_stiffness", "N*m/rad", "{force}*{length}/rad"),
        Kind("dimensionless", "", "1"),
    ]
}


def kind(name: str) -> Kind:
    try:
        return KINDS[name]
    except KeyError:
        raise ValueError(f"unknown quantity kind {name!r}") from None


# --------------------------------------------------------------------------- systems

BASE_KINDS = ("length", "force", "mass", "angle", "time")

PRESETS: dict[str, dict[str, str]] = {
    "SI": {"length": "m", "force": "N", "mass": "kg", "angle": "deg", "time": "s"},
    "SI-kN": {"length": "m", "force": "kN", "mass": "kg", "angle": "deg", "time": "s"},
    "SI-mm": {"length": "mm", "force": "N", "mass": "kg", "angle": "deg", "time": "s"},
    "US-in": {"length": "in", "force": "lbf", "mass": "lb", "angle": "deg", "time": "s"},
    "US-ft": {"length": "ft", "force": "lbf", "mass": "lb", "angle": "deg", "time": "s"},
}


@dataclass(frozen=True)
class UnitSystem:
    """Units used to interpret bare numbers and to display results.

    Base units cover length, force, mass, angle and time. Every other kind
    derives from them (moment = force*length, and so on) unless overridden.
    """

    length: str = "m"
    force: str = "N"
    mass: str = "kg"
    angle: str = "deg"
    time: str = "s"
    overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for name in BASE_KINDS:
            unit = _normalize(str(getattr(self, name)))
            _check_unit(unit, kind(name), f"units.{name}")
            object.__setattr__(self, name, unit)
        overrides = {}
        for name, unit in self.overrides.items():
            unit = _normalize(str(unit))
            _check_unit(unit, kind(name), f"units.{name}")
            overrides[name] = unit
        object.__setattr__(self, "overrides", overrides)

    @classmethod
    def from_spec(cls, spec: UnitSystem | str | Mapping[str, str] | None) -> UnitSystem:
        """Build from a preset name, a mapping of kind -> unit, or None (SI)."""
        if spec is None:
            return cls()
        if isinstance(spec, UnitSystem):
            return spec
        if isinstance(spec, str):
            if spec not in PRESETS:
                raise InputError(
                    f"unknown unit system {spec!r}; choose one of {', '.join(PRESETS)}"
                )
            return cls(**PRESETS[spec])
        spec = dict(spec)
        base = dict(PRESETS[spec.pop("system")]) if "system" in spec else {}
        if "system" not in spec and "mass" not in spec and "force" in spec:
            # imperial forces imply imperial mass unless told otherwise
            try:
                if ureg.Unit(_normalize(str(spec["force"]))) in (ureg.lbf, ureg.kip, ureg.ozf):
                    base.setdefault("mass", "lb")
            except Exception:  # invalid units are reported by __post_init__
                pass
        overrides = {}
        for name, unit in spec.items():
            if name in BASE_KINDS:
                base[name] = unit
            elif name in KINDS:
                overrides[name] = unit
            else:
                raise InputError(
                    f"units: unknown quantity {name!r}; expected one of "
                    f"{', '.join(['system', *KINDS])}"
                )
        return cls(**base, overrides=overrides)

    def with_overrides(self, **units: str) -> UnitSystem:
        return replace(self, overrides={**self.overrides, **units})

    def unit(self, kind_name: str) -> str:
        """The unit string this system uses for a quantity kind."""
        if kind_name in self.overrides:
            return self.overrides[kind_name]
        k = kind(kind_name)
        return k.derive.format(**{b: f"({getattr(self, b)})" for b in BASE_KINDS})

    def factor(self, kind_name: str) -> float:
        """Multiply an SI value by this to express it in this system's unit."""
        k = kind(kind_name)
        if k.name == "dimensionless":
            return 1.0
        return float(Q_(1.0, k.si).to(self.unit(kind_name)).magnitude)

    def label(self, kind_name: str) -> str:
        """Pretty unit label for tables, e.g. 'kN·m'."""
        if kind_name == "dimensionless":
            return ""
        return f"{ureg.Unit(self.unit(kind_name)):~P}"

    def format(self, value_si: float, kind_name: str, digits: int = 4, unit: bool = True) -> str:
        text = format_number(float(value_si) * self.factor(kind_name), digits)
        label = self.label(kind_name)
        return f"{text} {label}" if unit and label else text

    def unit_for(self, q: pint.Quantity, prefer: str | None = None) -> str | None:
        """This system's unit for the dimension of ``q`` (for bare numbers in
        sums). ``prefer`` names the kind being parsed, which wins when several
        kinds share a dimension (moment and rotational stiffness, say)."""
        if q.dimensionless:
            return self.angle if _is_angle(q) else None
        if prefer and prefer in KINDS and KINDS[prefer].dimensionality == q.dimensionality:
            return self.unit(prefer)
        for k in KINDS.values():
            if k.name in ("dimensionless", "angle"):
                continue
            if k.dimensionality == q.dimensionality:
                return self.unit(k.name)
        return None


def _check_unit(unit: str, k: Kind, where: str) -> None:
    try:
        u = ureg.Unit(unit)
    except Exception as exc:
        raise InputError(f"{where}: unknown unit {unit!r}") from exc
    if k.name == "angle":
        if not u.dimensionless:
            raise InputError(f"{where}: {unit!r} is not an angle unit")
        return
    if u.dimensionality != k.dimensionality:
        raise InputError(f"{where}: {unit!r} is not a {k.name.replace('_', ' ')} unit")


def _is_angle(q: pint.Quantity) -> bool:
    """A dimensionless quantity carrying an angle unit (deg, rad, rev ...)."""
    if q.unitless or not q.dimensionless:
        return False
    return "radian" in {str(u) for u in q.to_root_units().units._units}


def format_number(x: float, digits: int = 4) -> str:
    """Format a number for engineering tables: ~4 significant figures,
    no scientific notation in the everyday range, and no '-0'."""
    if not math.isfinite(x):
        return str(x)
    if x == 0 or abs(x) < 1e-12:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e7:
        decimals = max(0, digits - 1 - math.floor(math.log10(ax)))
        text = f"{x:,.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text
    return f"{x:.{digits - 1}e}"


# --------------------------------------------------------------------------- expressions

_FUNCTIONS = {
    "sin": ("angle", math.sin),
    "cos": ("angle", math.cos),
    "tan": ("angle", math.tan),
    "asin": ("number", math.asin),
    "acos": ("number", math.acos),
    "atan": ("number", math.atan),
    "sqrt": ("any", None),
    "abs": ("any", None),
    "atan2": ("pair", None),
    "hypot": ("pair", None),
}
_CONSTANTS = {"pi": math.pi}
# Python keywords that are also unit names; the tokenizer renames them.
_KEYWORD_UNITS = {"in": "inch"}
_FUNCTION_NAMES = set(_FUNCTIONS)


def _normalize(text: str) -> str:
    return (
        text.replace("·", "*")
        .replace("⋅", "*")
        .replace("×", "*")
        .replace("^", "**")
        .replace("°", " deg")
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("²", "**2")
        .replace("³", "**3")
    )


UNIT_PREFIX = "__unit__"


def _prepare(text: str, units_only: bool = False) -> str:
    """Make an expression valid Python and decide which names are units.

    * implicit multiplication: '10 kN' -> '10 * kN', '2 N m' -> '2 * N * m'
    * a name written directly after a number or ')' is a unit, and so is
      every name chained to it by '*', '/' or '**' without spaces
      ('9.81 m/s**2', '5 kN*m'). Such names never resolve to parameters, so
      a parameter called m cannot turn '2 m' into '2 * mass'.
    * a number with its units is one quantity: '10 kN / 2 m' is
      (10 kN) / (2 m), not (10 kN / 2) * m.
    """
    try:
        tokens = [
            t
            for t in tokenize.generate_tokens(io.StringIO(text).readline)
            if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT)
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise InputError(f"cannot parse {text!r}") from exc
    out: list[str] = []
    prev = None
    unit_next = False  # a NAME here would be a unit
    in_chain = False  # the last name was a unit
    after_pow = False
    for i, tok in enumerate(tokens):
        string = tok.string
        if tok.type == tokenize.NAME and keyword.iskeyword(string):
            if string not in _KEYWORD_UNITS:
                raise InputError(f"cannot use {string!r} in {text!r}")
            string = _KEYWORD_UNITS[string]
        if prev is not None:
            prev_value = prev.type in (tokenize.NUMBER, tokenize.NAME) or prev.string == ")"
            starts_value = tok.type == tokenize.NAME or string == "("
            is_call = (
                string == "(" and prev.type == tokenize.NAME and (prev.string in _FUNCTION_NAMES)
            )
            if prev_value and starts_value and not is_call:
                out.append("*")
        call_syntax = (
            tok.type == tokenize.NAME
            and i + 1 < len(tokens)
            and tokens[i + 1].string == "("
            and tokens[i + 1].start == tok.end
        )
        if call_syntax and string not in _FUNCTION_NAMES and not unit_next:
            raise InputError(
                f"unknown function {string!r} in {text!r}; available: "
                f"{', '.join(sorted(_FUNCTION_NAMES))} (for a product write {string}*(...))"
            )
        is_function = call_syntax and string in _FUNCTION_NAMES
        if tok.type == tokenize.NAME and not is_function:
            if units_only or unit_next:
                out.append(UNIT_PREFIX + string)
                in_chain, unit_next = True, True
            else:
                out.append(string)
                in_chain, unit_next = False, False
            after_pow = False
        else:
            out.append(string)
            tight = prev is not None and prev.end == tok.start
            if tok.type == tokenize.NUMBER:
                unit_next = in_chain if after_pow else True
                in_chain = in_chain and after_pow
                after_pow = False
            elif string in ("*", "/"):
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                unit_next = in_chain and tight and nxt is not None and nxt.start == tok.end
                after_pow = False
            elif string == "**":
                unit_next = False
                after_pow = in_chain
            elif string == ")":
                unit_next, in_chain, after_pow = True, False, False
            else:
                unit_next, in_chain, after_pow = False, False, False
        prev = tok
    return " ".join(_group_quantities(out))


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _chain_step(out: list[str], j: int) -> int:
    """How many tokens at ``j`` continue a unit chain: '* unit', '/ unit',
    '** 2' or '** -2'; 0 if the chain ends."""
    n = len(out)
    if j + 1 >= n:
        return 0
    if out[j] in ("*", "/") and out[j + 1].startswith(UNIT_PREFIX):
        return 2
    if out[j] == "**":
        if _is_number(out[j + 1]):
            return 2
        if out[j + 1] in ("+", "-") and j + 2 < n and _is_number(out[j + 2]):
            return 3
    return 0


def _group_quantities(out: list[str]) -> list[str]:
    """Parenthesise each number with its unit chain: '10 * kN / 2 * m' becomes
    '(10 * kN) / (2 * m)', so implicit multiplication binds tighter than '/'."""
    res: list[str] = []
    i, n = 0, len(out)
    while i < n:
        starts = (
            _is_number(out[i])
            and not (res and res[-1] == "**")
            and i + 2 < n
            and out[i + 1] == "*"
            and out[i + 2].startswith(UNIT_PREFIX)
        )
        if not starts:
            res.append(out[i])
            i += 1
            continue
        j = i + 3
        while (step := _chain_step(out, j)) > 0:
            j += step
        res += ["(", *out[i:j], ")"]
        i = j
    return res


def evaluate(
    text: str,
    parameters: Mapping[str, Any] | None = None,
    units_only: bool = False,
    units: UnitSystem | None = None,
    kind: str | None = None,
) -> pint.Quantity:
    """Safely evaluate an arithmetic expression with units and parameters.

    Supports + - * / ** ^, parentheses, numbers, unit names, parameter names,
    ``pi`` and the functions sin, cos, tan, asin, acos, atan, atan2, sqrt,
    abs and hypot. Trigonometric functions require angles with units
    (``sin(30 deg)``) so that degrees and radians can never be confused.

    A name directly after a number is always a unit (``2 m`` is two metres);
    elsewhere a parameter of that name wins (``2*m`` uses parameter m).

    With a unit system, a bare number added to a quantity takes that system's
    unit for the quantity's dimension: with lengths in mm, ``r + 3`` is r + 3 mm.
    """
    parameters = parameters or {}
    source = _prepare(_normalize(text.strip()), units_only)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise InputError(f"cannot parse {text!r}") from exc
    return _Evaluator(text, parameters, units, kind).visit(tree.body)


class _Evaluator:
    def __init__(
        self,
        text: str,
        parameters: Mapping[str, Any],
        units: UnitSystem | None = None,
        kind: str | None = None,
    ):
        self.kind = kind
        self.text = text
        self.parameters = parameters
        self.units = units

    def _match_bare(self, left: pint.Quantity, right: pint.Quantity):
        """Give a bare number the unit system's unit of the other term."""
        if self.units is None or left.unitless == right.unitless:
            return left, right
        bare, other = (left, right) if left.unitless else (right, left)
        unit = self.units.unit_for(other, prefer=self.kind)
        if unit is None:
            return left, right
        bare = Q_(bare.to("dimensionless").magnitude, unit)
        return (bare, other) if left.unitless else (other, bare)

    def fail(self, message: str) -> InputError:
        return InputError(f"in {self.text!r}: {message}")

    def visit(self, node: ast.AST) -> pint.Quantity:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise self.fail(f"unsupported syntax ({type(node).__name__})")
        return method(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise self.fail(f"unexpected value {node.value!r}")
        return Q_(float(node.value))

    def visit_Name(self, node: ast.Name):
        name = node.id
        if name.startswith(UNIT_PREFIX):
            name = name[len(UNIT_PREFIX) :]
            try:
                return Q_(1.0, ureg.Unit(name))
            except Exception:
                raise self.fail(f"unknown unit {name!r}") from None
        if name in self.parameters:
            return as_quantity(self.parameters[name])
        if name in _CONSTANTS:
            return Q_(_CONSTANTS[name])
        try:
            return Q_(1.0, ureg.Unit(name))
        except Exception:
            raise self.fail(f"unknown name {name!r} (not a parameter or a unit)") from None

    def visit_UnaryOp(self, node: ast.UnaryOp):
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise self.fail("unsupported operator")

    def visit_BinOp(self, node: ast.BinOp):
        left, right = self.visit(node.left), self.visit(node.right)
        try:
            match node.op:
                case ast.Add():
                    left, right = self._match_bare(left, right)
                    return left + right
                case ast.Sub():
                    left, right = self._match_bare(left, right)
                    return left - right
                case ast.Mult():
                    return left * right
                case ast.Div():
                    return left / right
                case ast.Pow():
                    if not right.unitless:
                        raise self.fail("exponent must be a plain number")
                    exponent = float(right.to("dimensionless").magnitude)
                    if left.magnitude < 0 and not exponent.is_integer():
                        raise self.fail("a negative number to a fractional power is undefined")
                    return left**exponent
        except pint.DimensionalityError as exc:
            raise self.fail(
                f"cannot add or subtract {exc.units1} and {exc.units2}; "
                "give every term a unit, e.g. '3 m + r'"
            ) from None
        except ZeroDivisionError:
            raise self.fail("division by zero") from None
        raise self.fail("unsupported operator")

    def _angle(self, q: pint.Quantity, fname: str) -> float:
        if q.unitless:
            raise self.fail(
                f"{fname}() needs an angle with units, e.g. {fname}(30 deg) or {fname}(0.5 rad)"
            )
        if not q.dimensionless:
            raise self.fail(f"{fname}() needs an angle, got {q.units:~P}")
        return float(q.to("rad").magnitude)

    def _number(self, q: pint.Quantity, fname: str) -> float:
        if not q.dimensionless:
            raise self.fail(f"{fname}() needs a plain number, got {q.units:~P}")
        return float(q.to("dimensionless").magnitude)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise self.fail("only sin, cos, tan, asin, acos, atan, atan2, sqrt, abs, hypot")
        if node.keywords:
            raise self.fail("keyword arguments are not supported")
        fname = node.func.id
        args = [self.visit(a) for a in node.args]
        arity = 2 if _FUNCTIONS[fname][0] == "pair" else 1
        if len(args) != arity:
            raise self.fail(f"{fname}() takes {arity} argument(s)")
        mode, fn = _FUNCTIONS[fname]
        a = args[0]
        try:
            if mode == "angle":
                return Q_(fn(self._angle(a, fname)))
            if mode == "number":
                return Q_(fn(self._number(a, fname)), "rad")
        except InputError:
            raise
        except (ValueError, OverflowError):
            raise self.fail(f"{fname}() of {a:~P} is undefined") from None
        if fname == "sqrt":
            if a.magnitude < 0:
                raise self.fail(f"sqrt() of a negative value ({a:~P})")
            return a**0.5
        if fname == "abs":
            return abs(a)
        b = args[1]
        try:
            b = b.to(a.units)
        except pint.DimensionalityError:
            raise self.fail(f"{fname}() arguments must have the same dimension") from None
        if fname == "atan2":
            return Q_(math.atan2(a.magnitude, b.magnitude), "rad")
        return Q_(math.hypot(a.magnitude, b.magnitude), a.units)


def as_quantity(value: Any) -> pint.Quantity:
    if isinstance(value, pint.Quantity):
        return value
    if isinstance(value, bool):
        raise InputError(f"expected a number, got {value!r}")
    if isinstance(value, int | float | np.integer | np.floating):
        return Q_(float(value))
    raise InputError(f"expected a number, got {value!r}")


# --------------------------------------------------------------------------- parsing


@dataclass(frozen=True)
class Context:
    """What bare numbers and names mean while parsing one model."""

    units: UnitSystem = field(default_factory=UnitSystem)
    parameters: Mapping[str, pint.Quantity] = field(default_factory=dict)

    def quantity(self, value: Any, kind_name: str | None = None) -> pint.Quantity:
        if isinstance(value, str):
            return evaluate(value, self.parameters, units=self.units, kind=kind_name)
        return as_quantity(value)

    def scalar(self, value: Any, kind_name: str) -> float:
        """Parse one value of the given kind and return it in SI."""
        return self._scalar(self.quantity(value, kind_name), kind(kind_name), value)

    def _scalar(self, q: pint.Quantity, k: Kind, original: Any) -> float:
        if q.unitless:
            if k.name == "dimensionless":
                return float(q.to("dimensionless").magnitude)
            q = Q_(q.to("dimensionless").magnitude, self.units.unit(k.name))
        return _to_si(q, k, original)

    def vector(self, value: Any, kind_name: str, size: int = 3) -> np.ndarray:
        """Parse a vector: a list of values, or a string like '[0, -10] kN'."""
        items, unit = split_vector(value)
        if len(items) != size:
            raise InputError(f"expected {size} components, got {len(items)} in {value!r}")
        if unit is None:
            quantities = [self.quantity(v, kind_name) for v in items]
            bare = [q.unitless and q.magnitude != 0 for q in quantities]
            explicit = [not q.unitless for q in quantities]
            if any(bare) and any(explicit):
                raise InputError(
                    f"{value!r} mixes bare numbers with explicit units; "
                    "give every non-zero component a unit, or put one unit after the brackets"
                )
            k = kind(kind_name)
            return np.array([self._scalar(q, k, v) for q, v in zip(quantities, items, strict=True)])
        unit_q = evaluate(unit, self.parameters, units_only=True)
        k = kind(kind_name)
        out = []
        for v in items:
            q = self.quantity(v)
            if not q.unitless:
                raise InputError(
                    f"{value!r}: give units once after the brackets or on each component, not both"
                )
            out.append(_to_si(q.to("dimensionless").magnitude * unit_q, k, value))
        return np.array(out)


def _to_si(q: pint.Quantity, k: Kind, original: Any) -> float:
    shown = original if isinstance(original, str | int | float) else f"{q:~P}"
    if k.name == "angle":
        if not q.dimensionless:
            raise InputError(f"{shown!r} is not an angle (it has units of {q.units:~P})")
        return float(q.to("rad").magnitude)
    if k.name in ("angular_velocity", "angular_acceleration") and any(
        "hertz" in str(name) for name in q.units._units
    ):
        raise InputError(
            f"{shown!r}: Hz is ambiguous for an {k.name.replace('_', ' ')}; "
            "write rev/s, rpm or rad/s"
        )
    try:
        return float(q.to(k.si).magnitude)
    except pint.DimensionalityError:
        noun = k.name.replace("_", " ")
        article = "an" if noun[0] in "aeiou" else "a"
        raise InputError(
            f"{shown!r} is not {article} {noun}: it has units of {q.units:~P}, "
            f"but {article} {noun} is measured in units like {ureg.Unit(k.si):~P}"
        ) from None


_VECTOR_RE = re.compile(r"^\s*[\[(](?P<items>.*)[\])]\s*(?P<unit>.*?)\s*$", re.S)


def split_vector(value: Any) -> tuple[list[Any], str | None]:
    """Split vector input into components and an optional shared unit."""
    if isinstance(value, pint.Quantity):
        mags = np.atleast_1d(value.magnitude)
        return [Q_(float(m), value.units) for m in mags], None
    if isinstance(value, str):
        m = _VECTOR_RE.match(value)
        if m and value.strip().startswith("(") and len(_split_top_level(m.group("items"))) < 2:
            m = None  # '(P + Q)*a' is a parenthesised scalar, not a vector
        if m and _unbalanced(m.group("items")):
            m = None
        if m:
            items_text, unit = m.group("items"), (m.group("unit") or None)
        elif "," in value:
            items_text, unit = value, None
        else:
            raise InputError(f"expected a vector like [1, 2, 3] m, got {value!r}")
        items = [s.strip() for s in _split_top_level(items_text)]
        if any(not s for s in items):
            raise InputError(f"empty component in {value!r}")
        return items, unit
    if isinstance(value, np.ndarray):
        return list(value.astype(float).ravel()), None
    if isinstance(value, Sequence):
        return list(value), None
    raise InputError(f"expected a vector, got {value!r}")


def _unbalanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += ch in "(["
        depth -= ch in ")]"
        if depth < 0:
            return True
    return depth != 0


def _split_top_level(text: str) -> list[str]:
    parts, depth, current = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def evaluate_parameters(
    raw: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    units: UnitSystem | None = None,
) -> dict[str, pint.Quantity]:
    """Evaluate parameters in order; later ones may reference earlier ones."""
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(raw)
    if unknown:
        raise InputError(
            f"cannot set {', '.join(sorted(unknown))}: not defined under 'parameters'"
            + (f" (defined: {', '.join(raw)})" if raw else "")
        )
    values: dict[str, pint.Quantity] = {}
    for name, value in raw.items():
        if not name.isidentifier() or keyword.iskeyword(name):
            raise InputError(f"parameter name {name!r} must be a valid identifier")
        value = overrides.get(name, value)
        try:
            values[name] = (
                evaluate(value, values, units=units)
                if isinstance(value, str)
                else as_quantity(value)
            )
        except InputError as exc:
            raise InputError(f"parameters.{name}: {exc}") from None
    return values
