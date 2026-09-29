# Input file reference

A model file is YAML. Every section is optional except that something must
hold the body up. The smallest useful file is a few lines:

```yaml
analysis: planar
supports:
  A: {type: pin, at: [0, 0]}
  B: {type: roller, at: [4, 0], normal: +y}
loads:
  - {force: [0, -10], at: [1, 0]}
```

With no `bodies` section there is one implicit body called `body`, and
supports and loads act on it.

For autocompletion and inline validation in VS Code (YAML extension) or any
editor using yaml-language-server, put this on the first line:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/davidson-engineering/engineering-mechanics/main/schema/engmech.schema.json
```

## Top-level sections

| Key | Meaning |
|---|---|
| `name`, `description` | Shown in the terminal and report. Indented lines in the description are kept as-is (good for hand calculations). |
| `analysis` | `spatial` (default, 3D) or `planar` (the xy-plane: forces Fx, Fy and moments Mz only). |
| `units` | How bare numbers are read, and the default display units. |
| `output_units` | Display units, if different from `units`. |
| `parameters` | Named values usable in any expression. |
| `gravity` | Applies weight to every body with mass. |
| `points` | Named points, usable anywhere a position is expected. |
| `bodies` | Rigid bodies (and particles), their mass, motion and loads. |
| `supports` | Connections from a body to the fixed ground. |
| `joints` | Connections between two bodies. |
| `loads` | Loads not listed under a body. |
| `combinations` | Factored combinations of load cases. |
| `checks` | Expected values and design limits to verify. |

## Units and values

Every value can be a number, a string with units, or an expression:

```text
10            # a bare number: read in the file's unit for that quantity
10 kN         # explicit unit (10kN also works)
2.5 kN*m      # also 2.5 kN m, 2.5 kN·m
45 deg        # angles: deg, rad, 45°
300 rpm
7850 kg/m^3
L/2           # parameter expression
P*cos(30 deg) # functions: sin cos tan asin acos atan atan2 sqrt abs hypot
```

`units` sets the unit of bare numbers. Give a preset, or base units, or
both (any other quantity can also be overridden by name):

```yaml
units: SI-kN                         # SI, SI-kN, SI-mm, US-in, US-ft
units: {length: mm, force: N}        # others default to SI (kg, s, deg)
units: {system: SI-kN, moment: kN*m, inertia: kg*mm^2}
```

Presets:

| Preset | length | force | mass | angle |
|---|---|---|---|---|
| `SI` (default) | m | N | kg | deg |
| `SI-kN` | m | kN | kg | deg |
| `SI-mm` | mm | N | kg | deg |
| `US-in` | in | lbf | lb | deg |
| `US-ft` | ft | lbf | lb | deg |

Derived units follow from these (moment = force × length, inertia = mass ×
length², and so on). If you give an imperial force unit without a mass unit,
mass defaults to lb. Bare angles are in degrees unless you change `angle`.

Every value is checked against the quantity it is used for. A length given
as `10 N` is an error, not a silent mistake.

### Rules for units in expressions

- A name directly after a number is a unit: `2 m` is two metres even if
  there is a parameter called `m`. Units chained to it with `*`, `/` or `**`
  without spaces are units too (`9.81 m/s^2`, `5 kN*m`).
- Anywhere else, a parameter wins over a unit of the same name: `2*m`,
  `m * g` and `L/2` use the parameters `m` and `L`.
- A bare number added to a quantity with units takes the file's unit for
  that quantity: with lengths in mm, `r + 3` is `r + 3 mm`.
- Trigonometric functions need angles with units (`sin(30 deg)`), so
  degrees and radians are never confused.

### Vectors, positions and directions

```yaml
at: [2, 0]                  # planar: [x, y]; spatial: [x, y, z]
at: A                       # a named point
at: "[100, 50, 0] mm"       # one unit for all components (quote it in YAML)
at: [100 mm, 50 mm, 0]      # or a unit on each non-zero component
at: [L*cos(theta), L*sin(theta)]
direction: -y               # +x -x +y -y +z -z
direction: [1, 1, 0]        # any vector; only its direction matters
direction: 30 deg           # planar: angle from +x towards +y
```

Mixing bare numbers and units in one vector (`[3, -10 kN]`) is rejected
because it is ambiguous. Zero is always allowed.

## parameters

```yaml
parameters:
  L: 2 m
  P: 12 kN
  theta: 30 deg
  M: P*L/4        # parameters can use earlier ones
```

Override from the command line with `--set "P=20 kN"` (repeatable), or sweep
one with `engmech sweep FILE --param "P=0 kN:20 kN:11"`.

## points

```yaml
points:
  A: [0, 0]
  B: [L, 0]
  C: [L/2, L*tan(theta)/2]
```

Point names must be simple identifiers. Use them wherever a position is
expected (`at: B`, `start: A`, `ends: [A, C]`).

## gravity

```yaml
gravity: -z                               # standard gravity, 9.80665 m/s²
gravity: "[0, -9.81] m/s^2"                # quote vectors that have a unit after them
gravity: {direction: -y, magnitude: 9.81 m/s^2, case: dead}
```

Weight acts at each body's centre of gravity. `case` puts the weights in a
load case (default: `default`).

## bodies

```yaml
bodies:
  beam:
    shapes:
      - {type: rod, density: 150 kg/m, start: A, end: B}
    loads:
      - {force: [0, -5 kN], at: B}
  bracket:
    mass: 12 kg
    cog: [0.1, 0.02, 0]
    inertia: [0.05, 0.08, 0.1]      # optional: diagonal, or a 3x3 tensor
  pulley: {}                        # a massless body
  node_B: {particle: true}          # takes forces only (truss joints, cable rings)
```

| Key | Meaning |
|---|---|
| `shapes` | List of shapes whose mass properties are combined (see below). |
| `mass`, `cog`, `inertia` | Mass properties given directly (for example from CAD). |
| `particle` | The body only has force equations: a joint where forces meet. |
| `motion` | Instantaneous motion, for inverse dynamics (see below). |
| `outline` | Points to draw the body through in the report (display only). |
| `loads` | Loads on this body (same form as the top-level `loads`). |

### Shapes

Each shape takes either `mass` or `density` (mass per volume; for a rod,
mass per length). `subtract: true` removes it (holes, cut-outs). Shapes are
placed in global coordinates.

| `type` | Geometry keys |
|---|---|
| `point` | `at` |
| `rod` | `start`, `end` (slender; no inertia about its own axis) |
| `box` | `size: [lx, ly, lz]`, `center`, optional `orientation` |
| `cylinder` / `disc` | `radius`, `length` (0 for a thin disc), `center`, `axis`, `inner_radius` (tube) |
| `sphere` | `radius`, `center`, `inner_radius` (shell) |
| `cone` | `radius`, `height`, `base_center`, `axis` (base towards apex) |
| `custom` | `mass`, `cog`, `inertia`, `about` (point the inertia is about; default cog), `orientation` |

An inertia tensor is the true tensor, so off-diagonal entries are the
negated products of inertia (Ixy = −∫xy dm). Mass properties are checked for
physical consistency (positive mass, triangle inequality).

`orientation` (for boxes, custom shapes, fixed and custom joints) is one of:

```yaml
orientation: {x: [1, 1, 0], z: +z}                  # any two local axes
orientation: {axis: +z, angle: 30 deg}              # rotation about an axis
orientation: {euler: [0, 30, 45], sequence: xyz}    # degrees; lowercase extrinsic, UPPER intrinsic
```

### motion (inverse dynamics)

```yaml
motion:
  angular_velocity: "[0, 0, 2] rad/s"      # planar: a single value about z
  angular_acceleration: 5 rad/s^2
  pivot: O                                 # a point whose acceleration is known...
  pivot_acceleration: [0, 0, 0]            # ...(default zero)
  # or instead of pivot: acceleration: [...] m/s^2 (of the centre of gravity)
```

The body's d'Alembert loads, −m·a at the centre of gravity and
−(I·α + ω × I·ω), are added to its equilibrium, so the same solver gives
Newton-Euler dynamics. Bodies without `motion` are at rest. The motion of
bodies connected by pins and welds is checked for consistency, and any
mismatch is reported.

## supports and joints

Supports connect a body to the ground; joints connect two bodies. Both are
maps of name to definition:

```yaml
supports:
  A: {type: pin, at: A, body: beam}          # 'body' is optional with one body
joints:
  C: {type: pin, at: C, bodies: [left, right]}
```

A joint's result is the force and moment **on the second body from the
first** (`bodies: [a, b]`: on `b` from `a`); the first receives the
opposite. Support results are on the body from the ground.

| `type` | Carries | Keys |
|---|---|---|
| `fixed` (`weld`) | all forces and moments | `at`, `orientation` |
| `pin` (`revolute`, `hinge`) | all forces; moments except about the axis | `at`, `axis` (required in 3D; z in planar), `actuated` |
| `ball` (`spherical`) | all forces, no moments | `at` |
| `bearing` | forces across the axis (plus along it with `thrust: true`), no moments | `at`, `axis`, `thrust` |
| `slider` (`prismatic`) | forces across the axis, all moments | `at`, `axis`, `actuated` |
| `cylindrical` | forces and moments across the axis | `at`, `axis` |
| `universal` | all forces, moment about the cross axis | `at`, `axes: [a1, a2]` |
| `roller` | one force N along `normal` (push or pull) | `at`, `normal` |
| `contact` | like roller, but can only push | `at`, `normal` |
| `link` (`strut`) | one axial force T (tension or compression) | support: `at` + `anchor`; joint: `ends: [on a, on b]` |
| `cable` | like link, but can only pull | as link |
| `custom` | the listed local components | `at`, `constrain: [Fx, Fz, My]`, `orientation` |

- `normal` is the direction the support pushes on the body. N is positive
  pushing; T is positive in tension.
- `actuated: true` on a pin or slider adds the drive torque or force as an
  unknown, reported as `drive`: the holding torque of a motor, or the force
  of a hydraulic cylinder.
- A cable in compression or a contact in tension is solved anyway, and
  flagged: the real support would go slack or lift off.
- `bearing` and `ball` are how textbooks model hinges and bearings that are
  "properly aligned and do not carry couples". A real `pin` carries
  moments, and several pins on one body usually make it statically
  indeterminate.

### stiffness

When a structure is statically indeterminate, equilibrium alone cannot
split the load between redundant supports. engmech says so and marks those
components `indet.`. To share them by stiffness, give the joints involved a
stiffness:

```yaml
B1: {type: pin, at: B1, stiffness: {translational: 200 kN/mm}}
B2: {type: fixed, at: B2, stiffness: {translational: "[1e6, 1e6, rigid] N/m", rotational: 5e4 N*m/rad}}
S:  {type: roller, at: S, normal: +y, stiffness: 2e5 N/m}   # one value for single-force types
```

Stiffnesses are in the joint's local axes. Redundant forces are then found
by least work (minimum complementary energy) with the bodies rigid. For
equal-stiffness fasteners under a rigid plate, this is the elastic method.
Joints without a stiffness stay rigid.

## loads

Each load is one of `force`, `moment`, `distributed` or `unknown`, with
optional `name`, `case` and `body`:

```yaml
loads:
  - {name: P, force: [0, -12 kN], at: B}
  - {force: {magnitude: 5 kN, direction: [3, -4]}, at: C}
  - {force: {magnitude: 5 kN, angle: -60 deg}, at: C}          # planar angle from +x
  - {force: {magnitude: 2 kN, toward: D}, at: C}               # from 'at' towards D
  - {force: {magnitude: 2 kN, along: [E, F]}, at: C}           # parallel to E -> F
  - {moment: 10 kN*m}                                          # planar: about z (CCW +)
  - {moment: "[0, 0, 5] kN*m", at: B}                          # a couple: 'at' only places the arrow
  - {moment: {magnitude: 5 kN*m, axis: [1, 1, 0]}}
  - distributed: {start: A, end: B, intensity: 3 kN/m, direction: -y}
  - distributed: {start: A, end: B, intensity: {start: 0, end: 6 kN/m}, direction: -y}
  - distributed: {start: A, end: C, intensity: 1 kN/m, direction: -y, projected: true}
  - {unknown: P, at: C, direction: +y}      # a force whose magnitude is solved for
  - {unknown: T, axis: +z}                  # a couple whose magnitude is solved for
```

- Distributed loads vary linearly from `start` to `end`. With
  `projected: true` the intensity is per unit of horizontal projection
  (normal to `direction`), as for snow on a slope.
- Unknown loads answer "what force holds this?". They are solved like
  reactions and reported in their own table, positive along the given
  direction.

## Load cases and combinations

Loads, gravity and motion without a `case` belong to the case `default`.
Give cases names and combine them with factors:

```yaml
loads:
  - {force: [0, -10], at: B, case: dead}
  - {force: [0, -6], at: B, case: live}
combinations:
  ULS: {dead: 1.2, live: 1.6}
  SLS: {dead: 1.0, live: 1.0}
```

Every case and combination is solved and reported. The CLI's `--case`
option shows just one.

## checks

```yaml
checks:
  - {target: A.Fy, expect: 28 kN}                    # default tolerance 0.1 %
  - {target: B.N, expect: 21 kN, tolerance: 0.05 kN}
  - {target: C.F, max: 15 kN, name: pin C capacity}  # design limit
  - {target: BD.T, min: 0, name: cable stays taut}
  - {target: P, expect: 1333.3 N}                    # a solved load, by name
  - {target: beam.mass, expect: 900 kg}
  - {target: A.Fy, case: ULS, max: 40 kN}
```

Targets are `<support or joint>.<component>`, where the component is one of
Fx Fy Fz Mx My Mz, F or M (magnitudes), or the joint's scalar (N, T,
drive); or `<body>.mass`; or the name of an unknown load. Values are shown
in the unit you wrote them in. Without `case`, `max`/`min` checks apply to
every case and combination. The CLI exits with status 2 if any check fails.

## Results and verification

- **Reactions** are the force and moment on the body from the ground, at
  the support point, in global axes. **Joint forces** are on the second body
  from the first.
- A **mechanism** (a free motion the supports do not prevent) is fine as
  long as the loads do no work on it; the free motion is still described,
  for example "rotates about an axis along +x through A". If the loads do
  drive it, the case is reported as not in equilibrium.
- **Indeterminate** components are marked `indet.`; the self-stress states
  (combinations that can be added in any amount) are listed.
- Every case is verified by summing all loads and joint forces on each body
  directly, independent of the solver. The relative residual must be below
  10⁻⁶.
