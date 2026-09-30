# engmech

[![PyPI](https://img.shields.io/pypi/v/engmech)](https://pypi.org/project/engmech/)
[![Python](https://img.shields.io/pypi/pyversions/engmech)](https://pypi.org/project/engmech/)
[![Tests](https://github.com/davidson-engineering/engmech/actions/workflows/python-app.yml/badge.svg)](https://github.com/davidson-engineering/engmech/actions/workflows/python-app.yml)

Rigid-body statics, dynamics and mass properties for engineers.

Describe a problem in a short YAML file with real units, solve it from the
command line, and get support reactions, joint forces, member forces and
actuator torques, each one verified against equilibrium, plus an
interactive HTML report with free-body diagrams.

![An engmech report: title, summary checks, reaction and joint-force tables, and a free-body diagram of a truss with members coloured by tension and compression](https://raw.githubusercontent.com/davidson-engineering/engmech/main/docs/images/report.png)

## Quick start

```bash
pip install engmech
engmech examples copy truss        # start from a bundled example
engmech report truss.yaml --open   # the report above
```

A model is a short YAML file:

```yaml
# beam.yaml
name: Simply supported beam
analysis: planar
units: {length: m, force: kN}

supports:
  A: {type: pin, at: [0, 0]}
  B: {type: roller, at: [6, 0], normal: +y}

loads:
  - {force: [0, -12], at: [2, 0]}
  - {distributed: {start: [3, 0], end: [6, 0], intensity: 4 kN/m, direction: -y}}
```

```console
$ engmech solve beam.yaml
╭─────────────────────────────────────────────────────────────────╮
│ Simply supported beam                                           │
│ planar · statics · 1 body · 2 supports · units: m, kN, kN⋅m, kg │
╰─────────────────────────────────────────────────────────────────╯

✓ Equilibrium verified (max relative residual 1e-16)
Support reactions  kN
Support   Type     Fx   Fy   Resultant    N
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A         pin       0   11          11
B         roller    –   13          13   13
Force and moment on the body from the ground, at the support point. A dash means
that connection cannot carry that component. N is the normal force, positive
when pushing on the body.
```

## What it does

- **Single-body and multi-body statics.** Beams, frames, machines, trusses,
  linkages and 3D structures. Every body's equilibrium is written into one
  global system and solved at once.
- **Inverse dynamics.** Give each body's angular velocity and acceleration
  (and the acceleration of its centre of gravity, or a pivot) and get the
  joint forces and motor torques that motion requires, including the
  gyroscopic terms (Newton-Euler).
- **Mass properties.** Build bodies from rods, boxes, cylinders, tubes,
  spheres, cones, point masses and CAD values, with holes as subtracted
  shapes. You get mass, centre of gravity, inertia about any point, and
  principal axes. Weights are applied automatically.
- **Units everywhere.** Write `10 kN`, `250 mm`, `45 deg`, `300 rpm` or
  `7850 kg/m^3`; bare numbers use the file's unit system. Every value is
  checked for the right dimension, and output can use any unit system (SI,
  SI-kN, SI-mm, US-in, US-ft or custom).
- **Honest diagnostics.** Before solving, the equations are scaled so the
  diagnosis does not depend on your units. If the supports cannot hold the
  loads, it tells you which free motion the loads drive (for example "beam
  rotates about the point (0, 0)"). If some reactions cannot be found from
  equilibrium alone, they are marked indeterminate instead of guessed. Give
  joints a stiffness and redundant forces are shared by least work, which
  gives the elastic method for bolt groups and similar problems.
- **Verification built in.** Each result is checked by summing every force
  and moment on each body directly, independently of the solver. Files can
  also carry `checks:` (hand calculations or design limits) that are
  reported as pass or fail, and `engmech validate` confirms that an
  installation reproduces the documented benchmark results.
- **Engineering conveniences.** Named points and parameters with
  expressions (`[L*cos(theta), L*sin(theta)]`), load cases and combinations,
  solved-for loads ("what force P holds this?"), actuated joints, cables
  and contacts that warn when they go slack or lift off, parameter sweeps,
  JSON/CSV export, and a JSON Schema for editor autocompletion.

## Reports

`engmech report model.yaml` writes one self-contained HTML file: it opens
offline, prints cleanly, and can be filed with the calculation it records.
A report leads with the results:

1. **Summary**: whether each load case is balanced and determinate, how
   many of the file's checks pass, and any warning that applies to the
   whole model, such as free motions or sensitivity to geometry.
2. **Load cases compared**, when there is more than one: every support
   reaction and joint force in every load case and combination side by
   side, with the maximum and minimum and the case each comes from.
3. **Results** for each load case and combination: support reactions,
   joint forces and solved loads, then an interactive free-body diagram.
   Planar models are drawn with engineering support symbols; spatial models
   in 3D, with z up or, for models built with y or x up (as from CAD),
   `report: {up: y}`. Two-force members are coloured by tension and
   compression, and buttons switch between the whole model and the free
   body of each part.
4. **Checks**, **notes** from the model's description (hand calculations,
   assumptions), and the **model** itself: determinacy, units, points,
   parameters and mass properties.
5. **Provenance**: the engmech, Python and library versions, the platform,
   parameter overrides, and the input file with its SHA-256, so a result
   can be traced to exactly what produced it.

Printed or saved as PDF, a report becomes a calculation document: every
diagram is drawn to the page width, every free body is shown with its own
caption, and collapsed sections (equilibrium verification, applied loads,
the input file) are printed in full.

![A 3D free-body diagram of a boom held by a ball joint and two cables](https://raw.githubusercontent.com/davidson-engineering/engmech/main/docs/images/report-3d.png)

**Company logo.** Reports have no logo unless you give one. Set it once for
every report in the user config file (`engmech config` shows where it is):

```toml
[report]
logo = "/path/to/logo.png"   # or an https:// URL; on Windows, 'C:\path\logo.png'
```

A model file's `report: {logo: ...}`, the `ENGMECH_LOGO` environment
variable, and `--logo FILE|URL` or `--no-logo` on the command line override
it, in that order from least to most specific. The image is embedded in the
report, so it stays self-contained.

## Install

```bash
pip install engmech
# or as an isolated command-line tool:
uv tool install engmech      # or: pipx install engmech
```

Requires Python 3.11 or newer, on Linux, macOS or Windows.

## Command line

```bash
engmech examples list                      # bundled, hand-verified examples
engmech examples copy frame my-frame.yaml  # start from one

engmech solve my-frame.yaml                # reactions, joint forces, checks
engmech solve my-frame.yaml -v             # plus loads, mass, verification tables
engmech solve my-frame.yaml --set "P=15 kN" --units SI-mm
engmech solve my-frame.yaml --json -       # machine-readable results

engmech report my-frame.yaml --open        # HTML report with interactive diagrams
engmech check my-frame.yaml                # is it stable? determinate? which DOF are free?
engmech mass bracket.yaml --about A        # mass, cog, inertia tensors, principal axes
engmech sweep beam.yaml --param "P=0 kN:20 kN:11" --output A.Fy,B.N --csv out.csv
engmech validate --report validation.html  # qualify this installation
engmech config                             # config file location and the report logo in use
engmech schema -o engmech.schema.json      # JSON Schema for editor autocompletion
```

`engmech solve` exits with status 0 when everything is in equilibrium and
all checks pass, 1 for input errors, and 2 when a check fails or the loads
cannot be balanced. Add `--strict` to also fail on indeterminate results.
That makes model files usable as regression tests in CI.

Input errors point at the line in the file:

```text
error: frame.yaml:10:28: supports.B.roller: unknown field 'nromal' (did you mean 'normal'?)
error: frame.yaml:7:3: points.C: ['3 N', 4] mixes bare numbers with explicit units; give every non-zero component a unit, or put one unit after the brackets
```

## Python

The same model can be built in Python. Values accept the same forms as the
file: numbers in the model's units, strings with units, or named points.

```python
import engmech as em

m = em.Model("Three-hinged frame", planar=True, units={"length": "m", "force": "kN"})
m.point("A", [0, 0])
m.point("B", [6, 0])
m.point("C", [3, 4])
m.body("left")
m.body("right")
m.support("A", em.Pin(at="A"), body="left")
m.support("B", em.Pin(at="B"), body="right")
m.joint("C", em.Pin(at="C"), bodies=("left", "right"))
m.load(em.Force([0, -12], at=[1.5, 2]), body="left")

result = m.solve()
result.show()                      # terminal tables
result.primary["B"].force          # numpy array in SI (N)
result.primary["C"].component("Fx")
result.report("frame.html")        # HTML report
model = em.load("frame.yaml")      # or load a file
```

## How to read the results

- **Support reactions** are the force and moment *on the body from the
  ground*, at the support point, in global axes.
- **Joint forces** are reported on the "On" body from the "From" body. For
  `bodies: [a, b]` that is the force on `b` from `a`; `a` receives the
  opposite.
- **N** is a normal (roller/contact) force, positive when pushing on the
  body. **T** is the axial force in a link or cable, positive in tension.
  **Drive** is the torque or force an actuated joint must supply.
- A dash (–) means that connection cannot carry that component; `indet.`
  means equilibrium alone cannot determine it.

## Examples

Every example states its hand calculation in its description, and its
`checks:` hold the hand-derived answers. The test suite runs them all.

| Example | Shows |
|---|---|
| `beam` | point, triangular and uniform loads, a couple |
| `cantilever` | 3D fixed support, self-weight from rod shapes |
| `frame` | three-hinged frame, joint forces, free-body views |
| `truss` | method of joints with particles and links |
| `boom` | 3D boom on a ball joint and two cables |
| `shaft` | shaft on bearings with a solved-for gear force |
| `slider-crank` | mechanism held by an actuated crank |
| `robot-arm` | holding torques of a two-link arm |
| `bolt-group` | eccentric bolt group by the elastic method (stiffness) |
| `load-combinations` | dead/live cases, ULS/SLS combinations, capacity checks |
| `motor-arm` | motor torque for an accelerating arm (dynamics) |
| `gyroscope` | gyroscopic precession of a spinning disc (dynamics) |

## Verification

engmech is verified against hand calculations and against independent,
established software. The
[verification and validation document](https://github.com/davidson-engineering/engmech/blob/main/docs/validation.md)
has the full evidence, the assumptions and limitations, and a procedure for
using engmech inside a quality system. In brief:

- **Hand calculations:** 22 benchmark models reproduce 122 hand-derived
  values. Ten of them were written and solved by a reviewer who never saw
  the solver code.
- **Independent solvers:**
  - statics agrees with the PyNite finite-element solver to 10⁻¹² on 200
    random frames and trusses;
  - inverse dynamics agrees with MuJoCo to 10⁻¹⁴ on 120 random 3D
    mechanisms;
  - mass properties agree with trimesh mesh integration to 10⁻¹³, with
    curved shapes converging at the expected rate.
- **Randomised tests:** property-based tests check physical invariants
  (units, rigid motions, the elastic method, the parallel-axis theorem),
  and input fuzzing checks that malformed and degenerate models fail
  cleanly.
- **Your own installation:** `engmech validate --report validation.html`
  re-runs the benchmark suite on your machine and writes a pass/fail
  report. Every release also carries the validation reports of its wheel on
  Linux, macOS and Windows.

## Documentation

- [Input file reference](https://github.com/davidson-engineering/engmech/blob/main/docs/input-format.md):
  every section, joint type, load type and unit rule.
- [Theory manual](https://github.com/davidson-engineering/engmech/blob/main/docs/theory.md):
  equations, conventions, algorithms and numerical tolerances.
- [Verification and validation](https://github.com/davidson-engineering/engmech/blob/main/docs/validation.md):
  evidence, limitations and quality-system use.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run engmech schema -o schema/engmech.schema.json   # after changing the file format
uv run --with pillow python scripts/make_screenshots.py  # after changing the report's look
```

[Releasing](https://github.com/davidson-engineering/engmech/blob/main/docs/releasing.md)
describes how versions are published to PyPI and GitHub.

## License

MIT
