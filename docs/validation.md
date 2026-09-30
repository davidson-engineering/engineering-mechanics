# Verification and validation

This document states what engmech is for, what has been done to show that
it computes what it claims, how to confirm that on your own installation,
and what it does not do. It follows the usual split in engineering
software quality assurance:

- **Verification** asks whether the software solves its equations
  correctly. The evidence is below.
- **Validation of a model** asks whether a rigid-body idealisation is
  adequate for a particular structure or machine. That judgement belongs
  to the engineer using the results; section 2 lists the assumptions that
  judgement must cover.

## 1. Intended use

engmech is intended for engineers computing, for rigid bodies:

- support reactions, joint forces, two-force member (link, cable) forces
  and actuator efforts in static equilibrium;
- the same quantities in dynamic equilibrium for a prescribed
  instantaneous motion (inverse dynamics);
- mass, centre of gravity and inertia properties of bodies built from
  standard shapes and CAD values;
- load cases, factored combinations, and simple capacity checks on the
  computed forces.

Results are intended as design inputs checked by a competent engineer, in
the same way as hand calculations or any other analysis program. engmech
is not certified to any sector-specific software standard (for example
ASME NQA-1, DO-178C or IEC 62304). An organisation that needs such a
qualification should treat this document and the validation suite as
supporting evidence for its own dedication process.

## 2. Assumptions and limitations

These are properties of the method, not defects. They are stated in full in
the theory manual (`docs/theory.md`, section 1).

1. Bodies are rigid. engmech gives forces, not stresses, strains or
   deflections, and no member design checks beyond user-defined force limits.
2. Small displacements: equilibrium in the given configuration only. There
   is no P-delta, buckling or large-rotation analysis.
3. Connections are linear and two-sided while solving. Contacts that would
   pull and cables that would compress are flagged, not released and
   re-solved. Friction is not modelled.
4. Statically indeterminate structures are solved only when joint
   stiffnesses are given, treating the bodies as rigid. Otherwise the
   indeterminate components are reported as such.
5. Dynamics is instantaneous and inverse: motion in, forces out. There is
   no time integration, and no forward dynamics except as a consistency
   check.
6. Planar analysis assumes out-of-plane equilibrium is satisfied elsewhere.
7. Solving uses dense linear algebra. Models of up to a few hundred bodies
   solve in seconds; models of more than about 1,000 bodies are outside the
   intended use.
8. Accuracy: forces are accurate to about 10⁻¹⁵ × κ relative to the largest
   forces in the model, where κ is the reported condition number. Results
   with κ > 10³ carry a sensitivity warning.

## 3. Verification evidence

Every item below runs automatically. Items 3.1, 3.3, 3.4 and 3.6 run on
each change, in CI on Linux, macOS and Windows with Python 3.11 to 3.14.
Item 3.2 runs in a separate CI job, because it needs the optional
`validation` dependencies. Item 3.5 happens every time a model is solved.

### 3.1 Analytical benchmarks (23 models, 129 values)

Thirteen examples (`src/engmech/examples`) and ten benchmark problems
(`src/engmech/benchmarks`). Each model's description contains its hand
derivation, and its `checks:` hold the hand-derived answers. The model
must reproduce every value and pass the equilibrium check. The tolerance
is 0.1 % for the examples and 0.001 % for the ten benchmark problems (or
a tight absolute tolerance where the expected value is zero).

The ten benchmark problems were written and solved by a reviewer who did
not see the solver code. They cover an inclined roller, a 3D sign on a
ball joint, hinge and cable, a frame with a pulley, a particle on three
cables in 3D, a boom in US units, snow and wind on a rafter (projected and
trapezoidal loads), a pushed cabinet on contacts (dynamics), an unbalanced
rotor (3D dynamics), a plate with a hole (composite mass properties) and a
hydraulically actuated boom.

The examples cover beams with point, triangular and uniform loads and
couples, a 3D cantilever with self-weight, a three-hinged frame, a Warren
truss, a boom on cables, the cylinders and track loads of an excavator, a
shaft on bearings with a solved gear force, a slider-crank, a robot arm's
holding torques, an eccentric bolt group, load combinations with capacity
checks, a motor-driven arm and a precessing gyroscope.

### 3.2 Independent solvers (oracles)

The same physical systems are built in established, independently
developed software and the results compared (`tests/oracles`).

| Oracle | What is compared | Cases | Largest difference |
|---|---|---|---|
| **PyNite** 3.2 (3D frame finite elements) | Reactions, joint wrenches and member forces of statically determinate structures, which do not depend on stiffness: 3D welded frames as one body and split into several bodies, planar frames on inclined rollers, three-hinged frames, 2D and 3D trusses | 120 random structures | 3.4 × 10⁻¹² relative |
| PyNite, elastic supports | Indeterminate structures on elastic supports (beams and plates on springs, bolt groups with anisotropic stiffness, a redundant hexapod on inclined springs) against FE models with members ρ times stiffer than the supports, ρ = 10…10⁴ | 80 random structures | Converges to engmech at order 1.00 ± 0.05 in 1/ρ, as expected when only member flexibility differs |
| **MuJoCo** 3.14 (multibody dynamics) | Drive torques and forces (`qfrc_inverse`) and every joint's reaction wrench (`cfrc_int`) for random 3D hinge chains, chains with slide joints, planar chains, and single links with full inertia tensors, random pose, velocity, acceleration and gravity | 120 random mechanisms (481 checks) | 1.7 × 10⁻¹⁴ relative |
| **trimesh** 5.1 (exact polyhedral integration) | Mass, centre of mass and full inertia tensor of boxes in random orientations (every orientation input form), through both the Python API and model files | 24 configurations | 9 × 10⁻¹⁴ relative |
| trimesh, curved shapes | Solid and hollow cylinders, cones, spheres and shells with random axes against successively finer meshes | 25 configurations | Mesh error falls 4.00× per halving of the mesh spacing, as it must; extrapolated agreement 1.3 × 10⁻⁷ or better |
| trimesh with manifold3d booleans | Composites of boxes and tubes, and plates with through holes, blind pockets and cross-bores (`subtract: true`) against true boolean geometry | 60 composites | 2 × 10⁻¹⁴ for boxes; 1.7 × 10⁻⁹ or better (extrapolated) with curved parts |
| trimesh, parallel-axis and principal axes | `inertia_about` at points up to 50 m away, principal moments and axes of asymmetric composites | 100 checks | 1.8 × 10⁻¹⁴; axes within 2.5 × 10⁻¹³ rad |
| trimesh, slender rods | engmech's slender-rod idealisation against thin cylinders | 80 checks | Difference equals the theoretical 6(r/L)²/(1 + 3(r/L)²) |

Each comparison was checked for sensitivity. Deliberate mistakes planted
in the comparison or in engmech's formulas all made cases fail: 9 for
MuJoCo, 10 for PyNite and 9 for trimesh. They included flipped axes,
dropped gravity, swapped loads, un-rotated inertia, wrong reference points,
a cylinder inertia off by 10⁻⁶ and a cone centroid at h/3. The agreement shown is therefore evidence, not an artefact of
a lenient test.

### 3.3 Physical invariants (property-based tests)

Randomised tests (hypothesis, and seeded cases in the validation suite)
check laws that any correct solver must obey, against computations that do
not use the solver:

- a fixed support cancels the resultant of any set of forces and couples;
- a body on six arbitrary links matches an independently assembled linear
  system;
- results are identical in m/kN and mm/N;
- rotating and translating a whole 3D model rotates every reaction with it;
- equal-stiffness bolt groups reproduce the elastic method for random
  layouts;
- composite inertia satisfies the parallel-axis theorem;
- a driven rod matches Newton-Euler in closed form;
- a precessing disc's joint moment equals I<sub>s</sub>·spin·precession.

Typical agreement is 10⁻¹⁵ to 10⁻¹⁴ relative, against a limit of 10⁻⁹.

### 3.4 Robustness

- **Extreme scales:** geometry from 1 µm to 100 km and loads from 1 µN to
  1 GN reproduce the closed-form answer to 10⁻¹⁵ relative.
- **Ill-conditioning:** nearly concurrent or parallel supports are solved
  exactly and flagged with the motion they barely resist.
- **Size:** 100-body models are solved exactly in the test suite; 400
  bodies take about 1.5 s.
- **Input fuzzing:** randomly generated malformed files (400 per test
  run, plus 3,000 in a dedicated run) end in clear input errors and never
  crash. Structurally valid random models, many degenerate (coincident
  points, zero-length links, near-singular supports; 500 per run, plus
  5,000 in a dedicated run), either raise a clear input error or solve.
  Every solution not reported as out of balance passes the independent
  equilibrium check.
- **Input forms:** every documented way of writing a value (vectors, unit
  strings, named points, magnitude with direction, angle, toward or along,
  each orientation form, each stiffness form, each unit-system form) is
  shown to give identical physics. Every documented input mistake is shown
  to be rejected with a located message.

### 3.5 Verification of each individual result

Separately from the test suite, every solution engmech produces is checked
at run time by summing all loads and joint forces on every body directly,
independently of the solver's matrices (theory manual, section 9). The
check must close to 10⁻⁶ relative. A failure is reported in the output,
and the CLI exits with status 2.

### 3.6 Test adequacy

- **Fault injection:** the built-in validation suite is itself tested by
  re-introducing the original couple bug and a sign error in distributed
  loads. It must report failure, and it does.
- **Coverage:** CI requires at least 96 % branch coverage of the
  calculation modules (solver, results, diagnostics, model, loads, joints,
  mass, shapes, inputs, units, spatial) and 90 % overall. The oracle tests
  run separately.
- **Regression tests:** every defect in section 5 has a test that fails on
  the defective code.

## 4. Installation qualification

Any installation can confirm that it reproduces the documented results:

```bash
engmech validate --report validation-report.html
```

This runs every benchmark model and the seeded invariant checks (about 440
comparisons, well under a second). It writes a report listing each case,
its evidence and the software environment, and exits with status 0 only if
everything passes. CI produces this report for every supported platform
and Python version and keeps it as a build artifact.

Every engmech report and JSON export also records the engmech version,
Python and library versions, platform, time, any parameter overrides, and
the input file's path and SHA-256. Together these tie a result to exactly
what produced it.

## 5. Defects found during verification

| Found by | Defect | Resolution |
|---|---|---|
| Rewrite (v0.1) | Applied couples and reaction moments picked up an extra r × M; the README's own sample output was wrong | Solver rewritten; regression test |
| Rewrite (v0.1) | Iterative multi-body solver carried the same error and converged slowly | Replaced by one global solve |
| Code review | Bare numbers added to angles were read as radians | Fixed; regression test |
| Code review | Planar sliders got a false kinematic warning | Fixed; regression test |
| Code review | Planar `{magnitude, axis}` accepted out-of-plane axes | Now rejected; regression test |
| Code review | A disc by density with no thickness had zero mass | Now rejected; regression test |
| Code review | Distributed loads on particles lost their moment | Now rejected; regression test |
| Code review | Bare numbers in sums took the wrong kind's unit where kinds share a dimension | Fixed; regression test |
| Code review | `Hz` accepted as rad/s | Now rejected; regression test |
| Code review | Stiffness vectors with arithmetic took the unit on the last term only | Fixed; regression test |
| Code review | `(P + Q)*a` parsed as a vector | Fixed; regression test |
| Code review | `sweep` and math-domain errors crashed | Now input errors; regression tests |
| Property test | A couple lost when all points coincided within 10⁻³⁸ m | Scaling floor; regression test |
| Structural fuzzing | Equilibrium check falsely failed a body with forces 10⁻¹⁰ of the model's | Check scale floored at 10⁻⁶ of the largest body; regression test |
| Structural fuzzing | A small load driving a lightly loaded sub-mechanism could be judged balanced | Imbalance judged body by body; regression test |
| MuJoCo cross-check | False motion-inconsistency note at fixed pivots (round-off) | Mismatch judged against the summed terms; regression test |
| Coverage review | `10 kN / 2 m` read as (10 kN / 2)·m | A number and its units form one quantity; regression test |
| trimesh cross-check | With lengths in mm, a bare `density: 7850` meant kg/mm³, a mass 10⁹ times too large | Densities outside 0.05 to 25 000 kg/m³ are rejected with an explanation; regression test |
| MuJoCo cross-check | Planar input rejected round-off such as z = 10⁻¹⁷ in computed geometry | Out-of-plane parts below 10⁻⁹ of the in-plane size are treated as zero; regression test |
| CI (Windows) | Files read and written without an explicit encoding; redirected output crashed | UTF-8 throughout, enforced in CI on every OS |

None of the oracle comparisons found an error in any computed reaction,
joint force, member force, actuator effort or mass property. Their findings
were in diagnostics and input handling, listed above.

## 6. Traceability

| Requirement | Evidence |
|---|---|
| Single-body statics reactions | 3.1 benchmarks; 3.2 PyNite (determinate); 3.3 resultant, six-link, rigid-motion; `tests/test_statics.py` |
| Multi-body statics (frames, machines, trusses) | 3.1; 3.2 PyNite (split frames, three-hinged frames, trusses); `tests/test_statics.py` |
| Indeterminacy detection and least-work sharing | 3.2 PyNite (elastic supports); 3.3 bolt group; `tests/test_statics.py` |
| Mechanism detection and description | `tests/test_statics.py`, `tests/test_dynamics.py`, `tests/test_robustness.py` |
| Inverse dynamics | 3.1 (motor arm, gyroscope, cabinet, rotor); 3.2 MuJoCo; 3.3 rod dynamics, gyroscope; `tests/test_dynamics.py` |
| Mass properties | 3.1 (plate with hole); 3.2 trimesh; 3.3 parallel axis; `tests/test_mass.py` |
| Units and expressions | `tests/test_units.py`, `tests/test_input_forms.py`, `tests/test_regressions.py` |
| Input validation and messages | `tests/test_loader.py`, `tests/test_input_forms.py`, `tests/test_api_paths.py`, 3.4 fuzzing |
| Result verification at run time | 3.5; `tests/test_robustness.py` |
| Reports, provenance, CLI | `tests/test_report.py`, `tests/test_cli.py`, `tests/test_validation.py` |

## 7. Using engmech in a quality system

A practical procedure:

1. Record the engmech version (`engmech --version`) and run
   `engmech validate --report` on each installation you use for project
   work. Keep the report.
2. Keep model files under version control. Each engmech report records the
   input's SHA-256, so a filed result can be matched to its input exactly.
3. Put hand-checkable expectations in `checks:`. They turn a model into a
   self-verifying calculation and fail loudly if anything changes.
4. Treat warnings as design information. An indeterminate or sensitive
   result means the rigid-body idealisation is doing real work, and the
   engineer should decide whether it is adequate.
5. For regulated work, use this document, the theory manual and the CI
   artifacts as inputs to your own software dedication.
