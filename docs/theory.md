# Theory manual

This document states the mechanics and numerics engmech implements, so that
an engineer can check the method independently of the code. Section numbers
are referenced by `docs/validation.md`.

## 1. Scope and idealisations

engmech computes the forces and moments that connections (supports and
joints) must transmit to hold rigid bodies in static or dynamic
equilibrium, and the mass properties of those bodies.

- **Rigid bodies.** Bodies do not deform. Results are support reactions,
  joint forces, member axial forces and actuator efforts, not stresses,
  strains or deflections.
- **Small displacements.** Geometry is analysed in the configuration given.
  Equilibrium is not re-evaluated in a deflected shape (no P-delta, no
  buckling).
- **Linear, bilateral connections.** Every connection component can carry
  force in both directions. Contacts and cables are solved the same way and
  then *checked*: a pulling contact or a compressed cable is reported, and
  the analysis is not iterated to release it. Friction is not modelled.
- **Instantaneous dynamics.** Inverse dynamics at one instant: the motion
  is given and the forces are found. There is no time integration.
- **Planar analysis** restricts equilibrium to the xy-plane (Fx, Fy, Mz)
  and assumes out-of-plane effects are carried elsewhere.

## 2. Conventions

- Global right-handed axes x, y, z. All values are stored in SI (m, N, kg,
  s, rad); units exist only at input and output (section 10).
- A **wrench** is a force **F** with a moment **M** about a stated point
  **p**. Its moment about another point **o** is **M** + (**p** − **o**) × **F**.
- A **couple** is a free vector: it contributes its moment and no force,
  regardless of where it is drawn.
- A joint connects body *a* to body *b* (for a support, *a* is the ground).
  Its result is the wrench **on *b* from *a***, about the joint point, in
  global axes. Body *a* receives the opposite wrench.
- Link and cable force **T** is positive in tension. Roller and contact
  force **N** is positive when pushing on the body along its `normal`.

## 3. Mass properties

Each shape yields mass *m*, centre of gravity **c** and inertia tensor
**I**<sub>c</sub> about **c** in global axes. The tensor is the true tensor:
I<sub>xx</sub> = ∫(y² + z²) dm and I<sub>xy</sub> = −∫xy dm. A shape
defined in local axes with rotation **R** (columns are the local axes in
global coordinates) has **I** = **R I**<sub>local</sub> **R**ᵀ.

| Shape | Local inertia about the cog (local z = axis) |
|---|---|
| slender rod, length L | m L²/12 (**E** − **uu**ᵀ), **u** along the rod |
| box a × b × c | m/12 · diag(b² + c², a² + c², a² + b²) |
| cylinder / tube, radii r<sub>o</sub>, r<sub>i</sub>, length h | I<sub>xx</sub> = I<sub>yy</sub> = m(3(r<sub>o</sub>² + r<sub>i</sub>²) + h²)/12, I<sub>zz</sub> = m(r<sub>o</sub>² + r<sub>i</sub>²)/2 |
| sphere / shell | (2/5) m (r<sub>o</sub>⁵ − r<sub>i</sub>⁵)/(r<sub>o</sub>³ − r<sub>i</sub>³) **E** |
| solid cone, radius r, height h | cog h/4 above the base; I<sub>zz</sub> = 3mr²/10, I<sub>xx</sub> = I<sub>yy</sub> = 3mr²/20 + 3mh²/80 |

A composite of parts *k* has m = Σ m<sub>k</sub>, **c** = Σ m<sub>k</sub>**c**<sub>k</sub> / m,
and **I**<sub>c</sub> = Σ [**I**<sub>k</sub> + m<sub>k</sub>(|**r**<sub>k</sub>|²**E** − **r**<sub>k</sub>**r**<sub>k</sub>ᵀ)]
with **r**<sub>k</sub> = **c**<sub>k</sub> − **c** (parallel-axis theorem). A hole is a
part with negative mass. The inertia about any point **q** follows from the same
theorem. Principal moments and axes are the eigenvalues and eigenvectors of
**I**<sub>c</sub>. Physical consistency is enforced: m > 0, **I**<sub>c</sub>
positive semi-definite, and I₁ + I₂ ≥ I₃ for the principal moments.

## 4. Loads

- **Point force** **F** at **p**: the wrench (**F**, **0**) at **p**.
- **Couple** **M**: the wrench (**0**, **M**).
- **Distributed line load** from **a** to **b** (length L, unit vector
  **u**), intensity varying linearly from w₁ to w₂ along direction **d**:
  resultant **F** = **d** L(w₁ + w₂)/2 and moment about **a**
  **M**<sub>a</sub> = (**u** × **d**) L²(w₁ + 2w₂)/6. That is ∫ s w(s) ds, so a
  load whose resultant is zero still carries its couple. With `projected`,
  the intensity is per unit length normal to **d**, multiplied by |**u** × **d**|.
- **Gravity** **g** on a body of mass m: (m**g**, **0**) at its cog.
- **Inertia (d'Alembert).** A body with prescribed angular velocity **ω**,
  angular acceleration **α** and cog acceleration **a** receives
  (−m**a**, −(**I**<sub>c</sub>**α** + **ω** × **I**<sub>c</sub>**ω**)) at its cog.
  With it, equilibrium is the Newton-Euler equations. If a pivot **p** with
  known acceleration **a**<sub>p</sub> is given instead of **a**, then
  **a** = **a**<sub>p</sub> + **α** × **r** + **ω** × (**ω** × **r**) with **r** = **c** − **p**.

## 5. Connections

Every connection type transmits a set of wrench components along the axes
of a local frame **R** at the joint point. Each component *k* is a unit
wrench **s**<sub>k</sub>: (**e**, **0**) for a force along axis **e**, or
(**0**, **e**) for a moment about **e**. Its magnitude λ<sub>k</sub> is an
unknown. The joint's wrench on body *b* is Σ λ<sub>k</sub> **s**<sub>k</sub>.

| Type | Transmitted components (local frame) |
|---|---|
| fixed | Fx Fy Fz Mx My Mz |
| pin (axis = local z) | Fx Fy Fz Mx My (+ Mz drive if actuated) |
| ball | Fx Fy Fz |
| bearing (axis z) | Fx Fy (+ Fz with thrust) |
| slider (axis z) | Fx Fy Mx My Mz (+ Fz drive if actuated) |
| cylindrical (axis z) | Fx Fy Mx My |
| universal (axes a₁, a₂) | Fx Fy Fz, M about a₁ × a₂ |
| roller / contact | F along the normal |
| link / cable | F along the line between its ends |
| custom | the listed components |
| unknown load | F along its direction, or M about its axis |

The local frame for an axis **z** is deterministic: if **z** is along global
z, then local x is global x. Otherwise local x = normalise(**ẑ** × **z**).
This keeps components either in the plane or normal to it, as planar
analysis needs.

In a **planar** analysis only in-plane forces and moments about z enter the
equations. A component is used if it is an in-plane force or a z-moment,
and dropped if it is an out-of-plane force or an in-plane moment. A
component that mixes the two is an input error. Moments are not
transmitted to **particles**.

## 6. Equilibrium equations

For every body the sum of wrenches about a common reference point **o**
vanishes. The rows are Fx, Fy, Fz, Mx, My, Mz per rigid body; Fx, Fy, Mz in
planar analysis; forces only for particles. The column for component *k*
of a joint holds +**s**<sub>k</sub> transported to **o** in the rows of body *b*,
and −**s**<sub>k</sub> (at the joint's point on *a*) in the rows of body *a*.
Ground rows are omitted. With all applied and inertial loads collected
into **b**, the system is

**A λ** = −**b**, with **A** of size m × n (equations × unknowns).

**o** is the centroid of all model points, and the characteristic length
L is the largest distance from **o** to a model point (L = 1 m if every
point coincides within 10⁻⁹ m).

## 7. Scaling and structural analysis

Before any rank decision the system is made dimensionless. Moment rows are
divided by L, and moment unknowns are multiplied by L:
**Â** = **D**<sub>r</sub> **A D**<sub>c</sub>. The answer to "is this stable?" or "is it
determinate?" therefore cannot depend on the length unit. A singular value
decomposition **Â** = **U Σ V**ᵀ gives the rank r, counting singular values
above 10⁻¹⁰ σ<sub>max</sub>.

- **Mechanisms** (r < m): the columns of **U** beyond r span the free
  motions. For each body the corresponding rows are a twist, the velocity
  **v** of **o** and the angular velocity **ω** (moment rows divided by L).
  If |**ω**| is negligible the body translates along **v**. Otherwise it
  rotates about the axis along **ω** through **o** + (**ω** × **v**)/|**ω**|², with
  pitch (**ω**·**v**)/|**ω**|². Reports name a model point on that axis when
  one exists.
- **Redundancy** (r < n): the columns of **V** beyond r are self-stress
  states. A component is statically determinate exactly when it takes no
  part in any of them (its row of the null-space basis is below 10⁻⁸).
  This is evaluated for each global component of each joint's wrench.
- **Conditioning.** κ = σ<sub>max</sub>/σ<sub>r</sub>. Well-braced models have κ < 10.
  Above κ = 10³, results carry a warning that the supports barely resist a
  motion, which is identified from the r-th left singular vector. The
  reactions are then much larger than the loads and very sensitive to
  geometry and support flexibility.

## 8. Solution

The particular solution is the pseudo-inverse solution in scaled
coordinates, μ = **V**<sub>r</sub> **Σ**<sub>r</sub>⁻¹ **U**<sub>r</sub>ᵀ **b̂**, with **λ** = **D**<sub>c</sub> μ.

**Out of balance.** The loads can be balanced only if they have no
component along a free motion. The unbalanced part is
**e** = **U**₀**U**₀ᵀ**b̂**, where **U**₀ is the mechanism basis. It is judged
body by body. A body is out of balance if its share of **e** exceeds
10⁻⁹ of the loads on that body, ignoring shares below 10⁻¹² of all loads
(round-off). The per-body test keeps a small load that drives a lightly
loaded sub-mechanism from being hidden by large loads elsewhere. Such cases
are reported as not in equilibrium, with the free motion the loads drive.
The forces shown are then a least-squares fit.

**Stiffness (least work).** If any joint component has a compliance
c<sub>k</sub> = 1/k<sub>k</sub>, redundant forces are chosen to minimise the
complementary energy ½ Σ c<sub>k</sub> λ<sub>k</sub>² over all equilibrium solutions
**λ** = **λ**<sub>p</sub> + **N z**. The optimal **z** solves (**N**ᵀ**CN**) **z** = −**N**ᵀ**C λ**<sub>p</sub>.
With rigid bodies this is exact for elastic supports. For equal-stiffness
fasteners under a rigid plate it is the elastic method, with bolt force
P/n + M r / Σr² perpendicular to r. Components in any self-stress state
the stiffnesses do not resolve (**N**ᵀ**CN** singular) remain indeterminate.

## 9. Verification of every solution

After solving, every body's equilibrium is re-checked independently of
**A**. Each applied load and each joint wrench (applied at its own point on
that body) is summed directly about the body's cog, or the centroid of its
joints for a massless body. The relative residual is |residual| /
max(forces on the body, 10⁻⁶ × forces on the most heavily loaded body),
counting only the equation rows analysed. It must be below 10⁻⁶. A case
that fails is reported as not verified, and the CLI exits with status 2.

For bodies with prescribed motion connected by joints that hold a point in
every direction (fixed, pin, ball, universal, bearing with thrust, and
custom joints holding Fx, Fy and Fz), the
acceleration of the shared point computed from both bodies must agree.
For pins and welds, the relative angular velocity and acceleration must
also be about the free axis. Mismatches are reported as notes.

## 10. Units and expressions

Input is converted to SI with pint. Every value is checked for dimension
against the quantity it is used for.

- **Bare numbers** use the file's unit system.
- **Unit names:** a name directly after a number, or chained to it with
  `*`, `/` or `**` without spaces, is a unit. Elsewhere a parameter of the
  same name takes precedence.
- **Quantities are atomic:** a number and its unit chain are grouped, so
  implicit multiplication binds tighter than division.
- **Mixed sums:** a bare number added to a quantity takes the unit-system
  unit of the quantity being parsed; an angle takes the angle unit.
- **Trigonometric functions** require angles with units.
- **Hz** is rejected for angular quantities, because it is ambiguous with
  rad/s.
- **Mixed vectors:** a vector that mixes bare numbers with explicit units
  is rejected.

## 11. Numerical tolerances

| Name | Value | Meaning |
|---|---|---|
| RANK_TOL | 10⁻¹⁰ | singular values below this × σ<sub>max</sub> are zero |
| NULL_TOL | 10⁻⁸ | participation in a null-space vector that makes a component indeterminate |
| BALANCE_TOL | 10⁻⁹ | unbalanced share of a body's loads tolerated |
| NOISE_FLOOR | 10⁻¹² | unbalanced share of all loads treated as round-off |
| COND_WARN | 10³ | condition number above which results are flagged as sensitive |
| EQUILIBRIUM_TOL | 10⁻⁶ | largest relative residual accepted by the verification check |
| BALANCE_FLOOR | 10⁻⁶ | smallest body load scale, as a fraction of the largest, in the check |
| MIN_LENGTH | 10⁻⁹ m | a model smaller than this is treated as a point |

## 12. Accuracy

All arithmetic is IEEE double precision. The solution is backward stable.
Forces are accurate to roughly 10⁻¹⁵ κ relative to the largest forces in
the model, where κ is the condition number above. A force many orders of
magnitude smaller than the largest has correspondingly fewer correct
digits. For example, a 10⁻⁹ N joint force in a model with 7 N loads is
correct to about 6 parts per million. The validation suite's random cases
agree with independent computations to about 10⁻¹⁴ relative.

Solving uses a dense SVD, whose cost grows with the cube of the number of
equations. 400 bodies (2,400 equations) solve in about 1.5 s on a laptop.
Models with more than about 1,000 bodies are outside the intended use.
