"""Cross-check engmech's inverse dynamics against MuJoCo.

MuJoCo's recursive Newton-Euler (``mj_inverse`` + ``mj_rnePostConstraint``) is
an independent implementation of rigid-body dynamics. For reproducible random
serial chains the same physical system is built twice:

* in MuJoCo, from an MJCF string, posed at random ``qpos``/``qvel``/``qacc``;
* in engmech, from MuJoCo's resulting kinematics: each body's world angular
  velocity, angular acceleration and centre-of-mass acceleration, its mass,
  centre of mass and inertia tensor rotated into world axes, and every joint
  as an actuated pin or slider at MuJoCo's joint anchor along its world axis.

Then engmech's drive of every joint must equal MuJoCo's generalised force
``qfrc_inverse``, and engmech's joint wrench (on the child from the parent)
must equal MuJoCo's ``cfrc_int``, to round-off.

MuJoCo conventions relied on (MuJoCo 3.x, checked by the tests below):

* ``qfrc_inverse`` of a hinge is the torque about ``+xaxis`` that the parent
  applies to the child; of a slide joint, the force along ``+xaxis``.
* ``cfrc_int[b]`` is the spatial force ``[torque, force]`` the parent exerts on
  body ``b``, in world axes, with the torque about ``subtree_com[body_rootid[b]]``.
* The world body's ``cacc`` is ``-gravity``, so ``mj_objectAcceleration``
  returns the proper acceleration ``a - g`` (what an accelerometer reads); the
  true centre-of-mass acceleration is that plus gravity.
* ``fullinertia="Ixx Iyy Izz Ixy Ixz Iyz"`` takes the entries of the inertia
  tensor (products of inertia with their tensor sign), in the body frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest

import engmech as em

mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.oracle

GRAVITY = 9.81
TOL = 1e-8  # relative: both sides are exact rigid-body algorithms in double precision
FD_TOL = 1e-7  # relative: fourth-order finite differences with h = 1e-3
CASES_PER_FAMILY = 24

# family: (seed prefix, link counts, slide joints, planar)
FAMILIES = {
    "single-link": (1, (1, 1), False, False),
    "hinge-chain": (2, (2, 4), False, False),
    "slide-chain": (3, (2, 4), True, False),
    "planar-hinge-chain": (4, (1, 4), False, True),
    "planar-slide-chain": (5, (2, 4), True, True),
}
CASES = [(family, seed) for family in FAMILIES for seed in range(CASES_PER_FAMILY)]
CASE_IDS = [f"{family}-{seed}" for family, seed in CASES]


# --------------------------------------------------------------------------- random chains


@dataclass
class Link:
    joint: str  # "hinge" | "slide"
    pos: np.ndarray  # body origin in the parent body frame, at the reference pose
    quat: np.ndarray  # body orientation relative to the parent (w, x, y, z)
    joint_pos: np.ndarray  # joint anchor in the body frame
    axis: np.ndarray  # joint axis in the body frame (unit)
    mass: float
    cog: np.ndarray  # centre of mass in the body frame
    inertia: np.ndarray  # 3x3 tensor about the centre of mass, body axes


@dataclass
class Chain:
    links: list[Link]
    gravity: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    planar: bool


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def random_rotation(rng) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def random_inertia(rng, mass: float, size: float) -> np.ndarray:
    """A full, positive-definite inertia tensor that meets the triangle inequality.

    With second moments s = (sum x^2 dm, sum y^2 dm, sum z^2 dm) the principal
    moments are (sy + sz, sx + sz, sx + sy), so I1 + I2 - I3 = 2 s_k > 0.
    """
    s = mass * size**2 * rng.uniform(0.05, 1.0, 3)
    R = random_rotation(rng)
    return R @ np.diag([s[1] + s[2], s[0] + s[2], s[0] + s[1]]) @ R.T


def quat_to_matrix(quat) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


def random_chain(family: str, seed: int) -> Chain:
    prefix, (n_min, n_max), with_slides, planar = FAMILIES[family]
    rng = np.random.default_rng([prefix, seed])
    n = int(rng.integers(n_min, n_max + 1))
    kinds = ["hinge"] * n
    if with_slides:  # one or more slides, and at least one hinge
        for i in rng.choice(n, size=int(rng.integers(1, n)), replace=False):
            kinds[i] = "slide"

    links = []
    frame = np.eye(3)  # world orientation of the parent at the reference pose
    previous_axis = None  # world axis of the parent's joint at the reference pose
    for kind in kinds:
        if planar:
            angle = rng.uniform(-np.pi, np.pi)
            quat = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
            pos = np.append(rng.uniform(-0.6, 0.6, 2), 0.0)
            joint_pos = np.append(rng.uniform(-0.2, 0.2, 2), 0.0)
            cog = np.append(rng.uniform(-0.4, 0.4, 2), 0.0)
            if kind == "hinge":  # about +z or -z, to exercise the sign convention
                axis = np.array([0.0, 0.0, rng.choice([-1.0, 1.0])])
            else:
                phi = rng.uniform(-np.pi, np.pi)
                axis = np.array([np.cos(phi), np.sin(phi), 0.0])
        else:
            quat = unit(rng.normal(size=4))
            pos = rng.uniform(-0.6, 0.6, 3)
            joint_pos = rng.uniform(-0.2, 0.2, 3)
            cog = rng.uniform(-0.4, 0.4, 3)
        frame = frame @ quat_to_matrix(quat)
        if not planar:
            while True:  # consecutive joint axes well away from parallel
                axis = unit(rng.normal(size=3))
                world_axis = frame @ axis
                if previous_axis is None or abs(world_axis @ previous_axis) < 0.9:
                    break
        previous_axis = frame @ axis
        mass = float(rng.uniform(0.5, 5.0))
        links.append(
            Link(kind, pos, quat, joint_pos, axis, mass, cog, random_inertia(rng, mass, 0.3))
        )

    if planar:
        phi = rng.uniform(-np.pi, np.pi)
        gravity = GRAVITY * np.array([np.cos(phi), np.sin(phi), 0.0])
    else:
        gravity = GRAVITY * unit(rng.normal(size=3))
    hinge = np.array([k == "hinge" for k in kinds])
    q = np.where(hinge, rng.uniform(-np.pi, np.pi, n), rng.uniform(-0.3, 0.3, n))
    qd = rng.normal(0.0, 2.0, n)
    qdd = rng.normal(0.0, 5.0, n)
    return Chain(links, gravity, q, qd, qdd, planar)


# --------------------------------------------------------------------------- MuJoCo side


def _v(values) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.atleast_1d(values))


def to_mjcf(chain: Chain) -> str:
    """Nested bodies b1..bn, each on one joint j1..jn to its parent."""
    body = ""
    for k in range(len(chain.links), 0, -1):
        link = chain.links[k - 1]
        I = link.inertia
        full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        body = f"""
    <body name="b{k}" pos="{_v(link.pos)}" quat="{_v(link.quat)}">
      <joint name="j{k}" type="{link.joint}" pos="{_v(link.joint_pos)}" axis="{_v(link.axis)}"
             armature="0" damping="0" frictionloss="0" stiffness="0"/>
      <inertial pos="{_v(link.cog)}" mass="{link.mass:.17g}" fullinertia="{_v(full)}"/>{body}
    </body>"""
    return f"""
<mujoco model="chain">
  <compiler angle="radian"/>
  <option gravity="{_v(chain.gravity)}">
    <flag contact="disable"/>
  </option>
  <worldbody>{body}
  </worldbody>
</mujoco>"""


@dataclass
class MujocoSolution:
    model: object  # mujoco.MjModel
    data: object  # mujoco.MjData


@cache
def mujoco_solution(family: str, seed: int) -> MujocoSolution:
    chain = random_chain(family, seed)
    model = mujoco.MjModel.from_xml_string(to_mjcf(chain))
    data = mujoco.MjData(model)
    data.qpos[:] = chain.q
    data.qvel[:] = chain.qd
    data.qacc[:] = chain.qdd
    mujoco.mj_inverse(model, data)
    mujoco.mj_rnePostConstraint(model, data)  # fills cacc and cfrc_int
    return MujocoSolution(model, data)


def body_motion(model, data, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World angular velocity, angular acceleration and centre-of-mass acceleration."""
    vel, acc = np.zeros(6), np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, b, acc, 0)
    # mjOBJ_BODY is centred on the body's centre of mass; its linear acceleration
    # is a - g (the world body accelerates at -gravity in MuJoCo's recursion)
    return vel[:3], acc[:3], acc[3:] + model.opt.gravity


def world_inertia(model, data, b: int) -> np.ndarray:
    """MuJoCo's inertia tensor of body b about its centre of mass, in world axes."""
    R = data.ximat[b].reshape(3, 3)
    I = R @ np.diag(model.body_inertia[b]) @ R.T
    return (I + I.T) / 2


def mujoco_joint_wrench(model, data, j: int, point) -> tuple[np.ndarray, np.ndarray]:
    """Force and moment about ``point`` that joint j's parent exerts on its child."""
    b = model.jnt_bodyid[j]
    torque, force = data.cfrc_int[b, :3], data.cfrc_int[b, 3:]
    c = data.subtree_com[model.body_rootid[b]]  # cfrc_int's reference point
    return force.copy(), torque + np.cross(c - np.asarray(point), force)


# --------------------------------------------------------------------------- engmech side


def engmech_model(chain: Chain, model, data) -> em.Model:
    """The same system in engmech, posed and moving as MuJoCo computed it."""
    planar = chain.planar

    def vec(v):  # planar engmech models take in-plane vectors
        if planar:
            assert abs(v[2]) <= 1e-12 * max(1.0, np.abs(v).max()), "left the xy-plane"
            return [float(v[0]), float(v[1])]
        return [float(x) for x in v]

    m = em.Model(planar=planar, gravity=vec(model.opt.gravity))
    for k, link in enumerate(chain.links, start=1):
        b, j = model.body(f"b{k}").id, model.joint(f"j{k}").id
        w, al, a = body_motion(model, data, b)
        if planar:  # planar engmech takes angular motion as a scalar about z
            for v in (w, al):
                assert np.abs(v[:2]).max() <= 1e-12 * max(1.0, abs(v[2])), "not about z"
            w, al = float(w[2]), float(al[2])
        else:
            w, al = vec(w), vec(al)
        m.body(
            f"b{k}",
            mass=float(model.body_mass[b]),
            cog=vec(data.xipos[b]),
            inertia=world_inertia(model, data, b).tolist(),
            motion=em.Motion(angular_velocity=w, angular_acceleration=al, acceleration=vec(a)),
        )
        kind = em.Pin if link.joint == "hinge" else em.Slider
        joint = kind(at=vec(data.xanchor[j]), axis=[float(x) for x in data.xaxis[j]], actuated=True)
        if k == 1:
            m.support(f"j{k}", joint, body=f"b{k}")
        else:
            m.joint(f"j{k}", joint, bodies=(f"b{k - 1}", f"b{k}"))
    return m


@cache
def engmech_result(family: str, seed: int):
    sol = mujoco_solution(family, seed)
    return engmech_model(random_chain(family, seed), sol.model, sol.data).solve()


# --------------------------------------------------------------------------- comparisons


def scales(model, data) -> tuple[float, float]:
    """Characteristic force and moment: the largest joint wrench in MuJoCo."""
    wrenches = [mujoco_joint_wrench(model, data, j, data.xanchor[j]) for j in range(model.njnt)]
    return max(np.linalg.norm(f) for f, _ in wrenches), max(np.linalg.norm(m) for _, m in wrenches)


def relative_error(actual, expected, scale: float) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))) / scale


@pytest.mark.parametrize(("family", "seed"), CASES, ids=CASE_IDS)
def test_mujoco_model_matches_the_specification(family, seed):
    """MJCF round trip: mass, centre of mass, inertia (incl. the sign of the
    products of inertia), joint anchors and axes are what was specified."""
    chain = random_chain(family, seed)
    model = mujoco_solution(family, seed).model
    data = mujoco.MjData(model)  # reference pose, qpos = 0
    mujoco.mj_kinematics(model, data)
    frame = np.eye(3)
    origin = np.zeros(3)
    for k, link in enumerate(chain.links, start=1):
        b, j = model.body(f"b{k}").id, model.joint(f"j{k}").id
        origin = origin + frame @ link.pos
        frame = frame @ quat_to_matrix(link.quat)
        assert model.jnt_type[j] == (
            mujoco.mjtJoint.mjJNT_HINGE if link.joint == "hinge" else mujoco.mjtJoint.mjJNT_SLIDE
        )
        assert model.body_mass[b] == pytest.approx(link.mass, rel=1e-15)
        np.testing.assert_allclose(data.xipos[b], origin + frame @ link.cog, atol=1e-14)
        np.testing.assert_allclose(data.xanchor[j], origin + frame @ link.joint_pos, atol=1e-14)
        np.testing.assert_allclose(data.xaxis[j], frame @ link.axis, atol=1e-14)
        scale = np.abs(link.inertia).max()
        np.testing.assert_allclose(
            world_inertia(model, data, b), frame @ link.inertia @ frame.T, atol=1e-12 * scale
        )


@pytest.mark.parametrize(("family", "seed"), CASES, ids=CASE_IDS)
def test_mujoco_kinematics_match_finite_differences(family, seed):
    """The motion handed to engmech, checked against positions alone.

    Along q(t) = q + qd t + qdd t^2 / 2, mj_kinematics gives each body's centre
    of mass x(t) and orientation R(t); fourth-order central differences then
    give v, a, and omega, alpha from R' = [omega] R and R'' = ([alpha] + [omega]^2) R.
    """
    chain = random_chain(family, seed)
    sol = mujoco_solution(family, seed)
    model = sol.model
    data = mujoco.MjData(model)
    h = 1e-3
    xs, Rs = [], []
    for t in (-2 * h, -h, 0.0, h, 2 * h):
        data.qpos[:] = chain.q + chain.qd * t + chain.qdd * t**2 / 2
        mujoco.mj_kinematics(model, data)
        xs.append(data.xipos.copy())
        Rs.append(data.ximat.reshape(-1, 3, 3).copy())
    xs, Rs = np.array(xs), np.array(Rs)

    def d1(f):
        return (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h)

    def d2(f):
        return (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h**2)

    def vee(S):
        return np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2

    v_fd, a_fd, Rd, Rdd = d1(xs), d2(xs), d1(Rs), d2(Rs)
    got, expected = [], []
    for b in range(1, model.nbody):
        R = Rs[2][b]
        W = Rd[b] @ R.T
        w, al, a = body_motion(model, sol.data, b)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, sol.data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        got += [w, al, vel[3:], a]
        expected += [vee(W), vee(Rdd[b] @ R.T - W @ W), v_fd[b], a_fd[b]]
    scale = max(1.0, *(np.linalg.norm(e) for e in expected))
    assert relative_error(np.concatenate(got), np.concatenate(expected), scale) < FD_TOL


@pytest.mark.parametrize(("family", "seed"), CASES, ids=CASE_IDS)
def test_drives_match_mujoco_inverse_dynamics(family, seed):
    """engmech's actuated-joint drive == MuJoCo's qfrc_inverse, for every joint."""
    sol = mujoco_solution(family, seed)
    model, data = sol.model, sol.data
    result = engmech_result(family, seed)
    assert result.status == "ok"
    assert result.primary.verified
    force_scale, moment_scale = scales(model, data)
    for j in range(model.njnt):
        name = model.joint(j).name
        drive = result.primary[name].scalars["drive"]
        expected = data.qfrc_inverse[model.jnt_dofadr[j]]
        hinge = model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        scale = moment_scale if hinge else force_scale
        err = relative_error(drive, expected, scale)
        assert err < TOL, f"{name}: engmech {drive!r}, MuJoCo {expected!r} (rel. {err:.1e})"


@pytest.mark.parametrize(("family", "seed"), CASES, ids=CASE_IDS)
def test_joint_wrenches_match_mujoco_interaction_forces(family, seed):
    """engmech's joint force and moment (on the child from the parent, about the
    joint point) == MuJoCo's cfrc_int moved to the same point."""
    chain = random_chain(family, seed)
    sol = mujoco_solution(family, seed)
    model, data = sol.model, sol.data
    result = engmech_result(family, seed)
    force_scale, moment_scale = scales(model, data)
    for j in range(model.njnt):
        joint = result.primary[model.joint(j).name]
        force, moment = mujoco_joint_wrench(model, data, j, joint.point)
        if chain.planar:  # the in-plane equations; out-of-plane reactions are not analysed
            force, moment = force[:2], moment[2:]
            got_force, got_moment = joint.force[:2], joint.moment[2:]
        else:
            got_force, got_moment = joint.force, joint.moment
        f_err = relative_error(got_force, force, force_scale)
        m_err = relative_error(got_moment, moment, moment_scale)
        assert f_err < TOL, f"{joint.name} force: engmech {got_force}, MuJoCo {force}"
        assert m_err < TOL, f"{joint.name} moment: engmech {got_moment}, MuJoCo {moment}"


# --------------------------------------------------------------------------- discrepancies


def test_pinned_body_with_its_own_pivot_motion_is_not_flagged():
    """A pendulum given by engmech's own ``pivot`` form: the pivot is fixed by
    construction, yet engmech warns that it accelerates differently on the
    ground and on the arm (and prints both accelerations as (0, 0))."""
    m = em.Model(planar=True)
    m.body(
        "arm",
        mass=2.0,
        cog=[1.1, 0.3],
        inertia=[1, 1, 0.5],
        motion=em.Motion(
            angular_velocity="12 rad/s", angular_acceleration="100 rad/s^2", pivot=[0, 0]
        ),
    )
    m.support("O", em.Pin(at=[0, 0], actuated=True), body="arm")
    result = m.solve()
    assert result.status == "ok"
    assert result.notes == []
