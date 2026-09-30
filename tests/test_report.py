from importlib import resources

import pytest

from engmech.io.loader import load_model
from engmech.report.html import render_report

EXAMPLES = resources.files("engmech") / "examples"


@pytest.mark.parametrize(
    "name", ["beam", "frame", "boom", "load-combinations", "gyroscope", "truss"]
)
def test_report_renders_every_section(name):
    results = load_model(str(EXAMPLES / f"{name}.yaml")).solve()
    html = render_report(results, plotly_cdn=True)
    for heading in (
        "Summary",
        "Support reactions",
        "Equilibrium verification",
        "Model",
        "Method and conventions",
    ):
        assert heading in html
    assert html.count('engmechFigure("figure-') == len(results.cases)


def test_figure_views_for_multibody():
    results = load_model(str(EXAMPLES / "frame.yaml")).solve()
    fig = results.figure()
    buttons = fig.layout.updatemenus[0].buttons
    assert [b.label for b in buttons] == ["Whole model", "Free body: left", "Free body: right"]


def test_indeterminate_report_explains_itself():
    from engmech import Force, Model, Pin

    m = Model("two pins", planar=True)
    m.support("A", Pin(at=[0, 0]))
    m.support("B", Pin(at=[4, 0]))
    m.load(Force([3, -10], at=[1, 0]))
    html = render_report(m.solve(), plotly_cdn=True)
    assert "Statically indeterminate" in html
    assert "indet." in html


def test_report_leads_with_results():
    """The header keeps only the description's opening paragraph; the rest
    (hand calculations) follows the results, and numbers precede the diagram."""
    results = load_model(str(EXAMPLES / "boom.yaml")).solve()
    html = render_report(results, plotly_cdn=True)
    header = html[: html.index('id="summary"')]
    assert "A 4 m boom" in header
    assert "Hand check" not in header
    results_at = html.index('id="case-1"')
    assert html.index("Support reactions", results_at) < html.index("Free-body diagram", results_at)
    assert html.index("Hand check") > html.index('id="checks"')
    assert html.index('id="notes"') < html.index('id="model"')


def _labels(fig, group):
    return [
        t
        for tr in fig.data
        if tr.type == "scatter3d" and tr.mode == "text" and tr.legendgroup == group and tr.visible
        for t in tr.text
    ]


def test_force_groups_have_distinct_colours():
    from engmech.report.figure import PALETTE

    groups = ["applied", "weight", "inertia", "reaction", "joint", "solved", "tension"]
    groups.append("compression")
    assert len({PALETTE[g] for g in groups}) == len(groups)


def test_member_forces_are_not_drawn_twice():
    """A cable's force is shown by colouring the cable, so the whole-model
    view has no reaction arrow lying along it."""
    fig = load_model(str(EXAMPLES / "boom.yaml")).solve().figure()
    assert _labels(fig, "reaction") == ["7.692 kN"]  # the ball joint only
    assert sorted(_labels(fig, "tension")) == ["4.142 kN T", "6.214 kN T"]


STAYED_BEAM = """
name: 3D beam with a stay
units: {length: m, force: kN}
points: {A: [0, 0, 0], B: [4, 0, 0], M: [0, 0, 3], N: [0, 0, 4]}
bodies:
  beam:
    outline: [A, B]
    loads:
      - {distributed: {start: A, end: B, intensity: 2 kN/m, direction: -z}}
  mast:
    outline: [M, N]
supports:
  A: {type: ball, body: beam, at: A}
  S: {type: link, body: beam, at: B, anchor: [4, 1, 0]}
  N: {type: fixed, body: mast, at: N}
joints:
  stay: {type: link, bodies: [mast, beam], ends: [M, B]}
"""


# the boom example turned so that y is up: (x, y, z) -> (x, z, -y)
BOOM_Y_UP = """
name: Boom held by two cables, y up
units: {length: m, force: kN}
points: {A: [0, 0, 0], B: [4, 0, 0], D: [0, 2, -3], E: [0, 3, 2]}
bodies:
  boom:
    outline: [A, B]
    loads:
      - {name: W, force: [0, -5, 0], at: B}
supports:
  A: {type: ball, at: A}
  BD: {type: cable, at: B, anchor: D}
  BE: {type: cable, at: B, anchor: E}
report: {up: y}
"""


def _results(name):
    from engmech.io.loader import loads_model

    if name == "stayed-beam":  # a distributed load and a link in tension, in 3D
        return loads_model(STAYED_BEAM).solve()
    if name == "boom-y-up":
        return loads_model(BOOM_Y_UP).solve()
    return load_model(str(EXAMPLES / f"{name}.yaml")).solve()


@pytest.mark.parametrize(
    "name", ["boom", "cantilever", "gyroscope", "shaft", "stayed-beam", "boom-y-up"]
)
def test_3d_view_is_orthographic_with_every_label_inside_the_scene(name):
    import numpy as np

    from engmech.report.figure import _label_boxes, _screen_axes

    fig = _results(name).figure()
    scene = fig.layout.scene
    assert scene.camera.projection.type == "orthographic"
    # one textposition per trace: plotly misplaces 3D text given short arrays of them
    texts = [tr for tr in fig.data if tr.type == "scatter3d" and tr.mode == "text"]
    assert texts
    assert all(isinstance(tr.textposition, str) for tr in texts)
    # plotly clips 3D text at the axis ranges, so the ranges take in the labels
    # (a reversed range draws a model axis that runs the other way)
    ranges = [scene.xaxis.range, scene.yaxis.range, scene.zaxis.range]
    sign = np.array([1.0 if r[0] < r[1] else -1.0 for r in ranges])
    lo, hi = np.array([min(r) for r in ranges]), np.array([max(r) for r in ranges])
    ratio = np.array([scene.aspectratio.x, scene.aspectratio.y, scene.aspectratio.z])
    per_px = (hi - lo).max() / ratio.max() / ((620 - 58) / 2)
    camera = scene.camera
    eye = np.array([camera.eye.x, camera.eye.y, camera.eye.z])
    up = np.array([camera.up.x, camera.up.y, camera.up.z])
    np.testing.assert_allclose(up, [0, 0, 1])  # plotly only turns a view about its z axis
    right, screen_up = _screen_axes(eye, up)
    boxes = _label_boxes(fig, right * sign, screen_up * sign, per_px)
    assert (boxes >= lo).all()
    assert (boxes <= hi).all()


def _drawn(fig):
    """What a 3D figure shows: the points of its lines, markers and labels as
    drawn (reversed axes applied), the axis titles, and each label's position.
    (Arrowhead cones are left out: their facets depend on the global axes.)"""
    import numpy as np

    scene = fig.layout.scene
    axes = (scene.xaxis, scene.yaxis, scene.zaxis)
    sign = np.array([1.0 if a.range[0] < a.range[1] else -1.0 for a in axes])
    points = [
        np.array([[np.nan if v is None else v for v in c] for c in (tr.x, tr.y, tr.z)], float).T
        * sign
        for tr in fig.data
        if tr.type == "scatter3d"
    ]
    titles = [a.title.text for a in axes]
    labels = {
        text: tr.textposition
        for tr in fig.data
        if tr.type == "scatter3d" and tr.mode == "text"
        for text in tr.text
    }
    return points, titles, labels


def test_a_y_up_model_is_drawn_like_the_same_model_with_z_up():
    """With report.up: y, a model built with y up is drawn exactly like the
    same model built with z up, with the axes named for what they show, so
    plotly's turntable rotation turns it about y."""
    import numpy as np

    points_z, titles_z, labels_z = _drawn(_results("boom").figure())
    points_y, titles_y, labels_y = _drawn(_results("boom-y-up").figure())
    assert titles_z == ["x (m)", "y (m)", "z (m)"]
    assert titles_y == ["x (m)", "z (m)", "y (m)"]
    assert len(points_y) == len(points_z)
    for a, b in zip(points_z, points_y, strict=True):
        # the scene's padding is symmetric, so only the positions matter
        np.testing.assert_allclose(b, a, atol=1e-9)
    assert labels_y == labels_z


def test_up_axis_argument_and_its_errors():
    import numpy as np

    from engmech.errors import InputError
    from engmech.report.figure import up_axis

    np.testing.assert_allclose(up_axis(), [0, 0, 1])
    np.testing.assert_allclose(up_axis("Y"), [0, 1, 0])
    np.testing.assert_allclose(up_axis("-x"), [-1, 0, 0])
    with pytest.raises(InputError, match="x, y or z"):
        up_axis("w")
    with pytest.raises(InputError, match="x, y or z"):
        up_axis([0, 1, 0])
    # the argument beats the model file, which beats the default
    results = _results("boom-y-up")
    scene = results.figure(up="-z").layout.scene
    assert scene.zaxis.title.text == "z (m)"
    assert scene.zaxis.range[0] > scene.zaxis.range[1]  # drawn upside down
    assert '"text":"x (m)"' in render_report(results, plotly_cdn=True, up="x")
    assert [t for t in _drawn(results.figure(up="x"))[1]] == ["y (m)", "z (m)", "x (m)"]
    # planar models are always drawn in the xy-plane
    beam = load_model(str(EXAMPLES / "beam.yaml")).solve()
    assert "scene" not in beam.figure(up="x").layout.to_plotly_json()


def test_3d_free_body_view_shows_the_member_force_as_an_arrow():
    """The whole model colours the stay; the beam's free body shows the
    stay's pull on it as an arrow instead."""
    fig = _results("stayed-beam").figure()
    assert "6.667 kN T" in _labels(fig, "tension")
    assert "6.667 kN" not in _labels(fig, "joint")
    beam = next(b for b in fig.layout.updatemenus[0].buttons if b.label == "Free body: beam")
    for tr, visible in zip(fig.data, beam.args[0]["visible"], strict=True):
        tr.visible = visible
    assert _labels(fig, "joint") == ["6.667 kN"]
    assert "2 kN/m" in _labels(fig, "applied")


def test_3d_members_are_named_on_hover_not_at_their_ends():
    """Supports and joints at a point are labelled with their names; a link
    or cable spans two points, so it is named in its hover text instead (at a
    truss node, the names of every member meeting there would pile up)."""
    fig = _results("stayed-beam").figure()
    assert sorted(_labels(fig, "supports")) == ["A", "N"]  # not the link S or the stay
    tension = next(
        tr
        for tr in fig.data
        if tr.type == "scatter3d" and tr.mode == "lines" and tr.legendgroup == "tension"
    )
    assert {t for t in tension.text if t} == {"<b>stay</b>: 6.667 kN T"}


def test_text_positions_point_away_from_the_line():
    from engmech.report.figure import _text_position

    assert _text_position(1, 0) == "middle right"
    assert _text_position(0, 1) == "top center"
    assert _text_position(-1, -1) == "bottom left"
    assert _text_position(-0.2, 1) == "top center"


def test_distributed_load_label_clears_a_mid_span_load():
    """Self-weight acts mid-span on a uniformly loaded beam; the line load's
    label moves along the beam instead of sitting under the weight arrow."""
    fig = load_model(str(EXAMPLES / "load-combinations.yaml")).solve().figure()
    for tr in fig.data:
        if tr.mode == "text" and tr.visible and "2 kN/m" in tr.text:
            x = tr.x[list(tr.text).index("2 kN/m")]
            assert abs(x - 3.0) > 0.5  # the weight acts at x = 3 m
            break
    else:
        pytest.fail("no distributed load label")


def test_load_cases_are_compared_side_by_side():
    from engmech.report import tables as t

    results = load_model(str(EXAMPLES / "load-combinations.yaml")).solve()
    table = t.envelope_table(results, results.model.output_units, "support")
    assert table.headers == ["Support", "", "dead", "live", "ULS", "SLS", "Max", "Min"]
    rows = {(r[0].text, r[1].text): [c.text for c in r[2:]] for r in table.rows}
    assert ("A", "Fx") not in rows  # zero in every case
    assert rows[("A", "Fy")] == ["10.41", "15.67", "37.56", "26.08", "37.56 (ULS)", "10.41 (dead)"]
    assert table.units == "kN"  # only forces are left: Mz is zero in every case
    html = render_report(results, plotly_cdn=True)
    assert "Load cases compared" in html
    assert '<a href="#case-3">ULS</a>' in html
    # a single case has nothing to compare
    boom = load_model(str(EXAMPLES / "boom.yaml")).solve()
    assert t.envelope_table(boom, boom.model.output_units, "support") is None
    assert "Load cases compared" not in render_report(boom, plotly_cdn=True)


def test_a_column_of_numbers_has_its_header_aligned_with_it():
    """The checks table's Value column sits between text columns; its header
    is right-aligned like its numbers (as in the terminal)."""
    html = render_report(load_model(str(EXAMPLES / "beam.yaml")).solve(), plotly_cdn=True)
    checks = html[html.index('id="checks"') :]
    assert '<th class="num">Value</th>' in checks
    assert '<th class="">Criterion</th>' in checks


def test_principal_axes_line_up():
    """Axis components are signed to one width (and never -0.0000), so the
    vectors printed one under another line up."""
    from engmech.report import tables as t

    model = load_model(str(EXAMPLES / "motor-arm.yaml")).build()
    axes = t.inertia_detail(model, "arm", model.output_units)["axes"]
    assert axes[0] == [" 0.8660", " 0.5000", " 0.0000"]
    assert axes[1] == ["-0.5000", " 0.8660", " 0.0000"]


def test_balance_table_shows_rounding_noise_as_zero():
    """Residuals far below the loads (here 1e-12 kN·m on the excavator's
    bucket) read as 0, like the rows that happen to cancel exactly."""
    from engmech.report import tables as t

    results = load_model(str(EXAMPLES / "excavator.yaml")).solve()
    table = t.balance_table(results, results.primary, results.model.output_units)
    assert [(r[1].text, r[2].text) for r in table.rows] == [("0 kN", "0 kN⋅m")] * 4


def test_report_figures_have_view_buttons_and_print_images():
    html = render_report(load_model(str(EXAMPLES / "frame.yaml")).solve(), plotly_cdn=True)
    assert '<div class="views" hidden></div>' in html
    assert '<div class="print-views"></div>' in html
    assert "function renderPrintViews" in html
    assert "Balance of each body" in html  # not a second "Equilibrium verification"


def test_each_planar_view_frames_what_it_shows():
    fig = load_model(str(EXAMPLES / "robot-arm.yaml")).solve().figure()
    for button in fig.layout.updatemenus[0].buttons:
        visible, ranges = button.args
        assert button.method == "update"
        xs, ys = [], []
        for tr, shown in zip(fig.data, visible["visible"], strict=True):
            if shown:
                xs += [v for v in tr.x if v is not None]
                ys += [v for v in tr.y if v is not None]
        x0, x1 = ranges["xaxis.range"]
        y0, y1 = ranges["yaxis.range"]
        assert x0 < min(xs), button.label
        assert max(xs) < x1, button.label
        assert y0 < min(ys), button.label
        assert max(ys) < y1, button.label


def test_member_labels_face_out_of_the_truss():
    """The two diagonals meeting under the load label on their outer sides,
    clear of each other and of the load's label."""
    fig = load_model(str(EXAMPLES / "truss.yaml")).solve().figure()
    tension = next(
        tr for tr in fig.data if tr.mode == "text" and tr.legendgroup == "tension" and tr.visible
    )
    sides = {
        round(x, 3): position
        for x, text, position in zip(tension.x, tension.text, tension.textposition, strict=True)
        if text == "7.211 kN T"
    }
    (left_x, left), (right_x, right) = sorted(sides.items())
    assert left_x < 4 < right_x
    assert "left" in left
    assert "right" in right


def test_force_labels_grow_away_from_their_arrows():
    """A label beyond the tail of a horizontal arrow grows away from it
    instead of being centred on it (where it would lie over the shaft)."""
    from engmech.io.loader import loads_model

    gate = loads_model(
        """
        analysis: planar
        supports:
          A: {type: pin, at: [0, 0]}
          B: {type: pin, at: [0, 0.9]}
        loads:
          - {force: [0, -600], at: [0.6, 0.45]}
        """
    ).solve()
    reaction = next(
        tr
        for tr in gate.figure().data
        if tr.mode == "text" and tr.legendgroup == "reaction" and tr.visible
    )
    # A (below) pushes right, so its label is left of the tail; B pulls left
    labels = sorted(zip(reaction.y, reaction.textposition, strict=True))
    assert labels == [(0.0, "middle left"), (0.9, "middle right")]


def test_moment_labels_grow_sideways_clear_of_a_vertical_reaction():
    fig = load_model(str(EXAMPLES / "robot-arm.yaml")).solve().figure()
    reaction = next(
        tr for tr in fig.data if tr.mode == "text" and tr.legendgroup == "reaction" and tr.visible
    )
    positions = dict(zip(reaction.text, reaction.textposition, strict=True))
    assert positions["53.94 N\u22c5m"] == "bottom left"  # the 107.9 N reaction runs straight down


@pytest.mark.parametrize("name", ["boom", "cantilever", "gyroscope", "shaft", "boom-y-up"])
def test_report_script_fits_3d_scenes_like_python(name, tmp_path):
    """The report refits 3D scenes to the size they are shown at with a copy
    of SceneFit.solve in figure.js; the two must agree."""
    import json
    import shutil
    import subprocess
    from importlib import resources as res

    import numpy as np

    from engmech.report.figure import SceneFit

    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the report's script")
    fit = _results(name).figure().layout.meta["engmech_fit"]
    script = (res.files("engmech") / "report" / "templates" / "figure.js").read_text("utf-8")
    runner = tmp_path / "fit.js"
    runner.write_text(
        "globalThis.window = {addEventListener() {}};\n"
        + script
        + f"\nconst m = {json.dumps(fit)};\n"
        + "console.log(JSON.stringify([[1.6, 562], [0.8, 330], [2.4, 700]].map("
        + "([a, h]) => window.engmechFitScene(m, a, h))));\n",
        encoding="utf-8",
    )
    out = subprocess.run(
        [node, str(runner)], capture_output=True, text=True, encoding="utf-8", check=True
    )
    labels = [(np.array(p), xs, ys) for p, xs, ys in fit["labels"]]
    lo, hi, right, up = (np.array(fit[k]) for k in ("lo", "hi", "right", "up"))
    python = SceneFit(lo, hi, labels, right, up)
    for (aspect, height), js in zip(
        [(1.6, 562), (0.8, 330), (2.4, 700)], json.loads(out.stdout), strict=True
    ):
        lo, hi, ratio = python.solve(aspect, height)
        np.testing.assert_allclose(js["lo"], lo, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(js["hi"], hi, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(js["ratio"], ratio, rtol=1e-9)


def test_coloured_members_are_not_also_drawn_as_dashed_links():
    """Two lines in the same place flicker in 3D: where a cable or link is
    coloured by its force, its plain dashed line is left out, and it comes
    back in free-body views, where members are not coloured."""

    def dashed(fig, view=None):
        visible = view.args[0]["visible"] if view else [tr.visible for tr in fig.data]
        return [
            tr
            for tr, shown in zip(fig.data, visible, strict=True)
            if shown and tr.type == "scatter3d" and tr.line and tr.line.dash == "dash"
        ]

    def ends(traces):
        return {(x, z) for tr in traces for x, z in zip(tr.x, tr.z, strict=True) if x is not None}

    assert dashed(_results("boom").figure()) == []  # both cables are coloured
    fig = _results("stayed-beam").figure()
    views = {b.label: b for b in fig.layout.updatemenus[0].buttons}
    stay = {(0.0, 3.0), (4.0, 0.0)}  # the stay (in tension) runs from M to B
    assert not stay <= ends(dashed(fig, views["Whole model"]))  # it is coloured instead
    assert (4.0, 0.0) in ends(dashed(fig, views["Whole model"]))  # link S carries nothing
    assert stay <= ends(dashed(fig, views["Free body: beam"]))
