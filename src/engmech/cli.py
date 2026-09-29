"""Command-line interface: ``engmech solve bracket.yaml`` and friends."""

from __future__ import annotations

import csv
import json
import sys
import webbrowser
from html import escape
from importlib import resources
from pathlib import Path

import click
import numpy as np
import pint
from rich import box
from rich.console import Console
from rich.table import Table as RichTable
from rich.text import Text

from engmech import __version__
from engmech.errors import InputError
from engmech.io.loader import ModelFileError, load_model
from engmech.model import Model
from engmech.units import UnitSystem, evaluate, format_number

EXIT_OK, EXIT_INPUT, EXIT_RESULT = 0, 1, 2

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- helpers


def _fail(message: str) -> None:
    err_console.print(Text("error: ", style="bold red") + Text(message))
    sys.exit(EXIT_INPUT)


def _load(path: str) -> Model:
    try:
        return load_model(path)
    except ModelFileError as exc:
        for problem in exc.problems:
            err_console.print(Text("error: ", style="bold red") + Text(problem))
        sys.exit(EXIT_INPUT)
    except InputError as exc:
        _fail(str(exc))


def _overrides(values: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for item in values:
        name, sep, value = item.partition("=")
        if not sep or not name.strip() or not value.strip():
            _fail(f"--set expects NAME=VALUE, got {item!r}")
        out[name.strip()] = value.strip()
    return out


def _report_input_error(exc: InputError) -> None:
    for problem in getattr(exc, "problems", None) or [str(exc)]:
        err_console.print(Text("error: ", style="bold red") + Text(problem))
    sys.exit(EXIT_INPUT)


def _solve(model: Model, overrides=None):
    try:
        return model.solve(overrides)
    except InputError as exc:
        _report_input_error(exc)


def _units(spec: str | None):
    if spec is None:
        return None
    try:
        return UnitSystem.from_spec(spec)
    except InputError as exc:
        _fail(str(exc))


def _exit_code(results, strict: bool) -> int:
    if results.status == "unbalanced" or not all(c.passed for c in results.checks):
        return EXIT_RESULT
    if not all(c.verified for c in results.cases.values()):
        return EXIT_RESULT
    if strict and results.status != "ok":
        return EXIT_RESULT
    return EXIT_OK


def _open(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())


class _Group(click.Group):
    def list_commands(self, ctx):
        return ["solve", "report", "check", "mass", "sweep", "examples", "schema"]


# --------------------------------------------------------------------------- commands

UNITS_HELP = "Display units: SI, SI-kN, SI-mm, US-in, US-ft (default: the file's output_units)."
SET_HELP = "Override a parameter, e.g. --set 'P=12 kN'. Repeatable."


@click.group(cls=_Group, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="engmech")
def main():
    """Rigid-body statics, dynamics and mass properties from YAML model files.

    Start from an example:  engmech examples copy beam my-beam.yaml
    """


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--set", "sets", multiple=True, metavar="NAME=VALUE", help=SET_HELP)
@click.option("--units", "units", metavar="SYSTEM", help=UNITS_HELP)
@click.option("--case", "cases", multiple=True, help="Only show this load case/combination.")
@click.option("-v", "--verbose", is_flag=True, help="Also show loads, mass properties and checks.")
@click.option("--json", "json_out", metavar="PATH", help="Write results as JSON ('-' for stdout).")
@click.option("--report", "report", type=click.Path(dir_okay=False), help="Write an HTML report.")
@click.option("--open", "open_report", is_flag=True, help="Open the HTML report when done.")
@click.option("--strict", is_flag=True, help="Exit with status 2 if anything is indeterminate.")
def solve(file, sets, units, cases, verbose, json_out, report, open_report, strict):
    """Solve a model and print support reactions and joint forces."""
    model = _load(file)
    results = _solve(model, _overrides(sets))
    unit_system = _units(units)
    unknown = [c for c in cases if c not in results.cases]
    if unknown:
        _fail(f"unknown case {unknown[0]!r} (cases: {', '.join(results.cases)})")
    if json_out == "-":
        click.echo(json.dumps(results.to_dict(unit_system), indent=2))
    else:
        from engmech.report.terminal import print_results

        print_results(
            results, console=console, units=unit_system, verbose=verbose, cases=list(cases) or None
        )
        if json_out:
            Path(json_out).write_text(json.dumps(results.to_dict(unit_system), indent=2))
            console.print(f"[dim]wrote {json_out}[/]")
    if report or open_report:
        path = Path(report or Path(file).with_suffix(".html"))
        results.report(path, units=unit_system, source_path=file)
        if json_out != "-":
            console.print(f"[dim]wrote {path}[/]")
        if open_report:
            _open(path)
    sys.exit(_exit_code(results, strict))


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Default: FILE.html")
@click.option("--set", "sets", multiple=True, metavar="NAME=VALUE", help=SET_HELP)
@click.option("--units", "units", metavar="SYSTEM", help=UNITS_HELP)
@click.option("--open", "open_report", is_flag=True, help="Open the report in a browser.")
@click.option("--cdn", is_flag=True, help="Load plotly from a CDN instead of embedding it.")
def report(file, output, sets, units, open_report, cdn):
    """Write a self-contained HTML report with an interactive 3D/2D diagram."""
    model = _load(file)
    results = _solve(model, _overrides(sets))
    path = Path(output or Path(file).with_suffix(".html"))
    results.report(path, units=_units(units), source_path=file, plotly_cdn=cdn)
    console.print(f"wrote {path}")
    if open_report:
        _open(path)
    sys.exit(_exit_code(results, strict=False))


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--set", "sets", multiple=True, metavar="NAME=VALUE", help=SET_HELP)
def check(file, sets):
    """Validate a model and describe how it is supported, without printing results."""
    from engmech.report import tables as t

    model = _load(file)
    results = _solve(model, _overrides(sets))
    units = results.model.output_units
    console.print(Text(f"{file}: valid", style="green"))
    grid = RichTable.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    for k, v in t.determinacy_rows(results):
        grid.add_row(k, v)
    console.print(grid)
    for mode in results.mechanism[:6]:
        for motion in mode:
            console.print(f"  free motion: {motion.describe(units, results.model.planar)}")
    for note in results.notes:
        console.print(Text(f"note: {note}", style="dim"))
    for case in results.cases.values():
        for w in case.warnings:
            console.print(Text(f"{case.name}: {w}", style="yellow"))
    sys.exit(_exit_code(results, strict=False))


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--set", "sets", multiple=True, metavar="NAME=VALUE", help=SET_HELP)
@click.option("--units", "units", metavar="SYSTEM", help=UNITS_HELP)
@click.option("--about", metavar="POINT", help="Also give inertia tensors about this point.")
def mass(file, sets, units, about):
    """Mass, centre of gravity and inertia of every body."""
    from engmech.inputs import Resolver
    from engmech.report import tables as t
    from engmech.report.terminal import render_table
    from engmech.units import Context

    model = _load(file)
    try:
        built = model.build(_overrides(sets))
    except InputError as exc:
        _report_input_error(exc)
    unit_system = _units(units) or built.output_units
    bodies = {n: b for n, b in built.bodies.items() if b.mass is not None}
    if not bodies:
        _fail("no body in this model has mass properties")
    point = None
    if about:
        r = Resolver(Context(built.units, built.parameters), built.points, built.planar)
        try:
            point = r.position(about)
        except InputError as exc:
            _fail(f"--about: {exc}")

    console.print(render_table(t.mass_table(built, unit_system)))
    f = unit_system.factor("inertia")
    for name, body in bodies.items():
        p = body.mass
        grid = RichTable(
            title=f"{name}",
            title_justify="left",
            title_style="bold cyan",
            show_header=False,
            box=None,
            pad_edge=False,
        )
        grid.add_column(style="dim")
        grid.add_column()
        _, axes = p.principal()
        grid.add_row(f"Inertia about cog ({unit_system.label('inertia')})", _matrix(p.inertia * f))
        if point is not None:
            grid.add_row(f"Inertia about {about}", _matrix(p.inertia_about(point) * f))
        grid.add_row("Principal axes (rows)", _matrix(axes.T, digits=4, fixed=True))
        radii = ", ".join(unit_system.format(x, "length") for x in p.radii_of_gyration())
        grid.add_row("Radii of gyration", radii)
        console.print(grid)
        console.print()


def _matrix(M, digits=4, fixed=False) -> str:
    rows = []
    for row in np.asarray(M):
        cells = [f"{x:+.{digits}f}" if fixed else format_number(x, digits) for x in row]
        rows.append("  ".join(c.rjust(12) for c in cells))
    return "\n".join(rows)


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--param",
    "param",
    required=True,
    metavar="NAME=START:STOP:COUNT",
    help="Parameter to vary, e.g. 'theta=0 deg:90 deg:19' or 'P=1 kN,2 kN,5 kN'.",
)
@click.option(
    "--output",
    "outputs",
    metavar="A.Fy,B.N",
    help="Comma-separated results to tabulate (default: all).",
)
@click.option("--case", "case", help="Load case or combination (default: the first).")
@click.option("--set", "sets", multiple=True, metavar="NAME=VALUE", help=SET_HELP)
@click.option("--units", "units", metavar="SYSTEM", help=UNITS_HELP)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False), help="Write a CSV file.")
@click.option("--plot", "plot_path", type=click.Path(dir_okay=False), help="Write an HTML chart.")
def sweep(file, param, outputs, case, sets, units, csv_path, plot_path):
    """Solve for a range of values of one parameter."""
    from engmech.report.export import flat_values

    model = _load(file)
    name, sep, spec = param.partition("=")
    name = name.strip()
    if not sep or name not in model.parameters:
        _fail(f"--param must name a parameter from the file ({', '.join(model.parameters)})")
    values = _sweep_values(spec)
    unit_system = _units(units)
    base = _overrides(sets)
    rows = []
    for q in values:
        results = _solve(model, {**base, name: q})
        c = case or next(iter(results.cases))
        if c not in results.cases:
            _fail(f"unknown case {c!r}")
        flat = flat_values(results, c, unit_system)
        rows.append((q, flat, results.status))
    keys = list(rows[0][1])
    if outputs:
        wanted = [o.strip() for o in outputs.split(",") if o.strip()]
        missing = [w for w in wanted if w not in keys]
        if missing:
            _fail(f"unknown output {missing[0]!r} (available: {', '.join(keys)})")
        keys = wanted
    else:
        keys = [k for k in keys if any(r[1][k] not in (None, 0.0) for r in rows)]
    u = unit_system or results.model.output_units
    param_unit = f"{values[0].units:~P}" if str(values[0].units) != "dimensionless" else ""
    headers = [f"{name}" + (f" ({param_unit})" if param_unit else "")]
    for k in keys:
        kind = _output_kind(results, k)
        headers.append(f"{k} ({u.label(kind)})")
    table = RichTable(
        title=f"Sweep of {name}",
        title_justify="left",
        title_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        show_edge=False,
    )
    for i, h in enumerate(headers):
        table.add_column(h, justify="right" if i else "left")
    for q, flat, status in rows:
        cells = [format_number(q.magnitude)]
        for k in keys:
            v = flat[k]
            cells.append("indet." if v is None else format_number(v))
        style = "" if status == "ok" else ("yellow" if status == "indeterminate" else "red")
        table.add_row(*cells, style=style)
    console.print(table)
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for q, flat, _ in rows:
                writer.writerow([q.magnitude] + [flat[k] for k in keys])
        console.print(f"[dim]wrote {csv_path}[/]")
    if plot_path:
        from engmech.report.figure import sweep_figure

        kinds = {_output_kind(results, k) for k in keys}
        y_label = u.label(kinds.pop()) if len(kinds) == 1 else ""
        title = f"{results.model.name}: sweep of {name}"
        fig = sweep_figure(
            [q.magnitude for q, _, _ in rows],
            headers[0],
            [[r[1][k] for r in rows] for k in keys],
            headers[1:],
            title=title,
            y_label=y_label,
        )
        html = fig.to_html(include_plotlyjs=True, full_html=True)
        html = html.replace("<head>", f"<head><title>{escape(title)}</title>", 1)
        Path(plot_path).write_text(html, encoding="utf-8")
        console.print(f"[dim]wrote {plot_path}[/]")


def _output_kind(results, key: str) -> str:
    joint, _, comp = key.partition(".")
    if comp in ("Fx", "Fy", "Fz"):
        return "force"
    if comp in ("Mx", "My", "Mz"):
        return "moment"
    j = results.primary.joints[joint]
    return j.scalar_kinds.get(comp, "force")


def _sweep_values(spec: str):
    spec = spec.strip()
    try:
        if ":" in spec:
            parts = spec.split(":")
            if len(parts) != 3:
                raise InputError("expected START:STOP:COUNT")
            start, stop = evaluate(parts[0]), evaluate(parts[1])
            count = int(parts[2])
            if count < 2:
                raise InputError("COUNT must be at least 2")
            if start.unitless and not stop.unitless:
                start = start.magnitude * stop.units  # '0:20 kN' means 0 kN to 20 kN
            elif stop.unitless and not start.unitless:
                stop = stop.magnitude * start.units
            stop = stop.to(start.units)
            return [start + (stop - start) * i / (count - 1) for i in range(count)]
        return [evaluate(v) for v in spec.split(",") if v.strip()]
    except (InputError, ValueError, pint.DimensionalityError) as exc:
        _fail(f"--param: {exc}")


@main.group()
def examples():
    """List, show and copy the bundled example models."""


def _example_dir():
    return resources.files("engmech") / "examples"


def _example_files():
    return sorted(
        (p for p in _example_dir().iterdir() if p.name.endswith(".yaml")), key=lambda p: p.name
    )


@examples.command("list")
def examples_list():
    """List the bundled examples."""
    from ruamel.yaml import YAML

    table = RichTable(box=None, pad_edge=False, header_style="bold")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    for p in _example_files():
        data = YAML(typ="safe").load(p.read_text())
        table.add_row(p.name[:-5], str(data.get("name", "")))
    console.print(table)
    console.print("[dim]engmech examples copy NAME [DEST] to start from one[/]")


def _find_example(name: str):
    for p in _example_files():
        if p.name[:-5] == name or p.name == name:
            return p
    _fail(f"no example {name!r}; see 'engmech examples list'")


@examples.command("show")
@click.argument("name")
def examples_show(name):
    """Print an example model."""
    click.echo(_find_example(name).read_text(), nl=False)


@examples.command("copy")
@click.argument("name")
@click.argument("dest", required=False, type=click.Path(dir_okay=True))
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def examples_copy(name, dest, force):
    """Copy an example to DEST (default: NAME.yaml in the current directory)."""
    src = _find_example(name)
    target = Path(dest or src.name)
    if target.is_dir():
        target = target / src.name
    if target.exists() and not force:
        _fail(f"{target} exists (use --force to overwrite)")
    target.write_text(src.read_text())
    console.print(f"wrote {target}")


@main.command()
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Default: stdout.")
def schema(output):
    """Print the JSON Schema of the model file format (for editor autocompletion)."""
    from engmech.io.schema import json_schema

    text = json.dumps(json_schema(), indent=2)
    if output:
        Path(output).write_text(text + "\n")
        console.print(f"wrote {output}")
    else:
        click.echo(text)


if __name__ == "__main__":  # pragma: no cover
    main()
