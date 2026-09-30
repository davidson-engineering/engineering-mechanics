"""Rich terminal output."""

from __future__ import annotations

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table as RichTable
from rich.text import Text

from engmech.report import tables as t
from engmech.results import CaseResult, Results
from engmech.units import UnitSystem

STYLES = {
    "": "",
    "muted": "dim",
    "warn": "yellow",
    "bad": "bold red",
    "good": "green",
    "strong": "bold",
}


def render_table(table: t.Table) -> Group:
    """The table, with its caption printed at full width underneath."""
    parts = [_rich_table(table)]
    if table.caption:
        parts.append(Text(table.caption, style="dim"))
    return Group(*parts)


def _rich_table(table: t.Table) -> RichTable:
    title = Text(table.title, style="bold cyan")
    if table.units:
        title.append(f"  {table.units}", style="dim")
    rt = RichTable(
        title=title,
        title_justify="left",
        title_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        pad_edge=False,
        show_edge=False,
    )
    for i, h in enumerate(table.headers):
        cells = [row[i] for row in table.rows]
        numeric = table.numeric(i)
        if numeric or i == 0:
            # never cut a number or a name: shrink the other columns instead
            width = max([len(c.text) for c in cells] + [min(len(h), 10)])
            rt.add_column(
                h,
                justify="right" if numeric else "left",
                no_wrap=True,
                min_width=width,
                overflow="fold",
            )
        else:
            rt.add_column(h, justify="left", overflow="fold")
    for row in table.rows:
        rt.add_row(*[Text(c.text, style=STYLES[c.style]) for c in row])
    return rt


def _header(results: Results, units: UnitSystem) -> Panel:
    m = results.model
    n_supports = sum(1 for j in m.joints.values() if j.kind == "support")
    n_joints = sum(1 for j in m.joints.values() if j.kind == "joint")
    dyn = any(b.motion is not None for b in m.bodies.values())
    facts = [
        "planar" if m.planar else "spatial",
        "dynamics" if dyn else "statics",
        f"{len(m.bodies)} bod{'y' if len(m.bodies) == 1 else 'ies'}",
        f"{n_supports} support{'s' * (n_supports != 1)}",
    ]
    if n_joints:
        facts.append(f"{n_joints} joint{'s' * (n_joints != 1)}")
    unit_text = ", ".join(
        f"{units.label(k)}" for k in ("length", "force", "moment", "mass") if units.label(k)
    )
    body = Text()
    body.append(m.name, style="bold")
    body.append("\n" + " · ".join(facts) + f" · units: {unit_text}", style="dim")
    if m.description:
        body.append("\n" + m.description.strip().split("\n\n")[0])
    return Panel(body, box=box.ROUNDED, expand=False)


def _case_block(results: Results, case: CaseResult, units: UnitSystem, verbose: bool) -> Group:
    parts = []
    title = Text()
    if len(results.cases) > 1 or case.kind == "combination":
        label = "Combination" if case.kind == "combination" else "Load case"
        title.append(f"{label}: {case.name}", style="bold magenta")
        if case.kind == "combination":
            factors = " + ".join(f"{f:g}×{c}" for c, f in case.factors.items())
            title.append(f"   {factors}", style="dim")
        parts.append(title)
    style, text = t.status_summary(results, case)
    mark = {"good": "✓", "warn": "!", "bad": "✗"}[style]
    parts.append(Text(f"{mark} {text}", style=STYLES[style] or "green"))
    for w in case.warnings:
        parts.append(Text(f"  • {w}", style="yellow" if case.status != "unbalanced" else "red"))
    for mode in case.excited_modes:
        for motion in mode:
            parts.append(
                Text(
                    f"    free motion: {motion.describe(units, results.model.planar)}", style="red"
                )
            )
    for mode in case.redundancy:
        items = ", ".join(f"{j}.{c} {w:+.3g}" for j, c, w in mode)
        parts.append(Text(f"    self-stress state: {items}", style="yellow"))
    for kind in ("support", "joint"):
        table = t.joint_table(results, case, units, kind)
        if table:
            parts.append(render_table(table))
    unknown = t.unknown_table(results, case, units)
    if unknown:
        parts.append(render_table(unknown))
    if verbose:
        parts.append(render_table(t.loads_table(results, case, units)))
        parts.append(render_table(t.resultant_table(results, case, units)))
        parts.append(render_table(t.balance_table(results, case, units)))
    return Group(*parts)


def print_results(
    results: Results,
    console: Console | None = None,
    units=None,
    verbose: bool = False,
    cases: list[str] | None = None,
) -> None:
    console = console or Console()
    units = results.model.output_units if units is None else UnitSystem.from_spec(units)
    console.print(_header(results, units))
    if verbose:
        rows = t.determinacy_rows(results)
        grid = RichTable.grid(padding=(0, 2))
        grid.add_column(style="dim")
        grid.add_column()
        for k, v in rows:
            grid.add_row(k, v)
        console.print(Padding(grid, (0, 0, 1, 0)))
        mass = t.mass_table(results.model, units)
        if mass:
            console.print(render_table(mass))
    if results.sensitivity:
        console.print(Text(f"! {results.sensitivity}", style="yellow"))
    for note in results.notes + t.dropped_notes(results):
        console.print(Text(f"note: {note}", style="dim"))
    for mode in results.mechanism if results.analysis.degrees_of_freedom <= 3 else []:
        if results.status == "unbalanced":
            break
        for motion in mode:
            console.print(
                Text(f"  free motion: {motion.describe(units, results.model.planar)}", style="dim")
            )
    selected = [c for name, c in results.cases.items() if not cases or name in cases]
    for case in selected:
        console.print()
        console.print(_case_block(results, case, units, verbose))
    checks = t.check_table(results, units)
    if checks:
        console.print()
        console.print(render_table(checks))
