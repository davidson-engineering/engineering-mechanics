"""Report logo: where it comes from, how it is embedded, and its errors."""

import base64
import io

import pytest
from click.testing import CliRunner

from engmech.cli import main
from engmech.errors import InputError
from engmech.io.loader import load_model
from engmech.io.loader import loads_model as load_model_text
from engmech.report import branding
from engmech.report.branding import UNSET, resolve_logo
from engmech.report.html import render_report

PNG = base64.b64decode(  # a 1x1 transparent PNG
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)
MODEL = """\
analysis: planar
supports:
  A: {type: pin, at: [0, 0]}
  B: {type: roller, at: [2, 0], normal: +y}
loads:
  - {force: [0, -1], at: [1, 0]}
"""


def uri(data: bytes, mime="image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


@pytest.fixture
def logo_file(tmp_path):
    path = tmp_path / "brand" / "acme.png"
    path.parent.mkdir()
    path.write_bytes(PNG)
    return path


def model_file(tmp_path, report=""):
    path = tmp_path / "m.yaml"
    path.write_text(MODEL + report, encoding="utf-8")
    return path


def test_no_logo_by_default():
    assert resolve_logo() is None
    html = render_report(load_model_text(MODEL).solve(), plotly_cdn=True)
    assert 'class="logo"' not in html


def test_precedence(tmp_path, logo_file, monkeypatch, isolated_user_settings):
    other = tmp_path / "other.png"
    other.write_bytes(PNG + b"\0")
    isolated_user_settings.write_text(f'[report]\nlogo = "{other}"\n', encoding="utf-8")
    assert resolve_logo().source == str(other)  # a config file sets a default
    monkeypatch.setenv("ENGMECH_LOGO", str(logo_file))
    assert resolve_logo().source == str(logo_file)  # environment beats config
    assert resolve_logo(UNSET, "none") is None  # model file beats environment
    assert resolve_logo(str(other), "none").source == str(other)  # argument beats all
    assert resolve_logo(None, str(logo_file)) is None  # explicit "no logo"


def test_model_file_logo_is_relative_to_the_file(tmp_path, logo_file):
    path = model_file(tmp_path, "report: {logo: brand/acme.png}\n")
    html = render_report(load_model(str(path)).solve(), plotly_cdn=True)
    assert uri(PNG) in html


def test_model_file_can_turn_the_logo_off(tmp_path):
    for value in ("none", "false"):
        path = model_file(tmp_path, f"report: {{logo: {value}}}\n")
        html = render_report(load_model(str(path)).solve(), plotly_cdn=True)
        assert 'class="logo"' not in html


def test_logo_from_a_url(monkeypatch):
    class Response(io.BytesIO):
        headers = type("H", (), {"get_content_type": staticmethod(lambda: "image/png")})()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(branding.urllib.request, "urlopen", lambda url, timeout: Response(PNG))
    logo = resolve_logo("https://example.com/logo.png")
    assert logo.data_uri == uri(PNG)


def test_url_errors(monkeypatch):
    def fail(url, timeout):
        raise branding.urllib.error.URLError("offline")

    monkeypatch.setattr(branding.urllib.request, "urlopen", fail)
    with pytest.raises(InputError, match="could not download"):
        resolve_logo("https://example.com/logo.png")


def test_file_errors(tmp_path, monkeypatch):
    with pytest.raises(InputError, match="not found"):
        resolve_logo(str(tmp_path / "missing.png"))
    (tmp_path / "logo.bmp").write_bytes(b"BM")
    with pytest.raises(InputError, match="unsupported logo type"):
        resolve_logo(str(tmp_path / "logo.bmp"))
    big = tmp_path / "big.png"
    big.write_bytes(b"0" * (branding.MAX_BYTES + 1))
    with pytest.raises(InputError, match="under 2 MB"):
        resolve_logo(str(big))


def test_invalid_config_file(isolated_user_settings):
    isolated_user_settings.write_text("[report\nlogo = 1", encoding="utf-8")
    with pytest.raises(InputError, match="invalid config file"):
        resolve_logo()


def test_cli_options(tmp_path, logo_file):
    path = model_file(tmp_path)
    out = tmp_path / "r.html"
    runner = CliRunner()
    assert (
        runner.invoke(
            main, ["report", str(path), "-o", str(out), "--cdn", "--logo", str(logo_file)]
        ).exit_code
        == 0
    )
    assert uri(PNG) in out.read_text(encoding="utf-8")
    assert (
        runner.invoke(main, ["report", str(path), "-o", str(out), "--cdn", "--no-logo"]).exit_code
        == 0
    )
    assert 'class="logo"' not in out.read_text(encoding="utf-8")
    bad = runner.invoke(main, ["report", str(path), "-o", str(out), "--logo", "nope.png"])
    assert bad.exit_code == 1
    assert "logo file not found" in bad.output
    report = tmp_path / "v.html"
    assert (
        runner.invoke(
            main, ["validate", "--cases", "2", "--report", str(report), "--logo", str(logo_file)]
        ).exit_code
        == 0
    )
    assert uri(PNG) in report.read_text(encoding="utf-8")


def test_config_command(isolated_user_settings):
    result = CliRunner().invoke(main, ["config"])
    assert str(isolated_user_settings) in result.output
    assert "Report logo: none (set one with [report] logo in the config file)" in result.output
