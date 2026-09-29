"""Keep the user's own engmech settings out of the test suite."""

import pytest


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path_factory, monkeypatch):
    """No user config file and no ENGMECH_LOGO, whatever the developer has set."""
    config = tmp_path_factory.mktemp("engmech-config") / "config.toml"
    monkeypatch.setenv("ENGMECH_CONFIG", str(config))
    monkeypatch.delenv("ENGMECH_LOGO", raising=False)
    return config
