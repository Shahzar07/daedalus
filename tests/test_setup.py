"""The first-run setup wizard (``dae setup``).

These tests drive the wizard end-to-end by feeding scripted answers to
``console.input`` — no terminal, no network. We always pick the offline ``mock``
provider or decline the live connection test, so the suite stays hermetic ($0, no
keys). Each test asserts the wizard wrote the right ``.env`` lines, dropped the
"setup complete" marker, and returned the surface the user chose.
"""

from __future__ import annotations

from pathlib import Path

from daedalus import cli
from daedalus.config import Settings


def _script(monkeypatch, tmp_path: Path, answers: list[str]) -> Path:
    """Wire scripted ``console.input`` answers and a throwaway DAE_HOME.

    Returns the home directory the marker file should land in. ``get_settings`` is
    patched so the wizard's marker write (and any probe timeout lookup) never touches
    the developer's real ``~/.dae``.
    """
    home = tmp_path / "home"
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(model_provider="mock", dae_home=home))

    it = iter(answers)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration as exc:  # a miscount means the wizard asked something unexpected
            raise AssertionError(f"wizard asked for more input than scripted: {prompt!r}") from exc

    monkeypatch.setattr(cli.console, "input", fake_input)
    return home


def _env(path: Path) -> dict[str, str]:
    """Parse a written .env into a dict for assertions."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


async def test_wizard_mock_provider_then_tui(monkeypatch, tmp_path):
    # Pick provider #7 (mock — no key, no probe), then surface #1 (TUI).
    home = _script(monkeypatch, tmp_path, ["7", "1"])
    env_path = tmp_path / ".env"

    surface = await cli._run_setup(env_path)

    assert surface == "tui"
    env = _env(env_path)
    assert env["MODEL_PROVIDER"] == "mock"
    assert env["MODEL_NAME"] == "mock"
    assert (home / ".setup-complete").exists()


async def test_wizard_keyed_provider_collects_key_and_picks_web(monkeypatch, tmp_path):
    # Provider #2 (Groq) -> key -> default model (blank) -> decline test -> surface #3 (Web).
    _script(monkeypatch, tmp_path, ["2", "gsk_test123", "", "n", "3"])
    env_path = tmp_path / ".env"

    surface = await cli._run_setup(env_path)

    assert surface == "web"
    env = _env(env_path)
    assert env["MODEL_PROVIDER"] == "groq"
    assert env["GROQ_API_KEY"] == "gsk_test123"
    assert env["MODEL_NAME"] == "llama-3.3-70b-versatile"  # the menu default


async def test_wizard_custom_manual_then_telegram(monkeypatch, tmp_path):
    # Provider #6 (custom) -> no curl -> URL/key/model by hand -> decline test ->
    # surface #4 (Telegram) -> token -> open mode (blank allow-list).
    _script(
        monkeypatch,
        tmp_path,
        ["6", "n", "https://api.example.test/v1", "sk-xyz", "my-model", "n", "4", "123:ABC", ""],
    )
    env_path = tmp_path / ".env"

    surface = await cli._run_setup(env_path)

    assert surface == "telegram"
    env = _env(env_path)
    assert env["MODEL_PROVIDER"] == "custom"
    assert env["OPENAI_BASE_URL"] == "https://api.example.test/v1"
    assert env["OPENAI_API_KEY"] == "sk-xyz"
    assert env["MODEL_NAME"] == "my-model"
    assert env["TELEGRAM_BOT_TOKEN"] == "123:ABC"


async def test_wizard_runs_live_test_when_accepted(monkeypatch, tmp_path):
    # Accepting the connection test must call _probe exactly once with the chosen target;
    # we stub _probe so nothing hits the network.
    _script(monkeypatch, tmp_path, ["2", "gsk_live", "llama-x", "y", "1"])
    env_path = tmp_path / ".env"

    calls: list[tuple] = []

    async def fake_probe(base_url, model, api_key):
        calls.append((base_url, model, api_key))
        return True, "ok"

    monkeypatch.setattr(cli, "_probe", fake_probe)

    surface = await cli._run_setup(env_path)

    assert surface == "tui"
    assert calls == [("https://api.groq.com/openai/v1", "llama-x", "gsk_live")]


def test_setup_env_path_prefers_cwd(monkeypatch, tmp_path):
    # With a project-local .env present, the wizard writes there (it wins at load time).
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MODEL_PROVIDER=mock\n", encoding="utf-8")
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(dae_home=tmp_path / "home"))
    assert cli._setup_env_path() == Path(".env")


def test_setup_env_path_falls_back_to_dae_home(monkeypatch, tmp_path):
    # No local .env -> write the global ~/.dae/.env so the installed command works anywhere.
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(dae_home=home))
    assert cli._setup_env_path() == home / ".env"


def test_wizard_never_blocks_a_headless_run():
    # Under pytest stdin/stdout aren't ttys, so the auto-trigger must stay silent —
    # a piped or CI invocation should never stop to ask setup questions.
    assert cli._wants_wizard() is False


def _force_tty(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)


def test_wants_wizard_true_on_fresh_interactive_machine(monkeypatch, tmp_path):
    _force_tty(monkeypatch)
    monkeypatch.chdir(tmp_path)  # no project-local .env here
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(dae_home=tmp_path / "home"))
    assert cli._wants_wizard() is True


def test_wants_wizard_false_when_already_configured(monkeypatch, tmp_path):
    _force_tty(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MODEL_PROVIDER=mock\n", encoding="utf-8")
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(dae_home=tmp_path / "home"))
    assert cli._wants_wizard() is False


def test_wants_wizard_false_after_setup_marker(monkeypatch, tmp_path):
    _force_tty(monkeypatch)
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".setup-complete").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(dae_home=home))
    assert cli._wants_wizard() is False
