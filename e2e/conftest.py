"""E2E fixtures: a real uvicorn server + a headless Chromium page that records every JS error."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
IGNORED_CONSOLE = ("favicon", "Failed to load resource: the server responded with a status of 404")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(env_extra: dict[str, str], data_dir: Path):
    import os
    port = _free_port()
    env = {**os.environ, "PI_LAB_IGNORE_ENV_LOCAL": "1", "PI_LAB_DATA_DIR": str(data_dir), **env_extra}
    env.pop("PI_LAB_USERS", None) if "PI_LAB_USERS" not in env_extra else None
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=1)
            return proc, url
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("server did not start")


@pytest.fixture(scope="session")
def server_url(tmp_path_factory):
    proc, url = _start_server({}, tmp_path_factory.mktemp("runs-open"))
    yield url
    proc.terminate()
    proc.wait(timeout=10)


ACCOUNTS = {"alice": "alice-pass-123", "bob": "bob-pass-456"}


@pytest.fixture(scope="session")
def auth_server_url(tmp_path_factory):
    sys.path.insert(0, str(ROOT))
    from backend.auth import hash_password
    users = ",".join(f"{u}:{hash_password(p, iterations=2000)}" for u, p in ACCOUNTS.items())
    proc, url = _start_server({"PI_LAB_USERS": users, "PI_LAB_SECRET": "e2e-secret-" + "x" * 40},
                              tmp_path_factory.mktemp("runs-auth"))
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


class App:
    """Thin page-object around the UI with invariant checks."""

    def __init__(self, page, url):
        self.page = page
        self.url = url
        self.errors: list[str] = []
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("console", self._on_console)
        page.set_default_timeout(15_000)

    def _on_console(self, msg):
        if msg.type == "error" and not any(s in msg.text for s in IGNORED_CONSOLE):
            self.errors.append(f"console.error: {msg.text}")

    # ---- actions -------------------------------------------------------
    def open(self):
        self.page.goto(self.url + "/", wait_until="domcontentloaded")
        self.page.wait_for_selector("#env-tabs button")
        self.page.wait_for_function("!!document.querySelector('[data-act=start]')")

    def js(self, expr, *args):
        return self.page.evaluate(expr, *args)

    def status(self):
        return self.js("document.querySelector('#status').textContent")

    def switch_env(self, key):
        self.page.click(f"[data-env={key}]")

    def set_option(self, name, value):
        self.js(
            """([name, value]) => {
                const el = document.querySelector(`[data-name="${name}"]`);
                if (el.type === 'checkbox') el.checked = value; else el.value = value;
                el.dispatchEvent(new Event('input', {bubbles: true}));
            }""",
            [name, value],
        )

    def train(self, wait=True, timeout=20_000):
        self.page.click("#train-card [data-act=start]")
        if wait:
            self.wait_trained(timeout)

    def wait_trained(self, timeout=20_000):
        self.page.wait_for_function(
            "() => /Converged|stopped|max iterations/i.test(document.querySelector('#status').textContent)"
            " && !document.querySelector('#train-card [data-act=start]').disabled",
            timeout=timeout,
        )

    def view(self, name):
        self.page.click(f"#view-tabs [data-view={name}]")

    # ---- state ---------------------------------------------------------
    def state(self):
        return self.js("""() => ({
            bodyEnv: document.body.dataset.env,
            activeTab: document.querySelector('#env-tabs button.active')?.dataset.env,
            formTitle: document.querySelector('#train-card h2')?.textContent,
            runVisible: !document.querySelector('#run-view').hidden,
            emptyVisible: !document.querySelector('#empty-state').hidden,
            header: document.querySelector('#run-header h1')?.textContent || null,
            chips: document.querySelectorAll('[data-panel=training] .it-chip[data-iter]').length,
            startDisabled: document.querySelector('#train-card [data-act=start]').disabled,
            waiting: document.body.innerText.includes('Waiting for π'),
        })""")

    def assert_consistent(self, env_title_by_key):
        s = self.state()
        assert s["bodyEnv"] == s["activeTab"], s
        assert env_title_by_key[s["activeTab"]] in s["formTitle"], s
        if s["runVisible"]:
            assert s["header"] == env_title_by_key[s["activeTab"]], f"run of another env shown: {s}"
        assert not (s["waiting"] and not s["startDisabled"]), f"stuck waiting with no training: {s}"
        assert self.js("1 + 1") == 2  # page responsive
        assert not self.errors, self.errors


def hermetic_context(browser):
    """Browser context that never touches the internet: CDN fonts/KaTeX get empty stubs.

    The app degrades gracefully without them (system fonts, plain-text equations), and the suite
    no longer depends on network weather.
    """
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})

    def handle(route):
        url = route.request.url
        if url.startswith(("http://127.0.0.1", "http://localhost")):
            route.continue_()
        else:
            ctype = "text/css" if url.endswith(".css") or "fonts.googleapis" in url else "application/javascript"
            route.fulfill(status=200, body="", content_type=ctype)

    ctx.route("**/*", handle)
    return ctx


@pytest.fixture()
def app(browser, server_url):
    ctx = hermetic_context(browser)
    page = ctx.new_page()
    a = App(page, server_url)
    a.open()
    yield a
    errors = list(a.errors)
    ctx.close()
    assert not errors, errors


TITLES = {"FrozenLake": "Frozen Lake", "CliffWalking": "Cliff Walking", "Taxi": "Taxi"}
