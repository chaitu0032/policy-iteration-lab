"""Multi-user / multi-browser behaviour against a server with accounts enabled."""

from __future__ import annotations

import pytest

from conftest import ACCOUNTS, App, hermetic_context


def new_browser(browser, url):
    ctx = hermetic_context(browser)  # separate cookies = separate browser
    page = ctx.new_page()
    app = App(page, url)
    page.goto(url + "/", wait_until="domcontentloaded")
    return ctx, app


def sign_in(app, user, password=None):
    app.page.wait_for_selector("#login:not([hidden]) #login-user")
    app.page.fill("#login-user", user)
    app.page.fill("#login-pass", password or ACCOUNTS[user])
    app.page.click("#login button[type=submit]")


def signed_in(app):
    app.page.wait_for_selector("#login", state="hidden")
    app.page.wait_for_function("!!document.querySelector('[data-act=start]')")


def runs_listed(app):
    return app.js("document.querySelectorAll('#run-list .run-item').length")


@pytest.fixture()
def contexts():
    opened = []
    yield opened
    for ctx, app in opened:
        errs = list(app.errors)
        ctx.close()
        assert not errs, errs


def test_wrong_password_then_success(browser, auth_server_url, contexts):
    ctx, app = new_browser(browser, auth_server_url)
    contexts.append((ctx, app))
    sign_in(app, "alice", "wrong")
    app.page.wait_for_function("document.querySelector('.login-error').textContent.includes('Wrong username or password')")
    app.errors.clear()  # the 401 is expected
    app.page.fill("#login-pass", ACCOUNTS["alice"])
    app.page.click("#login button[type=submit]")
    signed_in(app)
    assert app.js("document.querySelector('#account-name').textContent") == "alice"


def test_same_user_two_browsers_share_runs_other_user_isolated(browser, auth_server_url, contexts):
    ctx1, laptop = new_browser(browser, auth_server_url)
    ctx2, phone = new_browser(browser, auth_server_url)
    ctx3, bob = new_browser(browser, auth_server_url)
    contexts.extend([(ctx1, laptop), (ctx2, phone), (ctx3, bob)])
    for app, user in [(laptop, "alice"), (phone, "alice"), (bob, "bob")]:
        sign_in(app, user)
        signed_in(app)
    before = runs_listed(phone)
    laptop.set_option("map_name", "4x4")
    laptop.train()
    phone.page.click("#refresh-runs")
    phone.page.wait_for_function(f"document.querySelectorAll('#run-list .run-item').length === {before + 1}")
    phone.page.locator("#run-list .run-item").first.click()
    phone.page.wait_for_function("!document.querySelector('#run-view').hidden")
    assert phone.js("document.querySelector('#run-header h1').textContent") == "Frozen Lake"
    bob.page.click("#refresh-runs")
    bob.page.wait_for_timeout(500)
    assert runs_listed(bob) == 0, "bob must not see alice's runs"
    # both alice sessions train at the same time without interfering
    laptop.switch_env("Taxi")
    phone.switch_env("CliffWalking")
    laptop.page.click("#train-card [data-act=start]")
    phone.page.click("#train-card [data-act=start]")
    laptop.wait_trained()
    phone.wait_trained()


def test_sign_out_returns_to_login(browser, auth_server_url, contexts):
    ctx, app = new_browser(browser, auth_server_url)
    contexts.append((ctx, app))
    sign_in(app, "bob")
    signed_in(app)
    app.page.click("#logout")
    app.page.wait_for_selector("#login:not([hidden])")
    app.errors.clear()  # the post-logout session check is a 401 by design
