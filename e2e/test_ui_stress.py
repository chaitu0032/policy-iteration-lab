"""UI stress tests: drive the real app through every flow and check invariants after each step.

Run:  python3 -m pytest e2e -q          (needs `pip install playwright && playwright install chromium`)
"""

from __future__ import annotations

import random

import pytest

from conftest import TITLES

ENVS = list(TITLES)


def preview_size(app):
    return app.js("() => { const c = document.querySelector('#preview-canvas'); return [c.clientWidth, c.clientHeight]; }")


def expected_preview(app, key):
    # Aspect ratio of the gymnasium render: FrozenLake 8x8 square, CliffWalking 12x4, Taxi 11x7 tiles.
    w, h = preview_size(app)
    ratio = w / h if h else 0
    return {"FrozenLake": abs(ratio - 1) < 0.05, "CliffWalking": abs(ratio - 3) < 0.1,
            "Taxi": abs(ratio - 11 / 7) < 0.05}[key]


def run_list_count(app):
    return app.js("document.querySelectorAll('#run-list .run-item').length")


# ---------------------------------------------------------------- environment switching

def test_env_tabs_switch_consistently(app):
    for _ in range(3):
        for key in ENVS:
            app.switch_env(key)
            app.page.wait_for_function(f"() => document.body.dataset.env === '{key}'")
            app.page.wait_for_timeout(450)
            app.assert_consistent(TITLES)
            assert expected_preview(app, key), (key, preview_size(app))


def test_rapid_tab_switching_preview_ends_on_last_env(app):
    for _ in range(4):
        for key in ENVS:
            app.switch_env(key)
    app.page.wait_for_timeout(900)
    app.assert_consistent(TITLES)
    assert expected_preview(app, "Taxi"), preview_size(app)


def test_option_change_refreshes_what_is_shown(app):
    app.train()
    assert app.state()["runVisible"]
    app.set_option("map_name", "4x4")
    app.page.wait_for_timeout(500)
    s = app.state()
    assert not s["runVisible"] and s["emptyVisible"], "changing env options must show the new configuration"
    assert preview_size(app)[0] != 0
    app.train()
    assert app.js("document.querySelector('#run-header').textContent").count("4x4") >= 1
    app.assert_consistent(TITLES)


# ---------------------------------------------------------------- training

@pytest.mark.parametrize("key", ENVS)
def test_train_each_env(app, key):
    app.switch_env(key)
    before = run_list_count(app)
    app.train()
    s = app.state()
    assert s["runVisible"] and s["header"] == TITLES[key] and s["chips"] >= 2, s
    assert run_list_count(app) == before + 1
    app.assert_consistent(TITLES)


def test_switch_env_blocked_while_training_then_stop(app):
    app.set_option("delay_ms", 600)
    app.train(wait=False)
    app.page.wait_for_function("document.querySelectorAll('[data-panel=training] .it-chip[data-iter]').length >= 1")
    app.switch_env("Taxi")
    assert app.state()["activeTab"] == "FrozenLake"
    assert "Stop the running training" in app.status()
    app.page.click("#train-card [data-act=stop]")
    app.wait_trained()
    assert "stopped" in app.status().lower()
    app.switch_env("Taxi")
    app.page.wait_for_timeout(300)
    app.assert_consistent(TITLES)


def test_pause_resume_stop(app):
    app.set_option("delay_ms", 300)
    app.train(wait=False)
    app.page.wait_for_function("document.querySelectorAll('[data-panel=training] .it-chip[data-iter]').length >= 2")
    app.page.click("#train-card [data-act=pause]")
    app.page.wait_for_timeout(400)
    n = app.state()["chips"]
    app.page.wait_for_timeout(1200)
    assert app.state()["chips"] == n, "chips must not grow while paused"
    app.page.click("#train-card [data-act=pause]")  # resume
    app.page.wait_for_function(f"document.querySelectorAll('[data-panel=training] .it-chip[data-iter]').length > {n}")
    app.page.click("#train-card [data-act=stop]")
    app.wait_trained()
    app.assert_consistent(TITLES)


def test_double_click_start_creates_one_run(app):
    before = run_list_count(app)
    app.page.dblclick("#train-card [data-act=start]")
    app.wait_trained()
    app.page.wait_for_timeout(300)
    assert run_list_count(app) == before + 1
    app.assert_consistent(TITLES)


def test_invalid_custom_map_reports_error_and_recovers(app):
    app.set_option("map_name", "custom")
    app.set_option("custom_map", "SFX\nFFG")
    app.page.click("#train-card [data-act=start]")
    app.page.wait_for_function("document.querySelector('#status').classList.contains('error')")
    s = app.state()
    assert not s["startDisabled"] and not s["waiting"] and not s["runVisible"], s
    app.errors.clear()  # the invalid map legitimately produced a 422 console entry
    app.set_option("custom_map", "SFF\nFHF\nFFG")
    app.train()
    assert app.state()["runVisible"]


def test_reload_mid_training_recovers(app):
    app.set_option("delay_ms", 800)
    app.train(wait=False)
    app.page.wait_for_function("document.querySelectorAll('[data-panel=training] .it-chip[data-iter]').length >= 1")
    app.page.reload(wait_until="domcontentloaded")
    app.page.wait_for_function("!!document.querySelector('[data-act=start]')")
    s = app.state()
    assert not s["startDisabled"] and not s["waiting"], s
    app.train()
    app.assert_consistent(TITLES)


@pytest.mark.parametrize("seed", [1, 2])
def test_repeated_randomised_runs(app, seed):
    rng = random.Random(seed)
    for _ in range(5):
        key = rng.choice(ENVS)
        app.switch_env(key)
        if key == "FrozenLake":
            app.set_option("map_name", rng.choice(["4x4", "8x8"]))
            app.set_option("is_slippery", rng.random() < 0.5)
        elif key == "CliffWalking":
            app.set_option("is_slippery", rng.random() < 0.5)
        else:
            app.set_option("is_rainy", rng.random() < 0.5)
        app.set_option("eval_mode", rng.choice(["iterative", "exact"]))
        app.set_option("delay_ms", 0)
        app.train()
        for view in ["values", "policy", "training"]:
            app.view(view)
        app.assert_consistent(TITLES)


# ---------------------------------------------------------------- views, inference, benchmark, runs

def test_view_tabs_rapid_switching_during_and_after_training(app):
    app.set_option("delay_ms", 200)
    app.train(wait=False)
    for _ in range(10):
        for view in ["values", "policy", "training"]:
            app.view(view)
    app.wait_trained()
    for _ in range(5):
        for view in ["values", "policy", "inference", "benchmark", "training"]:
            app.view(view)
    app.assert_consistent(TITLES)


def test_inference_every_policy_and_board_modes(app):
    app.set_option("map_name", "4x4")
    app.train()
    app.view("inference")
    n = app.js("document.querySelectorAll('[data-role=policy] option').length")
    assert n >= 2
    app.js("() => { document.querySelector('[data-role=maxsteps]').value = 40; document.querySelector('[data-role=maxsteps]').dispatchEvent(new Event('change')); }")
    for k in range(n):
        app.page.select_option("[data-role=policy]", str(k))
        app.page.click("[data-panel=inference] [data-act=start]")
        app.page.wait_for_function("/goal|failed|truncated/.test(document.querySelector('[data-role=outcome]').textContent)", timeout=30_000)
    for mode in ["analysis", "gym", "both"]:
        app.page.click(f"[data-boardmode={mode}]")
        app.page.wait_for_timeout(100)
    app.page.click("[data-panel=inference] [data-act=reset]")
    app.page.click("[data-role=real]")
    app.page.wait_for_timeout(1200)
    app.assert_consistent(TITLES)


def test_open_other_run_while_episode_plays(app):
    app.train()
    app.switch_env("CliffWalking")
    app.train()
    app.view("inference")
    app.js("() => { const s = document.querySelector('[data-role=speed]'); s.value = 40; s.dispatchEvent(new Event('input')); }")
    app.page.click("[data-panel=inference] [data-act=start]")
    app.page.wait_for_timeout(300)
    frozen_run = app.page.locator("#run-list .run-item").filter(has_text="Frozen Lake")
    # FrozenLake runs are not listed while CliffWalking is selected; switch first.
    app.switch_env("FrozenLake")
    app.page.locator("#run-list .run-item").first.click()
    app.page.wait_for_timeout(500)
    assert frozen_run is not None
    app.assert_consistent(TITLES)


def test_open_saved_run_loads_its_options_into_the_form(app):
    app.set_option("map_name", "4x4")
    app.train()
    app.switch_env("Taxi")
    app.train()
    app.switch_env("FrozenLake")
    app.set_option("map_name", "8x8")
    app.page.locator("#run-list .run-item").first.click()
    app.page.wait_for_timeout(300)
    assert app.js("document.querySelector('[data-name=map_name]').value") == "4x4", "form must show the opened run's options"
    app.assert_consistent(TITLES)


def test_benchmark_run_and_stop(app):
    app.set_option("map_name", "4x4")
    app.train()
    app.view("benchmark")
    app.page.click("[data-panel=benchmark] [data-act=run]")
    app.page.wait_for_timeout(300)
    app.page.click("[data-panel=benchmark] [data-act=stop]")
    app.page.click("[data-panel=benchmark] [data-act=run]")
    app.page.wait_for_function("!!document.querySelector('[data-panel=benchmark] table')", timeout=60_000)
    app.view("training")
    app.view("benchmark")
    app.assert_consistent(TITLES)


def test_delete_open_run(app):
    app.train()
    before = run_list_count(app)
    app.page.locator("#run-list .run-item.active [data-act=delete]").click()
    app.page.wait_for_timeout(300)
    s = app.state()
    assert run_list_count(app) == before - 1 and not s["runVisible"], s
    app.assert_consistent(TITLES)


def test_taxi_slice_and_start_state_by_click(app):
    app.switch_env("Taxi")
    app.train()
    app.view("inference")
    app.page.select_option("[data-role=t-pass]", "2")
    app.page.select_option("[data-role=t-dest]", "3")
    box = app.page.locator("[data-panel=inference] canvas.board").bounding_box()
    app.page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    app.page.click("[data-panel=inference] [data-act=start]")
    app.page.wait_for_function("/goal|failed|truncated/.test(document.querySelector('[data-role=outcome]').textContent)", timeout=30_000)
    assert "goal" in app.js("document.querySelector('[data-role=outcome]').textContent")
    app.assert_consistent(TITLES)


def test_window_resize_keeps_boards_working(app):
    app.train()
    for w in [1600, 1000, 760, 1400]:
        app.page.set_viewport_size({"width": w, "height": 900})
        app.page.wait_for_timeout(300)
        for view in ["training", "inference"]:
            app.view(view)
    app.assert_consistent(TITLES)
