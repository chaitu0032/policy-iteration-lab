// Gymnasium render view: the episode drawn with gymnasium's own sprites (client-side replica of
// env.render()), with smooth agent motion; optionally the exact pygame frame from /api/render.

import { api } from "../api.js";
import { createWorld, ensureSprites } from "../world/index.js";

const MOVE_ACTIONS = new Set([0, 1, 2, 3]);
const MAX_TWEEN_MS = 320;

function lastMoveAction(steps, frame) {
  for (let i = frame - 1; i >= 0; i -= 1) {
    if (MOVE_ACTIONS.has(steps[i].action)) return steps[i].action;
  }
  return 0;
}

export function createGymView(container, ui) {
  let world = null;
  let run = null;
  let raf = null;
  let lastKey = null;
  let realFrames = false;
  const frameCache = new Map();

  container.innerHTML = `
    <div class="gym-frame"><canvas data-role="gym-canvas" aria-label="Gymnasium render of the environment"></canvas>
      <img data-role="gym-img" alt="Frame rendered by gymnasium env.render()" hidden /></div>
    <div class="gym-caption">
      <span data-role="gym-note">Drawn with gymnasium's own sprites</span>
      <label class="toggle" title="Ask the server for the exact env.render() frame (pygame)">
        <input type="checkbox" data-role="real"/>Real env.render()</label>
    </div>`;
  const canvas = container.querySelector("canvas");
  const img = container.querySelector("img");
  const note = container.querySelector('[data-role="gym-note"]');
  container.querySelector('[data-role="real"]').addEventListener("change", (e) => {
    realFrames = e.target.checked;
    lastKey = null;
    if (snapshot) update(snapshot);
  });

  let snapshot = null;

  async function mount(nextRun, maxWidth) {
    await ensureSprites();
    run = nextRun;
    world = createWorld(canvas, run);
    world.resize(maxWidth, 440);
    frameCache.clear();
    lastKey = null;
  }

  function agentAt(ep, frame, t) {
    if (!ep || frame === 0) return { from: snapshot.startState, to: snapshot.startState, t: 1 };
    const step = ep.steps[frame - 1];
    const finished = frame === ep.steps.length;
    return {
      from: step.state, to: step.next_state, t, action: step.action,
      fell: finished && step.terminated && !ep.success,
    };
  }

  function draw(t) {
    const { episode: ep, frame } = snapshot;
    const agent = agentAt(ep, frame, t);
    const path = ep ? [ep.start_state, ...ep.steps.slice(0, frame).map((s) => s.next_state)] : [];
    world.draw({
      agent, path, layers: { path: true },
      orientation: ep ? lastMoveAction(ep.steps, frame) : 0,
    });
  }

  async function showRealFrame() {
    const { episode: ep, frame } = snapshot;
    const state = ep ? (frame === 0 ? ep.start_state : ep.steps[frame - 1].next_state) : snapshot.startState;
    const action = ep && frame > 0 ? ep.steps[frame - 1].action : null;
    const key = `${state}:${action}`;
    try {
      if (!frameCache.has(key)) {
        const res = await api.render({ env_key: run.env.key, options: run.env.options, state, last_action: action });
        frameCache.set(key, res.image);
      }
      img.src = frameCache.get(key);
      img.style.width = `${canvas.clientWidth || 360}px`;
      img.hidden = false;
      canvas.hidden = true;
      note.textContent = "Exact frame from gymnasium env.render()";
    } catch (err) {
      realFrames = false;
      container.querySelector('[data-role="real"]').checked = false;
      ui.status(`Real render unavailable: ${err.message}. Showing the sprite replica instead.`, "error");
      img.hidden = true;
      canvas.hidden = false;
      draw(1);
    }
  }

  function update(next) {
    if (!world) return;
    const prev = snapshot;
    snapshot = next;
    const key = `${next.episode ? next.episode.steps.length : "-"}:${next.frame}:${next.startState}`;
    if (key === lastKey) return;
    lastKey = key;
    cancelAnimationFrame(raf);
    if (realFrames) {
      showRealFrame();
      return;
    }
    img.hidden = true;
    canvas.hidden = false;
    note.textContent = "Drawn with gymnasium's own sprites";
    const stepped = prev && next.playing && prev.episode === next.episode && next.frame === prev.frame + 1;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (!stepped || reduce) {
      draw(1);
      return;
    }
    const duration = Math.min(MAX_TWEEN_MS, next.speedMs * 0.8);
    const t0 = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / duration);
      draw(t);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  return { mount, update, resize: (w) => { world?.resize(w, 440); lastKey = null; if (snapshot) update(snapshot); } };
}
