// Short theory notes + equations (rendered with KaTeX when available, plain TeX otherwise).

const PRIMER = String.raw`
<h3>What is policy iteration?</h3>
<p>The environment is a Markov decision process: states \(s\), actions \(a\), transition probabilities
\(P(s'\mid s,a)\) and rewards \(r\). Gymnasium exposes the full model as <code>env.unwrapped.P</code>,
so the optimal policy can be <em>planned</em> exactly. No trial-and-error learning is needed.</p>
<ol class="eq-steps">
  <li><b>Policy evaluation:</b> compute how good the current policy \(\pi_k\) is. Repeat Bellman sweeps until \(\max_s|\Delta V| < \theta\):
    $$V_{j+1}(s) \leftarrow \sum_{s'} P(s'\mid s,\pi_k(s))\,\big[r + \gamma\,V_j(s')\big]$$
    or solve the linear system exactly: \(V^{\pi_k} = (I-\gamma P_{\pi_k})^{-1} R_{\pi_k}\).</li>
  <li><b>Action values:</b> \(\;Q^{\pi_k}(s,a) = \sum_{s'} P(s'\mid s,a)\,\big[r + \gamma\,V^{\pi_k}(s')\big]\)</li>
  <li><b>Policy improvement:</b> act greedily, \(\;\pi_{k+1}(s) = \arg\max_a Q^{\pi_k}(s,a)\) (ties keep the old action).</li>
  <li><b>Stop</b> when \(\pi_{k+1} = \pi_k\). That policy is optimal, \(\pi^\star\). Each step never makes things worse
    (\(V^{\pi_{k+1}} \ge V^{\pi_k}\)), and there are finitely many policies, so it always terminates, usually in only a handful of iterations.</li>
</ol>
<p class="muted">\(\gamma\) (discount) trades off immediate against future reward; \(\theta\) is the evaluation tolerance.</p>`;

const ENV_NOTES = {
  FrozenLake: String.raw`<p><b>Frozen Lake.</b> States: grid cells. Actions: ←↓→↑. Reaching G gives the goal reward; falling into
    H ends the episode. On slippery ice the agent moves in the intended direction with probability <em>success rate</em>
    and otherwise slides perpendicular, so the best path often hugs walls away from holes.</p>`,
  CliffWalking: String.raw`<p><b>Cliff Walking.</b> 4×12 grid, start bottom-left, goal bottom-right. Every step costs a reward,
    and stepping onto the cliff costs a large penalty and teleports you back to start. With \(\gamma\to 1\) the optimal path runs right along the cliff edge;
    with slippery moves or a lower \(\gamma\) it detours to safer rows.</p>`,
  Taxi: String.raw`<p><b>Taxi.</b> 500 states = taxi row × col (25) × passenger location (R, G, Y, B or in taxi) × destination (4).
    Six actions: four moves, Pickup and Dropoff. The boards show a 5×5 <em>slice</em> for one passenger/destination pair;
    change the slice to see the other parts of the value function.</p>`,
};

const VIEW_NOTES = {
  training: String.raw`Each chip is one policy \(\pi_k\). The board colours every state by \(V^{\pi_k}(s)\) and draws the arrow
    \(\pi_k(s)\). Amber outlines mark states where \(\pi_k(s)\ne\pi_{k-1}(s)\). Drag the <b>eval</b> slider to watch the
    Bellman sweeps \(V_0=0 \to V_1 \to \dots \to V^{\pi_k}\) converge; the residual chart shows \(\max_s|V_{j+1}(s)-V_j(s)|\)
    falling geometrically (rate \(\approx\gamma\)) until it drops below \(\theta\).`,
  values: String.raw`Small multiples of \(V^{\pi_0}, V^{\pi_1}, \dots, V^{\pi^\star}\) on one shared colour scale. The policy
    improvement theorem guarantees \(V^{\pi_{k+1}}(s) \ge V^{\pi_k}(s)\) for every state, so the maps can only get brighter.`,
  policy: String.raw`Rows are states, columns are policies. Each cell is the greedy action
    \(\pi_k(s)=\arg\max_a Q^{\pi_{k-1}}(s,a)\) or its value \(V^{\pi_k}(s)\). Amber cells are where the policy changed. Policy iteration
    usually fixes most states in the first one or two improvements.`,
  inference: String.raw`The inference engine plays the chosen policy in the real gymnasium env: \(a_t=\pi(s_t)\),
    \(s_{t+1}\sim P(\cdot\mid s_t,a_t)\). The return is \(G=\sum_t r_t\) and the discounted return is \(G_\gamma=\sum_t \gamma^t r_t\).
    On average over many episodes \(\mathbb{E}[G_\gamma\mid s_0]=V^\pi(s_0)\), the "Expected V(s₀)" tile. A single stochastic episode can land far from it.`,
  benchmark: String.raw`A Monte Carlo check of the planner. Each policy runs \(N\) episodes with the same seeds, and
    \(\hat V(s_0)=\tfrac1N\sum_{i} G^{(i)}_\gamma \to V^\pi(s_0)\) as \(N\) grows. If the yellow dashed model line and the measured line disagree,
    the env has dynamics the model doesn't capture (for example Taxi's fickle passenger) or episodes are being truncated.`,
};

function typeset(el) {
  if (window.renderMathInElement) {
    window.renderMathInElement(el, {
      delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }],
      throwOnError: false,
    });
  }
}

export function renderPrimer(el, envKey) {
  el.innerHTML = `${PRIMER}${ENV_NOTES[envKey] || ""}`;
  typeset(el);
}

/** Collapsible "What am I looking at?" note placed at the top of a view panel. */
export function explainerHtml(view) {
  return `<details class="explainer" data-explainer="${view}"><summary>ⓘ What am I looking at?</summary><div class="explainer-body">${VIEW_NOTES[view]}</div></details>`;
}

export function typesetExplainers(root) {
  root.querySelectorAll(".explainer-body:not([data-typeset])").forEach((el) => {
    typeset(el);
    el.dataset.typeset = "1";
  });
}

export function envNoteHtml(envKey) {
  return ENV_NOTES[envKey] || "";
}
