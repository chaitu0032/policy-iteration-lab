// Sign-in screen. Resolves with the username once the server has set the session cookie.

import { api } from "../api.js";
import { escapeHtml } from "./tooltip.js";

export function createLogin(root) {
  let resolver = null;
  let pending = null;

  root.innerHTML = `
    <form class="login-card" novalidate>
      <img src="assets/gym/elf_down.png" alt="" width="48" height="48" class="pixel" />
      <h1>Sign in to Policy Iteration Lab</h1>
      <p class="muted">Your saved runs follow your account to any browser.</p>
      <label for="login-user">Username</label>
      <input id="login-user" name="username" autocomplete="username" autocapitalize="none" spellcheck="false" required />
      <label for="login-pass">Password</label>
      <input id="login-pass" name="password" type="password" autocomplete="current-password" required />
      <p class="login-error" role="alert" aria-live="polite"></p>
      <button class="btn primary" type="submit">Sign in</button>
    </form>`;
  const form = root.querySelector("form");
  const error = root.querySelector(".login-error");
  const button = root.querySelector("button");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = form.username.value.trim();
    const password = form.password.value;
    if (!username || !password) {
      error.textContent = "Enter your username and password.";
      return;
    }
    button.disabled = true;
    button.textContent = "Signing in…";
    error.textContent = "";
    try {
      const { user } = await api.login(username, password);
      form.password.value = "";
      root.hidden = true;
      const done = resolver;
      resolver = null;
      pending = null;
      done?.(user);
    } catch (err) {
      error.innerHTML = escapeHtml(err.message);
      form.password.select();
    } finally {
      button.disabled = false;
      button.textContent = "Sign in";
    }
  });

  return {
    /** Show the screen and wait for a successful sign-in. */
    prompt() {
      root.hidden = false;
      setTimeout(() => form.username.focus(), 0);
      // Several callers may ask at once (boot + an expired request): they all share one sign-in.
      pending ??= new Promise((resolve) => { resolver = resolve; });
      return pending;
    },
  };
}
