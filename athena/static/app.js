const taskForm = document.querySelector("#task-form");
const factForm = document.querySelector("#fact-form");
const conversation = document.querySelector("#conversation");
const emptyState = document.querySelector("#empty-state");
const errorBanner = document.querySelector("#error-banner");
const askButton = document.querySelector("#ask-button");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function showError(error) {
  errorBanner.textContent = error.message || String(error);
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function percent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderDecision(decision) {
  emptyState?.remove();
  const card = element("article", "decision-card");
  card.dataset.decisionId = decision.id;

  card.append(element("div", "situation-copy", decision.situation));
  const main = element("div", "decision-main");
  const meta = element("div", "decision-meta");
  meta.append(element("span", "action-chip", decision.selected.action));
  meta.append(
    element(
      "span",
      "confidence-copy",
      `${percent(decision.predicted_reward)} predicted success · ${decision.context_key}`,
    ),
  );
  main.append(meta);
  main.append(element("p", "decision-response", decision.selected.response));

  const alternatives = element("div", "alternatives");
  decision.alternatives.forEach((item) => {
    const row = element("div", "alternative-row");
    row.append(element("span", "", item.candidate.action));
    const meter = element("progress", "meter");
    meter.max = 1;
    meter.value = item.predicted_reward;
    row.append(meter);
    row.append(element("span", "", percent(item.predicted_reward)));
    alternatives.append(row);
  });
  main.append(alternatives);
  card.append(main);

  const feedback = element("div", "feedback");
  const observation = document.createElement("input");
  observation.placeholder = "What actually happened? (optional)";
  observation.maxLength = 3000;
  feedback.append(observation);
  const actions = element("div", "feedback-actions");
  [
    ["Failed", 0],
    ["Partly", 0.5],
    ["Worked", 1],
  ].forEach(([label, reward]) => {
    const button = element("button", "", label);
    button.type = "button";
    button.dataset.reward = reward;
    button.addEventListener("click", async () => {
      clearError();
      actions.querySelectorAll("button").forEach((item) => {
        item.disabled = true;
      });
      try {
        const payload = await api("/api/learn", {
          method: "POST",
          body: JSON.stringify({
            decision_id: decision.id,
            reward: Number(reward),
            observation: observation.value,
            reliability: 1,
          }),
        });
        feedback.remove();
        const result = payload.report;
        const copy = result.knowledge.status === "provisional"
          ? `Experience stored. Prediction error ${result.prediction_error.toFixed(3)}.`
          : `Experience stored. Strategy is now ${result.knowledge.status}.`;
        card.append(element("div", "feedback-result", copy));
        renderState(payload.state);
      } catch (error) {
        actions.querySelectorAll("button").forEach((item) => {
          item.disabled = false;
        });
        showError(error);
      }
    });
    actions.append(button);
  });
  feedback.append(actions);
  card.append(feedback);
  conversation.prepend(card);
}

function renderStrategies(items) {
  const container = document.querySelector("#strategy-list");
  document.querySelector("#strategy-count").textContent = items.length;
  container.replaceChildren();
  container.className = "state-list";
  if (!items.length) {
    container.classList.add("empty-copy");
    container.textContent = "No strategy evidence yet.";
    return;
  }
  items.slice().reverse().forEach((item) => {
    const card = element("div", "state-item");
    const top = element("div", "state-item-top");
    top.append(element("strong", "", `${item.context_key} · ${item.action}`));
    top.append(element("small", `status-${item.status}`, item.status));
    card.append(top);
    card.append(
      element(
        "p",
        "",
        `${percent(item.mean_reward)} expected · ${item.effective_samples.toFixed(1)} evidence`,
      ),
    );
    container.append(card);
  });
}

function renderMemories(items) {
  const container = document.querySelector("#memory-list");
  document.querySelector("#memory-count").textContent = items.length;
  container.replaceChildren();
  container.className = "state-list";
  if (!items.length) {
    container.classList.add("empty-copy");
    container.textContent = "No outcomes recorded yet.";
    return;
  }
  items.slice(0, 8).forEach((item) => {
    const card = element("div", "state-item");
    const top = element("div", "state-item-top");
    top.append(element("strong", "", `${item.context_key} · ${item.action}`));
    const outcome = item.reward >= 0.75 ? "worked" : item.reward <= 0.25 ? "failed" : "partial";
    top.append(element("small", "", outcome));
    card.append(top);
    card.append(element("p", "", item.observation || item.situation));
    container.append(card);
  });
}

function renderFacts(items) {
  const container = document.querySelector("#fact-list");
  document.querySelector("#fact-count").textContent = items.length;
  container.replaceChildren();
  container.className = "state-list";
  if (!items.length) {
    container.classList.add("empty-copy");
    container.textContent = "No factual evidence yet.";
    return;
  }
  items.forEach((item) => {
    const card = element("div", "state-item");
    const top = element("div", "state-item-top");
    top.append(element("strong", "", item.key));
    top.append(
      element(
        "small",
        item.consolidated ? "status-preferred" : "status-provisional",
        item.consolidated ? "consolidated" : "provisional",
      ),
    );
    card.append(top);
    card.append(element("p", "", `${item.value} · ${percent(item.confidence)} confidence`));
    container.append(card);
  });
}

function renderState(state) {
  document.querySelector("#backend-badge").textContent = state.backend;
  document.querySelector("#experience-count").textContent =
    `${state.total_experiences} experience${state.total_experiences === 1 ? "" : "s"}`;
  renderStrategies(state.strategies || []);
  renderMemories(state.episodes || []);
  renderFacts(state.beliefs || []);
}

taskForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  askButton.disabled = true;
  const data = new FormData(taskForm);
  try {
    const payload = await api("/api/decide", {
      method: "POST",
      body: JSON.stringify({
        situation: data.get("situation"),
        context_key: data.get("context_key"),
      }),
    });
    renderDecision(payload.decision);
    renderState(payload.state);
    document.querySelector("#situation").value = "";
  } catch (error) {
    showError(error);
  } finally {
    askButton.disabled = false;
  }
});

factForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const data = new FormData(factForm);
  const submit = factForm.querySelector("button");
  submit.disabled = true;
  try {
    const payload = await api("/api/facts", {
      method: "POST",
      body: JSON.stringify({
        key: data.get("key"),
        value: data.get("value"),
        source: data.get("source"),
        reliability: Number(data.get("reliability")),
      }),
    });
    renderState(payload.state);
    factForm.reset();
    factForm.querySelector('[name="reliability"]').value = "1";
  } catch (error) {
    showError(error);
  } finally {
    submit.disabled = false;
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector("#situation").value = button.dataset.prompt;
    document.querySelector("#context-key").value = button.dataset.context;
    document.querySelector("#situation").focus();
  });
});

api("/api/state")
  .then((state) => {
    renderState(state);
    (state.pending_decisions || []).forEach(renderDecision);
  })
  .catch(showError);
