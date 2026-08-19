const taskForm = document.querySelector("#task-form");
const factForm = document.querySelector("#fact-form");
const conversation = document.querySelector("#conversation");
const emptyState = document.querySelector("#empty-state");
const errorBanner = document.querySelector("#error-banner");
const askButton = document.querySelector("#ask-button");
const discoverSkillButton = document.querySelector("#discover-skill-button");
const skillTrialResult = document.querySelector("#skill-trial-result");
const skillRunForm = document.querySelector("#skill-run-form");
const skillRunOutput = document.querySelector("#skill-run-output");
const toolLearningForm = document.querySelector("#tool-learning-form");
const toolLearningResult = document.querySelector("#tool-learning-result");
const plasticityLearningForm = document.querySelector("#plasticity-learning-form");
const plasticityLearningResult = document.querySelector("#plasticity-learning-result");
const plasticityRunForm = document.querySelector("#plasticity-run-form");
const plasticityRunOutput = document.querySelector("#plasticity-run-output");

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

function renderSkills(items) {
  const container = document.querySelector("#skill-list");
  const select = document.querySelector("#skill-select");
  const previousSelection = select.value;
  document.querySelector("#skill-count").textContent = items.length;
  container.replaceChildren();
  select.replaceChildren();
  container.className = "state-list";
  if (!items.length) {
    container.classList.add("empty-copy");
    container.textContent = "No procedural skills yet.";
    skillRunForm.hidden = true;
    return;
  }
  items.forEach((item) => {
    const card = element("div", "state-item");
    const top = element("div", "state-item-top");
    top.append(element("strong", "", item.name));
    top.append(element("small", "status-preferred", `v${item.version}`));
    card.append(top);
    card.append(
      element(
        "p",
        "",
        `${item.program} · ${item.verification_cases} held-out checks`,
      ),
    );
    container.append(card);

    const option = element("option", "", item.name);
    option.value = item.name;
    select.append(option);
  });
  if ([...select.options].some((option) => option.value === previousSelection)) {
    select.value = previousSelection;
  }
  skillRunForm.hidden = false;
}

function renderToolSkills(items) {
  const container = document.querySelector("#tool-skill-list");
  document.querySelector("#tool-skill-count").textContent = items.length;
  container.replaceChildren();
  container.className = "state-list";
  if (!items.length) {
    container.classList.add("empty-copy");
    container.textContent = "No tool workflows yet.";
    return;
  }
  items.forEach((item) => {
    const card = element("div", "state-item");
    const top = element("div", "state-item-top");
    top.append(element("strong", "", item.name));
    top.append(element("small", "status-preferred", `v${item.version}`));
    card.append(top);
    card.append(
      element(
        "p",
        "",
        `${item.task_kind} · ${item.required_capabilities.join(" → ")} · ${item.validation_worlds} worlds`,
      ),
    );
    container.append(card);
  });
}

function renderPlasticSkills(items) {
  const container = document.querySelector("#plastic-skill-list");
  const select = document.querySelector("#plasticity-select");
  const previousSelection = select.value;
  document.querySelector("#plastic-skill-count").textContent = items.length;
  container.replaceChildren();
  select.replaceChildren();
  container.className = "state-list";
  if (!items.length) {
    container.classList.add("empty-copy");
    container.textContent = "No neural operators yet.";
    plasticityRunForm.hidden = true;
    return;
  }
  items.forEach((item) => {
    const card = element("div", "state-item");
    const top = element("div", "state-item-top");
    top.append(element("strong", "", item.name));
    top.append(element("small", "status-preferred", `v${item.version}`));
    card.append(top);
    card.append(
      element(
        "p",
        "",
        `${percent(item.validation_accuracy)} verified · ${item.replay_examples} replay cases · ${item.checksum}`,
      ),
    );
    container.append(card);

    const option = element("option", "", item.name);
    option.value = item.name;
    select.append(option);
  });
  if ([...select.options].some((option) => option.value === previousSelection)) {
    select.value = previousSelection;
  }
  plasticityRunForm.hidden = false;
}

function renderPlasticityLearning(payload) {
  const report = payload.report;
  plasticityLearningResult.replaceChildren();
  plasticityLearningResult.className = "skill-trial-result";
  const summary = element("div", "trial-summary");
  [
    [percent(report.before_accuracy), "Before learning"],
    [percent(report.candidate_accuracy), "Held-out accuracy"],
    [percent(payload.transfer.accuracy), "Magnitude transfer"],
    [report.weight_delta.toFixed(2), "Neural weight change"],
  ].forEach(([value, label]) => {
    const metric = element("div", "trial-metric");
    metric.append(element("strong", "", value));
    metric.append(element("small", "", label));
    summary.append(metric);
  });
  plasticityLearningResult.append(summary);

  const log = element("div", "experiment-log");
  [
    ["candidate", `${report.training_examples} experiences changed isolated weights`],
    ["verification", `${report.validation_examples} unseen cases · ${percent(report.candidate_accuracy)}`],
    ["replay", `${report.replay_examples} cases protected · ${percent(report.regression_accuracy)}`],
    ["parameters", `${report.checksum_before} → ${report.checksum_after}`],
  ].forEach(([label, value]) => {
    const row = element("div");
    row.append(element("span", "", label));
    row.append(element("span", "", value));
    log.append(row);
  });
  plasticityLearningResult.append(log);
  plasticityLearningResult.append(
    element(
      "div",
      "feedback-result",
      report.promoted
        ? `Promoted neural expert v${report.version}: ${report.reason}.`
        : `Rolled back: ${report.reason}.`,
    ),
  );
}

function renderToolLearning(payload) {
  const learning = payload.learning;
  const acquisition = learning.acquisition;
  const transfer = payload.transfer;
  toolLearningResult.replaceChildren();
  toolLearningResult.className = "skill-trial-result";

  const passedValidations = learning.validations.filter((item) => item.passed).length;
  const summary = element("div", "trial-summary");
  [
    [acquisition.trace.length, "Acquisition calls"],
    [acquisition.reasoner_steps, "Reasoning decisions"],
    [`${passedValidations}/${learning.validations.length}`, "Held-out worlds"],
    [transfer?.success ? "Passed" : "Failed", "Cross-world transfer"],
  ].forEach(([value, label]) => {
    const metric = element("div", "trial-metric");
    metric.append(element("strong", "", value));
    metric.append(element("small", "", label));
    summary.append(metric);
  });
  toolLearningResult.append(summary);

  const heading = element("div", "tool-trace-heading");
  heading.append(element("span", "", "Prediction → tool observation"));
  heading.append(element("span", "permission-chip", "sandboxed"));
  toolLearningResult.append(heading);
  const log = element("div", "experiment-log");
  acquisition.trace.forEach((experience) => {
    const row = element("div", experience.result.ok ? "" : "trace-error");
    const observation = experience.result.error
      || experience.result.output.capability
      || JSON.stringify(experience.result.output);
    row.append(
      element(
        "span",
        "",
        `${experience.index}. ${experience.decision.tool_name}: ${experience.decision.hypothesis}`,
      ),
    );
    row.append(element("span", "", `${experience.permission} → ${observation}`));
    log.append(row);
  });
  toolLearningResult.append(log);
  const resultCopy = learning.consolidated && transfer?.success
    ? `Verified, consolidated, and transferred with ${transfer.reasoner_steps} new foundation-model decisions.`
    : `Not consolidated: ${learning.reason}`;
  toolLearningResult.append(element("div", "feedback-result", resultCopy));
}

function renderSkillTrial(payload) {
  const learning = payload.learning;
  const verification = learning.consolidation.verification;
  skillTrialResult.replaceChildren();
  skillTrialResult.className = "skill-trial-result";

  const summary = element("div", "trial-summary");
  [
    [learning.gap_before.hypotheses_remaining, "Initial hypotheses"],
    [learning.experiments.length, "Chosen experiments"],
    [payload.revealed_world_program, "Induced procedure"],
    [`${verification.passed_cases}/${verification.total_cases}`, "Held-out verification"],
  ].forEach(([value, label]) => {
    const metric = element("div", "trial-metric");
    metric.append(element("strong", "", value));
    metric.append(element("small", "", label));
    summary.append(metric);
  });
  skillTrialResult.append(summary);

  const log = element("div", "experiment-log");
  learning.experiments.forEach((experiment, index) => {
    const row = element("div");
    row.append(
      element(
        "span",
        "",
        `test ${index + 1}: ${JSON.stringify(experiment.input)} → ${JSON.stringify(experiment.observation)}`,
      ),
    );
    row.append(
      element(
        "span",
        "",
        `${experiment.hypotheses_before} → ${experiment.hypotheses_after}`,
      ),
    );
    log.append(row);
  });
  skillTrialResult.append(log);
  skillTrialResult.append(
    element(
      "div",
      "feedback-result",
      payload.transfer.passed
        ? `Consolidated and transferred: ${JSON.stringify(payload.transfer.input)} → ${JSON.stringify(payload.transfer.output)}`
        : "The candidate was not consolidated because verification failed.",
    ),
  );
}

function renderState(state) {
  document.querySelector("#backend-badge").textContent = state.backend;
  document.querySelector("#experience-count").textContent =
    `${state.total_experiences} experience${state.total_experiences === 1 ? "" : "s"}`;
  renderStrategies(state.strategies || []);
  renderMemories(state.episodes || []);
  renderFacts(state.beliefs || []);
  renderSkills(state.skills || []);
  renderToolSkills(state.tool_skills || []);
  renderPlasticSkills(state.plastic_skills || []);
}

function updateToolGoalFields() {
  const kind = toolLearningForm.querySelector('[name="kind"]').value;
  const field = toolLearningForm.querySelector(".tool-value-field");
  const input = field.querySelector("input");
  field.hidden = kind === "delete";
  input.disabled = kind === "delete";
  input.required = kind !== "delete";
}

toolLearningForm.querySelector('[name="kind"]').addEventListener("change", updateToolGoalFields);
updateToolGoalFields();

toolLearningForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const submit = toolLearningForm.querySelector("button");
  submit.disabled = true;
  submit.textContent = "Learning and verifying…";
  const data = new FormData(toolLearningForm);
  const now = Date.now();
  try {
    const payload = await api("/api/tools/learn", {
      method: "POST",
      body: JSON.stringify({
        name: `${data.get("kind")}-workflow-${now}`,
        kind: data.get("kind"),
        key: data.get("key"),
        value: data.get("value"),
        seed: now % 1000000,
      }),
    });
    renderToolLearning(payload);
    renderState(payload.state);
  } catch (error) {
    showError(error);
  } finally {
    submit.disabled = false;
    submit.textContent = "Learn another workflow";
  }
});

plasticityLearningForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const submit = plasticityLearningForm.querySelector("button");
  const data = new FormData(plasticityLearningForm);
  const now = Date.now();
  submit.disabled = true;
  submit.textContent = "Training and protecting…";
  try {
    const rule = String(data.get("rule"));
    const payload = await api("/api/plasticity/learn", {
      method: "POST",
      body: JSON.stringify({
        name: `${rule.replaceAll("_", "-")}-${now}`,
        rule,
        seed: now % 1000000,
      }),
    });
    renderPlasticityLearning(payload);
    renderState(payload.state);
  } catch (error) {
    showError(error);
  } finally {
    submit.disabled = false;
    submit.textContent = "Train another neural expert";
  }
});

plasticityRunForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  try {
    const data = new FormData(plasticityRunForm);
    const input = JSON.parse(String(data.get("input")));
    if (!Array.isArray(input)) throw new Error("Input must be a JSON array");
    const payload = await api("/api/plasticity/run", {
      method: "POST",
      body: JSON.stringify({ name: data.get("name"), input }),
    });
    plasticityRunOutput.textContent = `${JSON.stringify(payload.input)} → ${payload.prediction} (${percent(payload.probability)} confidence)`;
    plasticityRunOutput.hidden = false;
  } catch (error) {
    showError(error);
  }
});

discoverSkillButton.addEventListener("click", async () => {
  clearError();
  discoverSkillButton.disabled = true;
  discoverSkillButton.textContent = "Experimenting…";
  try {
    const now = Date.now();
    const payload = await api("/api/skills/discover", {
      method: "POST",
      body: JSON.stringify({
        name: `discovered-skill-${now}`,
        seed: now % 1000000,
      }),
    });
    renderSkillTrial(payload);
    renderState(payload.state);
  } catch (error) {
    showError(error);
  } finally {
    discoverSkillButton.disabled = false;
    discoverSkillButton.textContent = "Run another trial";
  }
});

skillRunForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  try {
    const data = new FormData(skillRunForm);
    const input = JSON.parse(String(data.get("input")));
    if (!Array.isArray(input)) throw new Error("Input must be a JSON array");
    const payload = await api("/api/skills/run", {
      method: "POST",
      body: JSON.stringify({ name: data.get("name"), input }),
    });
    skillRunOutput.textContent = `${JSON.stringify(payload.input)} → ${JSON.stringify(payload.output)}`;
    skillRunOutput.hidden = false;
  } catch (error) {
    showError(error);
  }
});

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
