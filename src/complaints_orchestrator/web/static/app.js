const scenarioSelect = document.getElementById("scenario-select");
const caseForm = document.getElementById("case-form");
const statusText = document.getElementById("status-text");
const errorText = document.getElementById("error-text");

const caseIdInput = document.getElementById("case-id");
const customerIdInput = document.getElementById("customer-id");
const orderIdInput = document.getElementById("order-id");
const emailSubjectInput = document.getElementById("email-subject");
const emailBodyInput = document.getElementById("email-body");

const summaryContent = document.getElementById("summary-content");
const decisionContent = document.getElementById("decision-content");
const actionsList = document.getElementById("actions-list");
const emailContent = document.getElementById("email-content");
const securityContent = document.getElementById("security-content");

const triageJson = document.getElementById("triage-json");
const contextJson = document.getElementById("context-json");
const resolutionJson = document.getElementById("resolution-json");
const finalizeJson = document.getElementById("finalize-json");

const scenariosById = new Map();

function setStatus(text) {
  statusText.textContent = text;
}

function clearError() {
  errorText.classList.add("hidden");
  errorText.textContent = "";
}

function showError(text) {
  errorText.textContent = text;
  errorText.classList.remove("hidden");
}

function formatJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function resetEmptyState(target, emptyText) {
  target.classList.add("empty");
  target.textContent = emptyText;
}

function removeEmptyState(target) {
  target.classList.remove("empty");
}

function renderSummary(payload) {
  const triage = payload.triage || {};
  const resolution = payload.resolution || {};
  const finalize = payload.finalize || {};

  summaryContent.innerHTML = "";
  removeEmptyState(summaryContent);

  const metrics = [
    ["Case ID", payload.case_id],
    ["Complaint Type", triage.complaint_type || "N/A"],
    ["Urgency", triage.urgency || "N/A"],
    ["Language", triage.response_language || "N/A"],
    ["Decision", resolution.decision || "N/A"],
    ["Status", finalize.status || "N/A"],
    ["Runtime", `${payload.runtime_ms} ms`],
  ];

  for (const [label, value] of metrics) {
    const item = document.createElement("div");
    item.className = "metric";

    const itemLabel = document.createElement("span");
    itemLabel.className = "metric-label";
    itemLabel.textContent = label;

    const itemValue = document.createElement("strong");
    itemValue.className = "metric-value";
    itemValue.textContent = value;

    item.appendChild(itemLabel);
    item.appendChild(itemValue);
    summaryContent.appendChild(item);
  }
}

function renderDecision(payload) {
  const resolution = payload.resolution || {};
  const lines = [
    `Decision: ${resolution.decision || "N/A"}`,
    `Rationale: ${resolution.rationale || "N/A"}`,
    `HITL Required: ${resolution.hitl_required === true ? "Yes" : "No"}`,
    `HITL Reason: ${resolution.hitl_reason || "-"}`,
  ];
  removeEmptyState(decisionContent);
  decisionContent.textContent = lines.join("\n");
}

function renderActions(payload) {
  const actions = (payload.resolution && payload.resolution.tool_actions) || [];
  actionsList.innerHTML = "";

  if (!actions.length) {
    actionsList.classList.add("empty");
    const item = document.createElement("li");
    item.textContent = "No actions were executed.";
    actionsList.appendChild(item);
    return;
  }

  removeEmptyState(actionsList);
  for (const action of actions) {
    const item = document.createElement("li");
    item.textContent = `${action.tool_name} | ${action.status} | ref=${action.reference_id}`;
    actionsList.appendChild(item);
  }
}

function renderEmail(payload) {
  const resolution = payload.resolution || {};
  const subject = resolution.response_subject || "";
  const body = String(resolution.response_body || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\r\n/g, "\n");
  removeEmptyState(emailContent);
  emailContent.textContent = `Subject: ${subject}\n\n${body}`;
}

function renderSecurity(payload) {
  const events = payload.security_events || [];
  const guard = payload.output_guard_passed ? "PASS" : "FAIL";
  const body = events.length ? events.join("\n") : "No security events.";
  removeEmptyState(securityContent);
  securityContent.textContent = `Output Guard: ${guard}\n\n${body}`;
}

function renderJsonBlocks(payload) {
  triageJson.textContent = formatJson(payload.triage);
  contextJson.textContent = formatJson(payload.context);
  resolutionJson.textContent = formatJson(payload.resolution);
  finalizeJson.textContent = formatJson(payload.finalize);
}

function fillFromScenario(scenario) {
  customerIdInput.value = scenario.customer_id || "";
  orderIdInput.value = scenario.order_id || "";
  emailSubjectInput.value = scenario.email_subject || "";
  emailBodyInput.value = scenario.email_body || "";
}

async function fetchScenarios() {
  const response = await fetch("/api/scenarios");
  if (!response.ok) {
    throw new Error(`Could not load scenarios (${response.status}).`);
  }
  const scenarios = await response.json();
  for (const scenario of scenarios) {
    scenariosById.set(scenario.id, scenario);
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = `${scenario.title} (${scenario.id})`;
    scenarioSelect.appendChild(option);
  }
}

async function runCase(payload) {
  const response = await fetch("/api/cases/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const rawBody = await response.text();
  let body = {};
  if (rawBody) {
    try {
      body = JSON.parse(rawBody);
    } catch (error) {
      body = { detail: rawBody };
    }
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `Request failed (${response.status}).`;
    throw new Error(detail);
  }
  return body;
}

scenarioSelect.addEventListener("change", () => {
  const selectedId = scenarioSelect.value;
  if (!selectedId) {
    return;
  }
  const scenario = scenariosById.get(selectedId);
  if (scenario) {
    fillFromScenario(scenario);
  }
});

caseForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  setStatus("Running workflow...");

  const payload = {
    case_id: caseIdInput.value.trim() || null,
    customer_id: customerIdInput.value.trim(),
    order_id: orderIdInput.value.trim(),
    email_subject: emailSubjectInput.value.trim(),
    email_body: emailBodyInput.value.trim(),
    channel: "EMAIL",
  };

  try {
    const result = await runCase(payload);
    renderSummary(result);
    renderDecision(result);
    renderActions(result);
    renderEmail(result);
    renderSecurity(result);
    renderJsonBlocks(result);
    setStatus("Completed");
  } catch (error) {
    showError(error.message);
    setStatus("Failed");
  }
});

(async function bootstrap() {
  setStatus("Loading scenarios...");
  try {
    await fetchScenarios();
    setStatus("Idle");
  } catch (error) {
    showError(error.message);
    setStatus("Idle (scenario load issue)");
  }
})();
