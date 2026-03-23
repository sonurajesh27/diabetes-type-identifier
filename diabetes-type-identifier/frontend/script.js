/**
 * script.js - AI Diabetes Type Identifier frontend logic
 */

const API_BASE = "http://localhost:5000";

// ── Chart instances ──
let confidenceChart = null;
let importanceChart = null;

// ── DOM refs ──
const loader       = document.getElementById("loader");
const resultPanel  = document.getElementById("result-panel");
const batchPanel   = document.getElementById("batch-result-panel");
const resultLabel  = document.getElementById("result-label");
const resultConf   = document.getElementById("result-confidence");
const trainResult  = document.getElementById("train-result");

// ── Utility ──
const show = el => el.classList.remove("hidden");
const hide = el => el.classList.add("hidden");

function showLoader() { show(loader); }
function hideLoader() { hide(loader); }

async function apiFetch(path, options = {}) {
  const res = await fetch(API_BASE + path, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

// ── Health check ──
async function checkHealth() {
  try {
    await apiFetch("/health");
  } catch {}
}

// ── Tabs ──
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(s => s.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ── Manual form predict ──
document.getElementById("manual-form").addEventListener("submit", async e => {
  e.preventDefault();
  const form = e.target;
  const fields = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                  "Insulin","BMI","DiabetesPedigreeFunction","Age"];
  const payload = {};
  fields.forEach(f => {
    const val = form.elements[f]?.value;
    if (val !== "" && val !== undefined) payload[f] = parseFloat(val);
  });

  showLoader();
  hide(resultPanel);
  hide(batchPanel);

  try {
    const data = await apiFetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    renderSingleResult(data);
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    hideLoader();
  }
});

// ── CSV file select ──
let csvFile = null;
const csvInput    = document.getElementById("csv-file-input");
const csvFilename = document.getElementById("csv-filename");
const csvDropZone = document.getElementById("csv-drop-zone");
const csvPredBtn  = document.getElementById("csv-predict-btn");

csvInput.addEventListener("change", () => {
  csvFile = csvInput.files[0];
  csvFilename.textContent = csvFile ? csvFile.name : "";
  csvPredBtn.disabled = !csvFile;
});

setupDrop(csvDropZone, file => {
  csvFile = file;
  csvFilename.textContent = file.name;
  csvPredBtn.disabled = false;
});

csvPredBtn.addEventListener("click", async () => {
  if (!csvFile) return;
  const fd = new FormData();
  fd.append("file", csvFile);

  showLoader();
  hide(resultPanel);
  hide(batchPanel);

  try {
    const data = await apiFetch("/predict", { method: "POST", body: fd });
    renderBatchResult(data.predictions);
  } catch (err) {
    alert("Batch prediction failed: " + err.message);
  } finally {
    hideLoader();
  }
});

// ── Train ──
let trainFile = null;
const trainInput    = document.getElementById("train-file-input");
const trainFilename = document.getElementById("train-filename");
const trainDropZone = document.getElementById("train-drop-zone");
const trainBtn      = document.getElementById("train-btn");

trainInput.addEventListener("change", () => {
  trainFile = trainInput.files[0];
  trainFilename.textContent = trainFile ? trainFile.name : "";
});

setupDrop(trainDropZone, file => {
  trainFile = file;
  trainFilename.textContent = file.name;
});

trainBtn.addEventListener("click", async () => {
  const modelType = document.getElementById("model-type").value;
  const fd = new FormData();
  fd.append("model_type", modelType);
  if (trainFile) fd.append("file", trainFile);

  showLoader();
  trainResult.innerHTML = "";

  try {
    const data = await apiFetch("/train", { method: "POST", body: fd });
    const m = data.metrics;
    trainResult.innerHTML = `
      <p style="color:var(--success);font-weight:600;margin-bottom:.75rem">✅ Model trained successfully!</p>
      <pre>Model Type     : ${m.model_type}
Training Samples: ${m.training_samples}
Weighted F1    : ${m.f1_score}
Confusion Matrix: ${JSON.stringify(m.confusion_matrix)}</pre>`;
    checkHealth();
  } catch (err) {
    trainResult.innerHTML = `<p style="color:var(--danger)">❌ Training failed: ${err.message}</p>`;
  } finally {
    hideLoader();
  }
});

// ── Render single result ──
function renderSingleResult(data) {
  const label = data.predicted_class;
  const cls   = labelClass(label);

  resultLabel.textContent = label;
  resultLabel.className   = "result-label " + cls;
  resultConf.innerHTML    = `Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong>`;

  renderConfidenceChart(data.confidence_scores);
  renderImportanceChart(data.feature_importance);

  show(resultPanel);
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Render batch result ──
function renderBatchResult(predictions) {
  const wrapper = document.getElementById("batch-table-wrapper");
  const rows = predictions.map(p => {
    const cls = labelClass(p.predicted_class);
    const scores = Object.entries(p.confidence_scores)
      .map(([k, v]) => `${k}: ${(v*100).toFixed(1)}%`).join(" | ");
    return `<tr>
      <td>${p.row}</td>
      <td><span class="pill ${cls}">${p.predicted_class}</span></td>
      <td>${(p.confidence * 100).toFixed(1)}%</td>
      <td style="font-size:.8rem;color:var(--muted)">${scores}</td>
    </tr>`;
  }).join("");

  wrapper.innerHTML = `
    <table>
      <thead><tr><th>#</th><th>Prediction</th><th>Confidence</th><th>All Scores</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  show(batchPanel);
  batchPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Charts ──
function renderConfidenceChart(scores) {
  const ctx = document.getElementById("confidence-chart").getContext("2d");
  if (confidenceChart) confidenceChart.destroy();

  const labels = Object.keys(scores);
  const values = Object.values(scores).map(v => +(v * 100).toFixed(1));
  const colors = labels.map(l => labelColor(l));

  confidenceChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderWidth: 2 }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.toFixed(1)}%` } }
      }
    }
  });
}

function renderImportanceChart(importance) {
  if (!importance || !Object.keys(importance).length) return;
  const ctx = document.getElementById("importance-chart").getContext("2d");
  if (importanceChart) importanceChart.destroy();

  const labels = Object.keys(importance);
  const values = Object.values(importance).map(v => +(v * 100).toFixed(2));

  importanceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Importance (%)",
        data: values,
        backgroundColor: "#3b82f6",
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } }
    }
  });
}

// ── Export PDF ──
document.getElementById("export-pdf-btn").addEventListener("click", () => {
  window.print();
});

// ── Clear ──
document.getElementById("clear-btn").addEventListener("click", () => {
  hide(resultPanel);
  hide(batchPanel);
  if (confidenceChart) { confidenceChart.destroy(); confidenceChart = null; }
  if (importanceChart) { importanceChart.destroy(); importanceChart = null; }
  document.getElementById("manual-form").reset();
});

// ── Drag & drop helper ──
function setupDrop(zone, onFile) {
  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".csv")) onFile(file);
    else alert("Please drop a .csv file.");
  });
}

// ── Label helpers ──
function labelClass(label) {
  if (label === "Type 1 Diabetes") return "type1";
  if (label === "Type 2 Diabetes") return "type2";
  return "none";
}

function labelColor(label) {
  if (label === "Type 1 Diabetes") return "#7c3aed";
  if (label === "Type 2 Diabetes") return "#dc2626";
  return "#16a34a";
}

// ── Init ──
checkHealth();
