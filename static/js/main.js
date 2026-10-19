// ---------------- FLOATING PARTICLE BACKGROUND ----------------
(function particles(){
  const canvas = document.getElementById("bg-canvas");
  if(!canvas) return;
  const ctx = canvas.getContext("2d");
  let W, H, particles;
  function resize(){
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  function init(){
    resize();
    particles = Array.from({length: 46}, () => ({
      x: Math.random() * W, y: Math.random() * H,
      r: Math.random() * 1.8 + 0.6,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      hue: Math.random() > 0.5 ? "34,211,238" : "129,140,248",
      a: Math.random() * 0.5 + 0.15,
    }));
  }
  function tick(){
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if(p.x < 0 || p.x > W) p.vx *= -1;
      if(p.y < 0 || p.y > H) p.vy *= -1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.hue},${p.a})`;
      ctx.fill();
    });
    requestAnimationFrame(tick);
  }
  window.addEventListener("resize", resize);
  init(); tick();
})();

// ---------------- COUNT-UP NUMBER ANIMATION ----------------
function animateNumber(el, target, decimals = 0, suffix = ""){
  const start = 0;
  const duration = 900;
  const t0 = performance.now();
  function frame(now){
    const p = Math.min((now - t0) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = start + (target - start) * eased;
    el.textContent = val.toFixed(decimals) + suffix;
    if(p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

const RISK_COLORS = {
  "Critical": "#ef4444",
  "High Risk": "#f97316",
  "Low Risk": "#eab308",
  "Good": "#22c55e",
  "Excellent": "#06b6d4",
};

const PAGE_TITLES = {
  health: ["Battery Health Prediction", "Live SOC / SOH / RUL estimation from impedance-spectroscopy parameters"],
  accuracy: ["Accuracy", "Model accuracy comparison across all trained algorithms"],
  f1: ["F1 Score", "Harmonic mean of precision and recall per algorithm"],
  confusion: ["Confusion Matrix", "Class-by-class prediction breakdown"],
  prf: ["Precision & Recall", "Weighted precision and recall per algorithm"],
  epochs: ["Epochs / Loss Curve", "Training vs validation loss for the deep models"],
  compare: ["Algorithm Comparison", "Accuracy, precision, recall and F1 side by side"],
  roc: ["ROC-AUC Curve", "One-vs-rest ROC curves per risk class"],
  heatmap: ["Correlation Heatmap", "Feature correlation with SOH / SOC / RUL"],
  report: ["Download Report", "Export a complete PDF summary of this project"],
};

let METRICS = null;
let charts = {};

function destroyChart(id){
  if(charts[id]){ charts[id].destroy(); delete charts[id]; }
}

// ---------------- NAV ----------------
document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const page = btn.dataset.page;
    document.querySelectorAll(".page").forEach(p => p.classList.add("d-none"));
    document.getElementById(`page-${page}`).classList.remove("d-none");
    document.getElementById("page-title").textContent = PAGE_TITLES[page][0];
    document.getElementById("page-sub").textContent = PAGE_TITLES[page][1];
    renderPage(page);
  });
});

function renderPage(page){
  if(page === "accuracy") renderBarMetric("accChart", "accuracy", "Accuracy");
  if(page === "f1") renderBarMetric("f1Chart", "f1_score", "F1 Score");
  if(page === "prf") renderPRF();
  if(page === "confusion") renderSelectImage("cm-select", "cm-img", "confusion_matrix");
  if(page === "roc") renderSelectImage("roc-select", "roc-img", "roc_auc");
}

// ---------------- SUMMARY + RISK DIST (health page) ----------------
async function loadSummary(){
  try{
    const res = await fetch("/api/summary");
    const data = await res.json();
    if(data.error) return;
    document.getElementById("summary-stats").innerHTML = `
      <div class="stat-chip"><b id="stat-n">0</b>samples</div>
      <div class="stat-chip"><b id="stat-soh">0%</b>avg SOH</div>
      <div class="stat-chip"><b id="stat-soc">0%</b>avg SOC</div>
      <div class="stat-chip"><b id="stat-rul">0</b>avg RUL (cycles)</div>
    `;
    animateNumber(document.getElementById("stat-n"), data.n_samples, 0);
    animateNumber(document.getElementById("stat-soh"), data.avg_soh, 1, "%");
    animateNumber(document.getElementById("stat-soc"), data.avg_soc, 1, "%");
    animateNumber(document.getElementById("stat-rul"), data.avg_rul, 1);
    const labels = Object.keys(data.risk_distribution);
    const values = Object.values(data.risk_distribution);
    destroyChart("riskDistChart");
    charts["riskDistChart"] = new Chart(document.getElementById("riskDistChart"), {
      type: "bar",
      data: { labels, datasets: [{ label: "Samples", data: values,
        backgroundColor: labels.map(l => RISK_COLORS[l] || "#6366f1"),
        borderRadius: 10, borderSkipped: false, maxBarThickness: 56 }] },
      options: {
        animation: { duration: 900, easing: "easeOutCubic" },
        plugins:{legend:{display:false}},
        scales:{
          x:{ ticks:{color:"#8b92ab"}, grid:{display:false} },
          y:{ ticks:{color:"#8b92ab"}, grid:{color:"rgba(255,255,255,.06)"} }
        }
      }
    });
  }catch(e){ console.error(e); }
}

// ---------------- PREDICTION FORM ----------------
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  const res = await fetch("/api/predict", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if(data.error){ alert(data.error); return; }

  document.getElementById("result-empty").classList.add("d-none");
  document.getElementById("result-body").classList.remove("d-none");

  const color = RISK_COLORS[data.risk_category] || "#22d3ee";

  const badge = document.getElementById("risk-badge");
  badge.textContent = data.risk_category;
  badge.style.background = color;
  badge.style.color = color;
  badge.style.setProperty("--shadow-color", color);
  badge.style.color = "#04121a";
  badge.style.boxShadow = `0 0 24px ${color}`;

  // circular gauge: circumference = 2*pi*92 ≈ 578
  const circumference = 578;
  const fill = document.getElementById("gauge-fill");
  const offset = circumference - (data.soh / 100) * circumference;
  fill.style.stroke = color;
  fill.style.color = color;
  requestAnimationFrame(() => { fill.style.strokeDashoffset = offset; });
  animateNumber(document.getElementById("gauge-soh-value"), data.soh, 1, "%");

  animateNumber(document.getElementById("val-soh"), data.soh, 1, "%");
  animateNumber(document.getElementById("val-soc"), data.soc, 1, "%");
  animateNumber(document.getElementById("val-rul"), data.rul_cycles, 0, "");

  document.getElementById("risk-marker").style.marginLeft = `calc(${data.soh}% - 1px)`;

  const list = document.getElementById("precaution-list");
  list.innerHTML = "";
  data.precautions.forEach(p => {
    const li = document.createElement("li");
    li.textContent = p;
    list.appendChild(li);
  });
});

// ---------------- METRICS CHARTS ----------------
async function loadMetrics(){
  const res = await fetch("/api/metrics");
  METRICS = await res.json();
}

function renderBarMetric(canvasId, key, label){
  if(!METRICS) return;
  const algos = Object.keys(METRICS.results);
  const values = algos.map(a => METRICS.results[a][key] ?? 0);
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext("2d");
  const grad = ctx.createLinearGradient(0, 0, 0, 300);
  grad.addColorStop(0, "#22d3ee");
  grad.addColorStop(1, "#818cf8");
  charts[canvasId] = new Chart(ctx, {
    type: "bar",
    data: { labels: algos, datasets: [{ label, data: values, backgroundColor: grad,
      borderRadius: 10, borderSkipped: false, maxBarThickness: 52 }] },
    options: {
      animation: { duration: 900, easing: "easeOutCubic" },
      scales: {
        y:{ min:0, max:1, ticks:{color:"#8b92ab"}, grid:{color:"rgba(255,255,255,.06)"} },
        x:{ ticks:{color:"#8b92ab"}, grid:{display:false} }
      },
      plugins:{legend:{display:false}}
    }
  });
}

function renderPRF(){
  if(!METRICS) return;
  const algos = Object.keys(METRICS.results);
  destroyChart("prfChart");
  charts["prfChart"] = new Chart(document.getElementById("prfChart"), {
    type: "bar",
    data: {
      labels: algos,
      datasets: [
        { label: "Precision", data: algos.map(a => METRICS.results[a].precision ?? 0), backgroundColor: "#818cf8", borderRadius: 6 },
        { label: "Recall", data: algos.map(a => METRICS.results[a].recall ?? 0), backgroundColor: "#22d3ee", borderRadius: 6 },
      ]
    },
    options: { scales: { y:{ min:0, max:1, ticks:{color:"#8b92ab"} }, x:{ticks:{color:"#8b92ab"}} },
               plugins:{legend:{labels:{color:"#e6e9f5"}}} }
  });
}

function renderSelectImage(selectId, imgId, prefix){
  if(!METRICS) return;
  const sel = document.getElementById(selectId);
  if(sel.options.length === 0){
    Object.keys(METRICS.results).forEach(a => {
      const opt = document.createElement("option");
      opt.value = a; opt.textContent = a;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => updateImg(imgId, prefix, sel.value));
  }
  if(sel.value) updateImg(imgId, prefix, sel.value);
  else if(sel.options.length){ sel.selectedIndex = 0; updateImg(imgId, prefix, sel.value); }
}

function updateImg(imgId, prefix, algo){
  document.getElementById(imgId).src = `/static/img/${prefix}_${algo}.png?ts=${Date.now()}`;
}

document.getElementById("epoch-select")?.addEventListener("change", (e) => {
  document.getElementById("epoch-img").src = `/static/img/training_curve_${e.target.value}.png?ts=${Date.now()}`;
});

// ---------------- INIT ----------------
(async function init(){
  await loadSummary();
  await loadMetrics();
  document.getElementById("epoch-img").src = "/static/img/training_curve_hybrid.png";
})();
