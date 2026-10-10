/**
 * Urban Traffic Flow Optimization - Application Logic
 * Interactive dashboard tabs, dynamic figure gallery, live prediction engine,
 * and Vercel serverless integration.
 */

// Comprehensive Figure Catalog with Insights
const FIGURES_CATALOG = [
  {
    id: "fig-hour",
    file: "figures/average_traffic_by_hour.png",
    title: "Diurnal Traffic Flow by Hour",
    category: "temporal",
    pillClass: "pill-temporal",
    catLabel: "Temporal Pattern",
    insight: "Bimodal peak commuter profile: sharp surges at 07:00–08:00 AM (~4,950 veh/hr) and 16:00–17:00 PM (~5,400 veh/hr). Overnight low (< 600 veh/hr) between 01:00 and 03:00 AM.",
  },
  {
    id: "fig-day",
    file: "figures/average_traffic_by_day.png",
    title: "Average Traffic Volume by Day of Week",
    category: "temporal",
    pillClass: "pill-temporal",
    catLabel: "Temporal Pattern",
    insight: "Midweek commuter crest: Tuesday through Thursday maintain the highest average volume (~3,600 veh/hr), with Sunday displaying the lowest highway load (~2,500 veh/hr).",
  },
  {
    id: "fig-weekend",
    file: "figures/weekday_vs_weekend.png",
    title: "Weekday Commuter vs. Weekend Distribution",
    category: "temporal",
    pillClass: "pill-temporal",
    catLabel: "Temporal Pattern",
    insight: "Distinct diurnal regime difference: Weekdays exhibit sharp twin peaks for rush hours, whereas weekends present a smooth midday plateau between 12:00 PM and 16:00 PM.",
  },
  {
    id: "fig-month",
    file: "figures/monthly_traffic.png",
    title: "Monthly & Seasonal Traffic Cycles",
    category: "temporal",
    pillClass: "pill-temporal",
    catLabel: "Temporal Pattern",
    insight: "Summer transit crest: June through August record elevated highway flow (~3,450 veh/hr), while sub-zero January and February weather suppresses non-essential travel.",
  },
  {
    id: "fig-timeline",
    file: "figures/traffic_over_time.png",
    title: "Longitudinal Traffic Timeline (2012–2018)",
    category: "temporal",
    pillClass: "pill-temporal",
    catLabel: "Temporal Pattern",
    insight: "Full multi-year longitudinal record of 48,204 sensor observations on I-94 Eastbound illustrating annual growth and seasonal consistency over 6 years.",
  },
  {
    id: "fig-temp",
    file: "figures/temperature_vs_traffic.png",
    title: "Ambient Temperature vs. Traffic Density",
    category: "weather",
    pillClass: "pill-weather",
    catLabel: "Weather Impact",
    insight: "Temperature in Celsius: volume stabilizes across temperate ranges (10°C to 25°C), while severe freezing sub-zero conditions slightly lower peak traffic speeds and density.",
  },
  {
    id: "fig-weather",
    file: "figures/weather_traffic.png",
    title: "Traffic Volume by Weather Condition",
    category: "weather",
    pillClass: "pill-weather",
    catLabel: "Weather Impact",
    insight: "Adverse weather impact: Clear and Clouds record standard flow, whereas intense Snow, Rainstorms, and Squalls cause systematic velocity drops and lane throughput suppression.",
  },
  {
    id: "fig-rain",
    file: "figures/rainfall_vs_traffic.png",
    title: "Hourly Rainfall (mm) vs. Traffic Flow",
    category: "weather",
    pillClass: "pill-weather",
    catLabel: "Weather Impact",
    insight: "Precipitation effect: heavy rainfall above 10mm/h correlates with increased headway distance and reduced hourly vehicle throughput.",
  },
  {
    id: "fig-corr",
    file: "figures/correlation_heatmap.png",
    title: "Feature Correlation Matrix",
    category: "model",
    pillClass: "pill-model",
    catLabel: "Model Diagnostics",
    insight: "Inter-feature collinearity check: Hour and cyclical trigonometric components show the strongest correlation with volume (r ≈ 0.35–0.42), while raw weather metrics show low linear collinearity.",
  },
  {
    id: "fig-pred",
    file: "figures/actual_vs_predicted.png",
    title: "Actual vs. Predicted Traffic (Test Set)",
    category: "model",
    pillClass: "pill-model",
    catLabel: "Model Diagnostics",
    insight: "Champion Random Forest out-of-sample fit: closely tracks the true bimodal diurnal cycles across the 9,638 held-out test records (R² = 0.9266, MAE = 300.7 veh/hr).",
  },
  {
    id: "fig-resid",
    file: "figures/prediction_residuals.png",
    title: "Residual Error Distribution & Diagnostics",
    category: "model",
    pillClass: "pill-model",
    catLabel: "Model Diagnostics",
    insight: "Error homoscedasticity: residuals are centered tightly around zero with a near-Gaussian distribution, verifying model unbiasedness and absence of systematic drift.",
  },
  {
    id: "fig-feat",
    file: "figures/feature_importance.png",
    title: "Random Forest Feature Importances",
    category: "model",
    pillClass: "pill-model",
    catLabel: "Model Diagnostics",
    insight: "Feature attribution: Hour of day accounts for over 52% of total split variance, followed by cyclical trigonometric time representations (18.6%) and day-of-week indicators (11.4%).",
  },
];

// Presets Definition
const SCENARIO_PRESETS = {
  "morning-rush": {
    datetime: "2018-10-15T07:30",
    weather: "Clear",
    temp: 12.0,
    rain: 0.0,
    snow: 0.0,
    clouds: 15,
    holiday: false,
  },
  "evening-peak": {
    datetime: "2018-10-16T17:00",
    weather: "Clouds",
    temp: 16.0,
    rain: 0.0,
    snow: 0.0,
    clouds: 40,
    holiday: false,
  },
  "night-quiet": {
    datetime: "2018-10-17T02:30",
    weather: "Clear",
    temp: 6.0,
    rain: 0.0,
    snow: 0.0,
    clouds: 5,
    holiday: false,
  },
  "weekend-midday": {
    datetime: "2018-10-20T13:30",
    weather: "Clear",
    temp: 21.0,
    rain: 0.0,
    snow: 0.0,
    clouds: 20,
    holiday: false,
  },
  "heavy-rain": {
    datetime: "2018-10-18T17:30",
    weather: "Rain",
    temp: 11.0,
    rain: 14.0,
    snow: 0.0,
    clouds: 95,
    holiday: false,
  },
  "snow-storm": {
    datetime: "2018-11-20T08:00",
    weather: "Snow",
    temp: -5.0,
    rain: 0.0,
    snow: 3.5,
    clouds: 100,
    holiday: false,
  },
};

// Application State Controller
const app = {
  activeTab: "overview",
  figures: FIGURES_CATALOG,
  activeFilter: "all",

  init() {
    this.bindNavigation();
    this.renderFigures();
    this.bindFigureFilters();
    this.bindPredictorForm();
    this.bindModal();
    this.updateDateTimeHint();

    // Run initial demo prediction
    this.runPrediction(false);
  },

  // Tab Switching
  switchTab(tabId) {
    this.activeTab = tabId;

    // Update Nav buttons
    document.querySelectorAll(".nav-link").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabId);
    });

    // Update Tab panes
    document.querySelectorAll(".tab-pane").forEach((pane) => {
      pane.classList.toggle("active", pane.id === `tab-${tabId}`);
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
  },

  bindNavigation() {
    document.querySelectorAll(".nav-link").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.switchTab(btn.dataset.tab);
      });
    });

    document.getElementById("brand-logo")?.addEventListener("click", () => {
      this.switchTab("overview");
    });

    document.getElementById("cta-quick-predict")?.addEventListener("click", () => {
      this.switchTab("predictor");
    });
  },

  // Render Figures Gallery
  renderFigures() {
    const container = document.getElementById("figures-container");
    if (!container) return;

    const filtered = this.activeFilter === "all"
      ? this.figures
      : this.figures.filter((f) => f.category === this.activeFilter);

    container.innerHTML = filtered
      .map(
        (fig) => `
      <div class="figure-card" data-category="${fig.category}" id="${fig.id}">
        <div class="figure-img-wrap" onclick="app.openFigureModal('${fig.id}')">
          <img src="${fig.file}" alt="${fig.title}" class="figure-img" loading="lazy">
          <div class="figure-overlay">
            <span class="zoom-badge">🔍 Click to Expand</span>
          </div>
        </div>
        <div class="figure-info">
          <span class="figure-cat-pill ${fig.pillClass}">${fig.catLabel}</span>
          <h3 class="figure-title">${fig.title}</h3>
          <p class="figure-desc">${fig.insight}</p>
          <button class="btn btn-outline btn-sm" onclick="app.openFigureModal('${fig.id}')">
            Enlarge & Inspect →
          </button>
        </div>
      </div>
    `
      )
      .join("");
  },

  bindFigureFilters() {
    document.querySelectorAll("#figure-filter-group .filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#figure-filter-group .filter-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.activeFilter = btn.dataset.filter;
        this.renderFigures();
      });
    });
  },

  openFigureModal(figId) {
    const fig = this.figures.find((f) => f.id === figId);
    if (!fig) return;

    const modal = document.getElementById("figure-modal");
    const modalImg = document.getElementById("modal-fig-img");
    const modalTitle = document.getElementById("modal-fig-title");
    const modalCaption = document.getElementById("modal-fig-caption");

    modalTitle.textContent = fig.title;
    modalImg.src = fig.file;
    modalImg.alt = fig.title;
    modalCaption.innerHTML = `<strong>Corridor Insight:</strong> ${fig.insight}`;

    modal.classList.add("active");
  },

  bindModal() {
    const modal = document.getElementById("figure-modal");
    const closeBtn = document.getElementById("modal-close-btn");

    const closeModal = () => modal.classList.remove("active");

    closeBtn?.addEventListener("click", closeModal);
    modal?.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modal.classList.contains("active")) {
        closeModal();
      }
    });
  },

  // Predictor Controls
  bindPredictorForm() {
    const tempInput = document.getElementById("input-temp");
    const tempDisplay = document.getElementById("temp-display");
    const rainInput = document.getElementById("input-rain");
    const rainDisplay = document.getElementById("rain-display");
    const snowInput = document.getElementById("input-snow");
    const snowDisplay = document.getElementById("snow-display");
    const cloudsInput = document.getElementById("input-clouds");
    const cloudsDisplay = document.getElementById("clouds-display");
    const dtInput = document.getElementById("input-datetime");

    tempInput?.addEventListener("input", (e) => {
      tempDisplay.textContent = `${parseFloat(e.target.value).toFixed(1)}°C`;
    });

    rainInput?.addEventListener("input", (e) => {
      rainDisplay.textContent = `${parseFloat(e.target.value).toFixed(1)} mm`;
    });

    snowInput?.addEventListener("input", (e) => {
      snowDisplay.textContent = `${parseFloat(e.target.value).toFixed(1)} mm`;
    });

    cloudsInput?.addEventListener("input", (e) => {
      cloudsDisplay.textContent = `${e.target.value}%`;
    });

    dtInput?.addEventListener("change", () => {
      this.updateDateTimeHint();
    });

    // Preset Buttons
    document.querySelectorAll(".preset-pill").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        document.querySelectorAll(".preset-pill").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        this.applyPreset(btn.dataset.preset);
      });
    });

    // Form Submit
    const form = document.getElementById("prediction-form");
    form?.addEventListener("submit", (e) => {
      e.preventDefault();
      this.runPrediction(true);
    });
  },

  applyPreset(presetKey) {
    const preset = SCENARIO_PRESETS[presetKey];
    if (!preset) return;

    document.getElementById("input-datetime").value = preset.datetime;
    document.getElementById("input-weather").value = preset.weather;

    const tempInput = document.getElementById("input-temp");
    tempInput.value = preset.temp;
    document.getElementById("temp-display").textContent = `${preset.temp.toFixed(1)}°C`;

    const rainInput = document.getElementById("input-rain");
    rainInput.value = preset.rain;
    document.getElementById("rain-display").textContent = `${preset.rain.toFixed(1)} mm`;

    const snowInput = document.getElementById("input-snow");
    snowInput.value = preset.snow;
    document.getElementById("snow-display").textContent = `${preset.snow.toFixed(1)} mm`;

    const cloudsInput = document.getElementById("input-clouds");
    cloudsInput.value = preset.clouds;
    document.getElementById("clouds-display").textContent = `${preset.clouds}%`;

    document.getElementById("input-holiday").checked = preset.holiday;

    this.updateDateTimeHint();
    this.runPrediction(true);
  },

  updateDateTimeHint() {
    const dtVal = document.getElementById("input-datetime")?.value;
    const hintEl = document.getElementById("datetime-hint");
    if (!dtVal || !hintEl) return;

    try {
      const dt = new Date(dtVal);
      const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
      const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const dayName = days[dt.getDay()];
      const hour = dt.getHours();
      const isWeekend = dt.getDay() === 0 || dt.getDay() === 6;

      let windowDesc = "Off-Peak";
      if (!isWeekend && ((hour >= 7 && hour <= 9) || (hour >= 16 && hour <= 18))) {
        windowDesc = hour <= 9 ? "Morning Commute Rush" : "Evening Commute Rush";
      } else if (isWeekend && hour >= 11 && hour <= 16) {
        windowDesc = "Weekend Midday Plateau";
      } else if (hour >= 1 && hour <= 4) {
        windowDesc = "Late Night Low Volume";
      }

      hintEl.textContent = `${dayName}, ${months[dt.getMonth()]} ${dt.getDate()}, ${dt.getFullYear()} (Hour ${hour}:00 — ${windowDesc})`;
    } catch {
      hintEl.textContent = "Selected Transit Timestamp";
    }
  },

  // Client-Side Calibrated Regression Engine (Fallback & Instant Zero-Cold-Start)
  calculateCalibratedInference(params) {
    const dt = new Date(params.date_time);
    const hour = isNaN(dt.getHours()) ? 17 : dt.getHours();
    const dayOfWeek = isNaN(dt.getDay()) ? 1 : (dt.getDay() + 6) % 7; // Monday = 0
    const month = isNaN(dt.getMonth()) ? 10 : dt.getMonth() + 1;
    const isWeekend = dayOfWeek >= 5 ? 1 : 0;

    const weekdayHourly = {
      0: 480, 1: 300, 2: 240, 3: 360, 4: 850, 5: 2750, 6: 5100, 7: 5400,
      8: 4850, 9: 4400, 10: 4300, 11: 4600, 12: 4850, 13: 4850, 14: 5150,
      15: 5500, 16: 5650, 17: 5400, 18: 4350, 19: 3300, 20: 2750, 21: 2450,
      22: 1950, 23: 1100,
    };
    const weekendHourly = {
      0: 950, 1: 650, 2: 500, 3: 380, 4: 410, 5: 680, 6: 1250, 7: 1700,
      8: 2400, 9: 3400, 10: 4100, 11: 4450, 12: 4650, 13: 4600, 14: 4550,
      15: 4500, 16: 4450, 17: 4300, 18: 3900, 19: 3300, 20: 2850, 21: 2550,
      22: 2150, 23: 1450,
    };

    let base = isWeekend ? weekendHourly[hour] || 2500 : weekdayHourly[hour] || 3200;

    // Monthly seasonality
    const monthModifiers = {
      1: 0.90, 2: 0.93, 3: 0.96, 4: 0.99, 5: 1.03, 6: 1.05,
      7: 1.04, 8: 1.05, 9: 1.03, 10: 1.04, 11: 0.98, 12: 0.92,
    };
    base *= monthModifiers[month] || 1.0;

    // Weather impact
    const weatherFactors = {
      Clear: 1.01,
      Clouds: 0.99,
      Drizzle: 0.96,
      Mist: 0.94,
      Rain: 0.91,
      Haze: 0.95,
      Fog: 0.88,
      Snow: 0.82,
      Thunderstorm: 0.78,
      Squall: 0.65,
    };
    base *= weatherFactors[params.weather_main] || 1.0;

    // Precipitation degradation
    if (params.rain_1h > 5.0) {
      base *= Math.max(0.70, 1.0 - params.rain_1h * 0.015);
    }
    if (params.snow_1h > 0.0) {
      base *= Math.max(0.65, 1.0 - params.snow_1h * 0.04);
    }

    // Temperature degradation
    if (params.temp_c < -15.0) base *= 0.91;
    else if (params.temp_c > 35.0) base *= 0.96;

    // Holiday impact
    if (params.is_holiday) base *= 0.62;

    const volume = Math.max(100, Math.min(7280, Math.round(base)));
    const maxCapacity = 7280;
    const capacityPct = Math.min(100, (volume / maxCapacity) * 100).toFixed(1);

    let level = "Moderate Commuter Flow";
    let color = "#f59e0b";
    let advisory = "Steady highway transit flow. Normal headways recommended.";

    if (volume >= 4800) {
      level = "High Congestion / Peak Rush";
      color = "#ef4444";
      advisory = "Severe commuter volume detected. Anticipate delays and consider alternate routes.";
    } else if (volume >= 2800) {
      level = "Moderate Commuter Flow";
      color = "#f59e0b";
      advisory = "Normal steady transit flow. Maintain safe following distances.";
    } else if (volume >= 1200) {
      level = "Light Traffic Flow";
      color = "#10b981";
      advisory = "Favorable transit conditions with minimal to no delays.";
    } else {
      level = "Minimal / Off-Peak Free Flow";
      color = "#06b6d4";
      advisory = "Virtually unhindered transit flow. Ideal maintenance and travel window.";
    }

    return {
      predicted_traffic_volume: volume,
      model_engine: "Calibrated Regression Engine (Zero-Cold-Start)",
      traffic_level: level,
      traffic_color: color,
      capacity_pct: capacityPct,
      advisory: advisory,
      confidence_bounds: {
        lower_bound: Math.max(0, Math.round(volume - 300.7)),
        upper_bound: Math.round(volume + 300.7),
        mae_margin: 300.7,
      },
      input_features: {
        hour,
        day_of_week: dayOfWeek,
        month,
        is_weekend: isWeekend,
        temp_c: params.temp_c,
        rain_1h: params.rain_1h,
        snow_1h: params.snow_1h,
        clouds_all: params.clouds_all,
        weather_main: params.weather_main,
        is_holiday: params.is_holiday ? 1 : 0,
      },
    };
  },

  async runPrediction(showToast = false) {
    const spinner = document.getElementById("pred-spinner");
    const btnText = document.getElementById("btn-predict-text");

    if (spinner) spinner.style.display = "inline-block";
    if (btnText) btnText.textContent = "Calculating Highway Flow...";

    // Assemble payload
    const payload = {
      date_time: document.getElementById("input-datetime")?.value.replace("T", " ") + ":00",
      weather_main: document.getElementById("input-weather")?.value || "Clear",
      temp_c: parseFloat(document.getElementById("input-temp")?.value || 15.0),
      rain_1h: parseFloat(document.getElementById("input-rain")?.value || 0.0),
      snow_1h: parseFloat(document.getElementById("input-snow")?.value || 0.0),
      clouds_all: parseInt(document.getElementById("input-clouds")?.value || 20, 10),
      is_holiday: document.getElementById("input-holiday")?.checked ? 1 : 0,
    };

    let result = null;

    try {
      // First attempt serverless API endpoint
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        result = await response.json();
      } else {
        // Fallback to local calibrated engine
        result = this.calculateCalibratedInference(payload);
      }
    } catch {
      // Offline / Static deployment fallback
      result = this.calculateCalibratedInference(payload);
    } finally {
      if (spinner) spinner.style.display = "none";
      if (btnText) btnText.textContent = "Run Traffic Flow Prediction";
    }

    if (result) {
      this.displayPredictionResult(result, payload);
      if (showToast) {
        this.showToast(`Forecast Updated: ${result.predicted_traffic_volume.toLocaleString()} veh/hr`);
      }
    }
  },

  displayPredictionResult(res, requestPayload) {
    const vol = Math.round(res.predicted_traffic_volume);
    const volEl = document.getElementById("pred-volume-value");
    const statusBadge = document.getElementById("traffic-status-badge");
    const statusText = document.getElementById("traffic-level-text");
    const engineLabel = document.getElementById("result-engine-label");
    const progressBar = document.getElementById("capacity-progress-bar");
    const capacityText = document.getElementById("capacity-pct-text");
    const confidenceText = document.getElementById("confidence-range-text");
    const advisoryText = document.getElementById("advisory-text");
    const jsonPre = document.getElementById("json-payload-display");

    // Animate counter
    if (volEl) {
      volEl.textContent = vol.toLocaleString();
    }

    // Engine Label
    if (engineLabel && res.model_engine) {
      engineLabel.textContent = res.model_engine;
    }

    // Badge styling
    const color = res.traffic_color || "#6366f1";
    if (statusBadge) {
      statusBadge.style.background = `${color}20`;
      statusBadge.style.borderColor = `${color}50`;
      statusBadge.style.color = color;
      const dot = statusBadge.querySelector(".badge-dot");
      if (dot) dot.style.background = color;
    }
    if (statusText) statusText.textContent = res.traffic_level || "Standard Flow";

    // Capacity bar
    const capPct = res.capacity_pct || ((vol / 7280) * 100).toFixed(1);
    if (progressBar) {
      progressBar.style.width = `${Math.min(100, capPct)}%`;
      progressBar.style.background = color;
    }
    if (capacityText) {
      capacityText.textContent = `${capPct}%`;
    }

    // Confidence
    if (confidenceText && res.confidence_bounds) {
      confidenceText.textContent = `${Math.round(res.confidence_bounds.lower_bound).toLocaleString()} – ${Math.round(res.confidence_bounds.upper_bound).toLocaleString()} veh/hr`;
    }

    // Advisory
    if (advisoryText && res.advisory) {
      advisoryText.textContent = res.advisory;
    }

    // JSON Payload
    if (jsonPre) {
      const inspectData = {
        request: requestPayload,
        response: res,
      };
      jsonPre.textContent = JSON.stringify(inspectData, null, 2);
    }
  },

  // Toast Notification
  showToast(message) {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<span>🚦</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(100%)";
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  },

  // Clipboard Helper
  copyCode(text) {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(() => {
        this.showToast("Copied to clipboard!");
      });
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
      this.showToast("Copied to clipboard!");
    }
  },
};

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  app.init();
});
