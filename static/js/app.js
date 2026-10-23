/**
 * CreditIQ — Interactive Application Logic
 * Handles tab transitions, preset loading, prediction dispatch,
 * and SHAP waterfall driver rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initForm();
    // Pre-load Prime preset as default starting point
    loadPreset("prime");
});

/* ── Tab Switching ─────────────────────────────────────────────────────────── */
function initTabs() {
    const tabs = document.querySelectorAll(".nav-tab");
    const panes = document.querySelectorAll(".tab-pane");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetId = tab.getAttribute("data-tab");

            tabs.forEach(t => t.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }
        });
    });
}

/* ── Persona Preset Loader ─────────────────────────────────────────────────── */
async function loadPreset(presetType) {
    try {
        const response = await fetch(`/api/sample/${presetType}`);
        if (!response.ok) throw new Error("Failed to load preset");
        
        const data = await response.json();
        
        // Populate form inputs
        for (const [key, val] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (el) {
                el.value = val;
                // Add a subtle flash effect
                el.style.borderColor = "var(--indigo-400)";
                setTimeout(() => { el.style.borderColor = ""; }, 400);
            }
        }
        
        // Auto-evaluate preset so the user sees results immediately
        evaluateForm();
    } catch (err) {
        console.error("Error loading preset:", err);
    }
}

/* ── Form Handling & Prediction ────────────────────────────────────────────── */
function initForm() {
    const form = document.getElementById("application-form");
    if (!form) return;

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        evaluateForm();
    });
}

async function evaluateForm() {
    const form = document.getElementById("application-form");
    const submitBtn = document.getElementById("btn-submit");
    const placeholder = document.getElementById("decision-placeholder");
    const results = document.getElementById("decision-results");

    // Gather form payload
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
        payload[key] = value;
    });

    // Loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
        <svg class="spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <line x1="12" y1="2" x2="12" y2="6"></line>
            <line x1="12" y1="18" x2="12" y2="22"></line>
            <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
            <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
            <line x1="2" y1="12" x2="6" y2="12"></line>
            <line x1="18" y1="12" x2="22" y2="12"></line>
            <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
            <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
        </svg>
        Evaluating Application Risk...
    `;

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === "success") {
            renderDecision(data);
            if (placeholder) placeholder.style.display = "none";
            if (results) results.style.display = "flex";
        } else {
            alert("Error scoring applicant: " + (data.message || "Unknown error"));
        }
    } catch (err) {
        console.error("Evaluation error:", err);
        alert("Failed to connect to underwriting backend.");
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            Run Automated Underwriting Decision
        `;
    }
}

/* ── Render Decision Results & SHAP Waterfall ──────────────────────────────── */
function renderDecision(data) {
    const banner = document.getElementById("decision-banner");
    const icon = document.getElementById("decision-icon");
    const title = document.getElementById("decision-title");
    const tier = document.getElementById("decision-tier");
    const probVal = document.getElementById("approval-prob-val");
    const barFill = document.getElementById("approval-bar-fill");
    const rejVal = document.getElementById("rejection-prob-val");
    const driversList = document.getElementById("drivers-list");

    const isApproved = data.decision === "APPROVED";

    // Update banner styling
    banner.className = `decision-banner ${isApproved ? "approved" : "rejected"}`;
    title.textContent = data.decision;
    tier.textContent = data.risk_tier;
    tier.style.color = data.risk_color;

    // Update icon
    icon.innerHTML = isApproved ? `
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
            <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
    ` : `
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
    `;

    // Probability & Gauge
    probVal.textContent = `${data.approval_probability}%`;
    probVal.style.color = isApproved ? "var(--emerald-400)" : "var(--rose-400)";
    barFill.style.width = `${Math.min(100, Math.max(0, data.approval_probability))}%`;
    barFill.style.background = isApproved ? 
        "linear-gradient(90deg, #F59E0B, #10B981)" : 
        "linear-gradient(90deg, #F43F5E, #FB7185)";
    
    rejVal.textContent = `Rejection Probability: ${data.rejection_probability}%`;

    // Render SHAP Driver list
    driversList.innerHTML = "";
    if (data.top_drivers && data.top_drivers.length > 0) {
        data.top_drivers.forEach(driver => {
            const isPos = driver.impact === "positive";
            const item = document.createElement("div");
            item.className = `driver-item ${driver.impact}`;
            item.innerHTML = `
                <div class="driver-badge">${isPos ? "↑" : "↓"}</div>
                <div class="driver-info">
                    <div class="driver-title">${driver.label} (${driver.feature})</div>
                    <div class="driver-reason">${driver.human_reason}</div>
                </div>
                <div class="driver-weight">${isPos ? "+" : ""}${driver.shap_value}</div>
            `;
            driversList.appendChild(item);
        });
    }
}
