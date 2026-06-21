"""
Bengaluru Traffic Event Intelligence System (BTEIS)
Streamlit Dashboard — Prototype v2
Structure:
  Page 1 — Live Prediction      (primary, demo lead)
  Page 2 — Model Performance & Bias  (analytics + feedback merged)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# ───────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTEIS — Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────
# Global CSS
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] p  { color: #c9d1d9 !important; font-size: 0.85rem; }
section[data-testid="stSidebar"] label {
    color: #e6edf3 !important; font-size: 0.93rem !important; font-weight: 500 !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="caption"] { color: #8b949e !important; }
section[data-testid="stSidebar"] hr { border-color: #21262d !important; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .value { font-size: 2rem; font-weight: 700; margin: 0.2rem 0; }
.metric-card .sub   { font-size: 0.8rem; color: #8b949e; }
.good  { color: #3fb950; }
.warn  { color: #d29922; }
.bad   { color: #f85149; }
.info  { color: #58a6ff; }

/* ── Section headers ── */
.section-header {
    font-size: 0.85rem; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 1.6rem 0 0.7rem; padding-bottom: 0.35rem;
    border-bottom: 1px solid #21262d;
}

/* ── Context pill (corridor recurring count) ── */
.ctx-pill {
    display: inline-block;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 0.6rem 1rem; font-size: 0.82rem; color: #8b949e;
    margin-bottom: 0.4rem; width: 100%;
}
.ctx-pill .ctx-val { font-size: 1.3rem; font-weight: 700; color: #58a6ff; display: block; }

/* Plotly chart container */
.js-plotly-plot .plotly { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_LOG = os.path.join(BASE_DIR, "feedback_log.csv")
DATA_CSV     = os.path.join(BASE_DIR, "Astram_Event_Data_Anonymized.csv")

# ───────────────────────────────────────────────────────────────
# Dark Plotly template
# ───────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    margin=dict(l=20, r=20, t=40, b=20),
)
TIER_COLORS = {"LOW": "#3fb950", "MEDIUM": "#d29922", "HIGH": "#f0883e", "CRITICAL": "#f85149"}

# ───────────────────────────────────────────────────────────────
# Rules engine (self-contained — no notebook dependency)
# ───────────────────────────────────────────────────────────────
_TIER_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

TIER_BASE_OFFICERS   = {"LOW": 2, "MEDIUM": 4, "HIGH": 8,  "CRITICAL": 15}
TIER_BASE_BARRICADES = {"LOW": 2, "MEDIUM": 5, "HIGH": 9,  "CRITICAL": 14}
BARRICADE_SPACING_M  = {"LOW": 50, "MEDIUM": 50, "HIGH": 30, "CRITICAL": 20}
ROAD_CLOSURE_MULT    = 1.5
MAX_CORRIDOR_WEIGHT  = 1.4

DIVERSION_MAP = {
    "Outer Ring Road":      ["Mysore Road → Tumkur Road via Peenya", "Bannerghatta Road → Hosur Road via Electronic City"],
    "Bangalore-Mysore Road":["NICE Road (toll)", "Kanakapura Road"],
    "Hosur Road":           ["Sarjapur Road → ORR", "Bannerghatta Road → JP Nagar 6th Phase"],
    "Tumkur Road":          ["NH-44 bypass (Dabaspet)", "Yeshwantpur → Rajajinagar → Chord Road"],
    "Bellary Road 1":       ["Hennur Road → Kalyan Nagar → ORR", "Nagawara Road"],
    "Bellary Road 2":       ["Hennur Road → Kalyan Nagar → ORR", "Nagawara Road"],
    "Mysore Road":          ["NICE Road (toll)", "Kanakapura Road"],
    "Sarjapur Road":        ["Marathahalli → ORR → Silk Board", "Carmelaram Road"],
    "Old Madras Road":      ["KR Puram → ORR", "Whitefield Main Road"],
    "Bannerghata Road":     ["JP Nagar → BTM → Silk Board", "Electronic City flyover"],
    "ORR North 1":          ["Hebbal via Nagawara", "Yelhanka alternate"],
    "ORR North 2":          ["Hebbal via Nagawara", "Yelhanka alternate"],
    "ORR East 1":           ["Whitefield via Varthur", "KR Puram via Old Madras Road"],
    "ORR East 2":           ["Whitefield via Varthur", "KR Puram via Old Madras Road"],
    "ORR West 1":           ["Magadi Road → Rajajinagar", "Chord Road"],
    "Magadi Road":          ["Chord Road → Rajajinagar", "Tumkur Road via Peenya"],
    "Non-corridor":         ["Use parallel service lanes", "Contact local police station for alternate advisory"],
}

def get_response_tier(severity_score, predicted_duration_mins):
    sev_tier = "CRITICAL" if severity_score >= 5 else "HIGH" if severity_score >= 4 else "MEDIUM" if severity_score >= 3 else "LOW"
    dur_tier = "CRITICAL" if predicted_duration_mins > 480 else "HIGH" if predicted_duration_mins > 120 else "MEDIUM" if predicted_duration_mins > 30 else "LOW"
    sev_idx  = _TIER_ORDER.index(sev_tier)
    dur_idx  = _TIER_ORDER.index(dur_tier)
    return _TIER_ORDER[min(max(sev_idx, dur_idx), sev_idx + 1)]

def recommend_manpower(tier, requires_road_closure, corridor_freq, affected_stretch_km):
    base   = TIER_BASE_OFFICERS[tier]
    c_mult = ROAD_CLOSURE_MULT if requires_road_closure else 1.0
    c_wt   = 1.0 + min(corridor_freq / 0.30, 1.0) * (MAX_CORRIDOR_WEIGHT - 1.0)
    return int(np.ceil(base * c_mult * c_wt)) + max(0, int(affected_stretch_km - 0.5))

def recommend_barricades(tier, affected_stretch_km):
    base      = TIER_BASE_BARRICADES[tier]
    spacing_m = BARRICADE_SPACING_M[tier]
    stretch_m = affected_stretch_km * 1000
    addon     = max(0, int(stretch_m / spacing_m) - 1) if stretch_m > spacing_m else 0
    total     = base + addon
    desc      = f"{total} barricades — {base} tier-base" + (f" + {addon} stretch-addon @ {spacing_m} m" if addon else "")
    return total, desc

def recommend_diversion(corridor, tier, requires_road_closure):
    if tier in ("LOW", "MEDIUM") and not requires_road_closure:
        return ["No diversion needed — manage in-place with traffic marshalling"]
    routes = DIVERSION_MAP.get(corridor)
    return routes if routes else ["Activate VMS boards for upstream diversion", "Contact TOC for dynamic route advisory"]

# ───────────────────────────────────────────────────────────────
# Static lookup tables (from EDA)
# ───────────────────────────────────────────────────────────────
CAUSE_MEDIAN_MINS = {
    "construction": 381, "road_conditions": 246, "public_event": 121,
    "water_logging": 107, "tree_fall": 90, "others": 75,
    "congestion": 72, "procession": 58, "accident": 41,
    "vehicle_breakdown": 41, "pot_holes": 28, "vip_movement": 11,
    "protest": 3, "debris": 5,
}
CORRIDOR_FREQ = {
    "Non-corridor": 0.0, "Mysore Road": 0.090, "Bellary Road 1": 0.074,
    "Tumkur Road": 0.056, "Bellary Road 2": 0.046, "Hosur Road": 0.036,
    "ORR North 1": 0.033, "Old Madras Road": 0.032, "Magadi Road": 0.030,
    "ORR East 1": 0.030, "ORR North 2": 0.028, "Bannerghata Road": 0.026,
    "ORR East 2": 0.023, "West of Chord Road": 0.021, "ORR West 1": 0.021,
}
# Approximate recurring event counts per corridor (from EDA top-15)
CORRIDOR_EVENT_COUNT = {
    "Non-corridor": 3102, "Mysore Road": 739, "Bellary Road 1": 606,
    "Tumkur Road": 457, "Bellary Road 2": 376, "Hosur Road": 297,
    "ORR North 1": 270, "Old Madras Road": 261, "Magadi Road": 245,
    "ORR East 1": 244, "ORR North 2": 233, "Bannerghata Road": 209,
    "ORR East 2": 185, "West of Chord Road": 174, "ORR West 1": 168,
}

def naive_predict_duration(event_cause):
    return float(CAUSE_MEDIAN_MINS.get(event_cause.lower().strip(), 60))

def simple_severity(priority, requires_road_closure):
    return {"Low": 1, "Medium": 2, "High": 3}.get(priority, 1) + (2 if requires_road_closure else 0)

# ───────────────────────────────────────────────────────────────
# Feedback log loader
# ───────────────────────────────────────────────────────────────
def load_feedback():
    if not os.path.isfile(FEEDBACK_LOG):
        return pd.DataFrame()
    df = pd.read_csv(FEEDBACK_LOG)
    for col in ["pred_duration_mins", "actual_duration_mins", "duration_error_mins",
                "pred_severity_score", "actual_severity_score", "severity_correct",
                "tier_correct", "rec_officer_count", "actual_officers_deployed",
                "officer_gap", "rec_barricade_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ───────────────────────────────────────────────────────────────
# Sidebar nav — 2 pages only
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 BTEIS")
    st.markdown("**Bengaluru Traffic Event\nIntelligence System**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🎯  Live Prediction", "📊  Model Performance & Bias"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Prototype · Flipkart Hackathon Round 2")

page_key = page.split("  ")[1]

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════
if page_key == "Live Prediction":
    st.markdown("## 🚦 Live Event Response Planner")
    st.markdown("Enter an incoming traffic event to get an instant response recommendation.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Event Details</div>', unsafe_allow_html=True)
        event_cause = st.selectbox("Event Cause", [
            "vehicle_breakdown", "accident", "tree_fall", "pot_holes", "water_logging",
            "construction", "road_conditions", "congestion", "public_event",
            "procession", "vip_movement", "protest", "debris", "others",
        ])
        corridor = st.selectbox("Corridor", list(CORRIDOR_FREQ.keys()))
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        requires_road_closure = st.toggle("Road Closure Required", value=False)
        affected_stretch_km   = st.slider("Affected Stretch (km)", 0.0, 5.0, 0.5, 0.1)
        is_rush_hour          = st.toggle("Rush Hour (8–11 AM / 5–8 PM)", value=True)

        # ── Corridor context (folded-in from hotspot data) ──
        st.markdown('<div class="section-header">Corridor Context</div>', unsafe_allow_html=True)
        hist_count  = CORRIDOR_EVENT_COUNT.get(corridor, 0)
        corr_freq   = CORRIDOR_FREQ.get(corridor, 0.0)
        cause_median = CAUSE_MEDIAN_MINS.get(event_cause.lower(), 60)
        ctx1, ctx2, ctx3 = st.columns(3)
        with ctx1:
            st.markdown(f"""<div class="ctx-pill">
                <span class="ctx-val">{hist_count:,}</span>
                historical events on this corridor</div>""", unsafe_allow_html=True)
        with ctx2:
            st.markdown(f"""<div class="ctx-pill">
                <span class="ctx-val">{corr_freq*100:.1f}%</span>
                of all city events</div>""", unsafe_allow_html=True)
        with ctx3:
            st.markdown(f"""<div class="ctx-pill">
                <span class="ctx-val">{cause_median} min</span>
                typical {event_cause} duration</div>""", unsafe_allow_html=True)

        predict_btn = st.button("⚡  Generate Response Plan", use_container_width=True, type="primary")

    # ── Result panel ──
    if predict_btn:
        pred_duration = naive_predict_duration(event_cause)
        if is_rush_hour:
            pred_duration = round(pred_duration * 1.25, 1)
        pred_severity = simple_severity(priority, requires_road_closure)
        tier          = get_response_tier(pred_severity, pred_duration)
        corridor_freq = CORRIDOR_FREQ.get(corridor, 0.02)
        officers      = recommend_manpower(tier, requires_road_closure, corridor_freq, affected_stretch_km)
        barricades, barricade_desc = recommend_barricades(tier, affected_stretch_km)
        diversion     = recommend_diversion(corridor, tier, requires_road_closure)

        with col_result:
            tier_color = TIER_COLORS[tier]

            # Tier hero card
            st.markdown(f"""
            <div class="metric-card" style="border-color:{tier_color}66;">
                <div class="label">Response Tier</div>
                <div class="value" style="color:{tier_color}; font-size:2.4rem; letter-spacing:0.03em">{tier}</div>
                <div class="sub">Severity score {pred_severity} · Predicted duration {pred_duration:.0f} min</div>
            </div>""", unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Predicted Duration</div>
                    <div class="value info">{pred_duration:.0f} min</div>
                    <div class="sub">≈ {pred_duration/60:.1f} hrs</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Officers to Deploy</div>
                    <div class="value" style="color:{tier_color}">{officers}</div>
                    <div class="sub">personnel</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Barricade Units</div>
                    <div class="value" style="color:{tier_color}">{barricades}</div>
                    <div class="sub">{BARRICADE_SPACING_M[tier]} m spacing</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Barricade Placement</div>', unsafe_allow_html=True)
            st.info(f"📍 {barricade_desc}")

            st.markdown('<div class="section-header">Diversion Routes</div>', unsafe_allow_html=True)
            for i, route in enumerate(diversion, 1):
                st.markdown(f"**Route {i}:** {route}")

            # Deployment breakdown table
            st.markdown('<div class="section-header">Deployment Breakdown</div>', unsafe_allow_html=True)
            role_templates = {
                "LOW":      {"Scene officers": 1, "Supervisor": 1},
                "MEDIUM":   {"Scene officers": 2, "Crowd control": 1, "Supervisor": 1},
                "HIGH":     {"Scene officers": 4, "Diversion pts": 2, "PI officer": 1, "Supervisor": 1},
                "CRITICAL": {"Scene officers": 6, "Diversion pts": 4, "Crowd control": 2, "PI officer": 2, "DCP": 1},
            }
            roles = dict(role_templates[tier])
            extra = officers - sum(roles.values())
            if extra > 0:
                roles["Corridor/stretch extra"] = extra
            st.dataframe(
                pd.DataFrame(list(roles.items()), columns=["Role", "Count"]),
                use_container_width=True, hide_index=True,
            )

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE & BIAS
# ═══════════════════════════════════════════════════════════════
elif page_key == "Model Performance & Bias":
    st.markdown("## 📊 Model Performance & Feedback Bias Analysis")
    st.markdown(
        "Evidence that the feedback loop works and honest accounting of the duration model's limits. "
        "Per-cause and per-corridor bias rows surface systematic errors the naive baseline cannot."
    )

    df     = load_feedback()
    closed = df[df["actual_duration_mins"].notna()].copy() if not df.empty else pd.DataFrame()

    # ── 5 KPI cards ──
    k1, k2, k3, k4, k5 = st.columns(5)
    if closed.empty:
        for col, label in zip([k1, k2, k3, k4, k5],
                              ["Events Logged", "Duration MAE", "Prediction Bias", "Severity Accuracy", "Tier Accuracy"]):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value" style="color:#30363d">—</div>
                    <div class="sub">no data yet</div></div>""", unsafe_allow_html=True)
    else:
        mae       = closed["duration_error_mins"].abs().mean()
        bias      = closed["duration_error_mins"].mean()
        sev_acc   = closed["severity_correct"].mean() * 100
        tier_acc  = closed["tier_correct"].mean() * 100
        n_events  = len(closed)

        with k1:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Events Logged</div>
                <div class="value info">{n_events}</div>
                <div class="sub">with actuals</div></div>""", unsafe_allow_html=True)
        with k2:
            c = "bad" if mae > 60 else "good"
            st.markdown(f"""<div class="metric-card">
                <div class="label">Duration MAE</div>
                <div class="value {c}">{mae:.0f} min</div>
                <div class="sub">mean abs error</div></div>""", unsafe_allow_html=True)
        with k3:
            direction = "over" if bias > 0 else "under"
            c = "warn" if abs(bias) > 30 else "good"
            st.markdown(f"""<div class="metric-card">
                <div class="label">Prediction Bias</div>
                <div class="value {c}">{bias:+.0f} min</div>
                <div class="sub">{direction}-predicting</div></div>""", unsafe_allow_html=True)
        with k4:
            c = "good" if sev_acc >= 70 else "bad"
            st.markdown(f"""<div class="metric-card">
                <div class="label">Severity Accuracy</div>
                <div class="value {c}">{sev_acc:.0f}%</div>
                <div class="sub">pred = actual</div></div>""", unsafe_allow_html=True)
        with k5:
            c = "good" if tier_acc >= 70 else "bad"
            st.markdown(f"""<div class="metric-card">
                <div class="label">Tier Accuracy</div>
                <div class="value {c}">{tier_acc:.0f}%</div>
                <div class="sub">response tier match</div></div>""", unsafe_allow_html=True)

        # ── Retrain alert ──
        needs_retrain = mae > 60 or tier_acc < 70
        if needs_retrain:
            st.error(f"⚠️  **Retraining recommended** — MAE {mae:.0f} min  |  Tier accuracy {tier_acc:.0f}%  "
                     f"(threshold: MAE < 60 min, tier acc ≥ 70%)")
        else:
            st.success(f"✅  Model within acceptable range — MAE {mae:.0f} min  |  Tier accuracy {tier_acc:.0f}%")

    # ── Charts: Predicted vs Actual + Error by Cause ──
    if not closed.empty:
        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            st.markdown('<div class="section-header">Predicted vs Actual Duration</div>', unsafe_allow_html=True)
            max_val = max(closed["pred_duration_mins"].max(), closed["actual_duration_mins"].max()) * 1.1
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], mode="lines",
                line=dict(color="#30363d", dash="dash", width=1),
                name="Perfect prediction", showlegend=False,
            ))
            dot_colors = [TIER_COLORS.get(str(t), "#8b949e") for t in closed.get("pred_tier", [])]
            fig_scatter.add_trace(go.Scatter(
                x=closed["actual_duration_mins"],
                y=closed["pred_duration_mins"],
                mode="markers+text",
                text=closed["event_cause"] if "event_cause" in closed.columns else None,
                textposition="top center",
                textfont=dict(size=9, color="#8b949e"),
                marker=dict(size=12, color=dot_colors, line=dict(width=1, color="#30363d")),
                hovertemplate="<b>%{text}</b><br>Actual: %{x:.0f} min<br>Predicted: %{y:.0f} min<extra></extra>",
            ))
            fig_scatter.update_layout(**DARK,
                title=dict(text="Dashed = perfect prediction  ·  dots coloured by predicted tier",
                           font=dict(size=11, color="#8b949e")))
            fig_scatter.update_xaxes(title="Actual Duration (min)")
            fig_scatter.update_yaxes(title="Predicted Duration (min)")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with ch2:
            st.markdown('<div class="section-header">Duration Error by Event Cause</div>', unsafe_allow_html=True)
            if "event_cause" in closed.columns:
                cause_err = (closed.groupby("event_cause")["duration_error_mins"]
                             .agg(avg_error="mean", n="count").reset_index()
                             .sort_values("avg_error"))
                bar_colors = ["#f85149" if e > 30 else "#3fb950" if e < -30 else "#58a6ff"
                              for e in cause_err["avg_error"]]
                fig_cause = go.Figure(go.Bar(
                    x=cause_err["avg_error"], y=cause_err["event_cause"], orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:+.0f} min (n={n})"
                          for v, n in zip(cause_err["avg_error"], cause_err["n"])],
                    textposition="outside", textfont=dict(size=10),
                    hovertemplate="%{y}: %{x:+.1f} min avg error<extra></extra>",
                ))
                fig_cause.add_vline(x=0, line_color="#30363d", line_width=1)
                fig_cause.update_layout(**DARK,
                    title=dict(text="Red = over-predicting  ·  Green = under  ·  Blue = accurate",
                               font=dict(size=11, color="#8b949e")))
                fig_cause.update_xaxes(title="Avg Error (min)")
                st.plotly_chart(fig_cause, use_container_width=True)

        # ── Corridor bias ──
        if "corridor" in closed.columns:
            st.markdown('<div class="section-header">Duration Error by Corridor</div>', unsafe_allow_html=True)
            corr_err = (closed.groupby("corridor")["duration_error_mins"]
                        .agg(avg_error="mean", n="count").reset_index()
                        .sort_values("avg_error"))
            corr_colors = ["#f85149" if abs(e) > 60 else "#d29922" if abs(e) > 30 else "#3fb950"
                           for e in corr_err["avg_error"]]
            fig_corr = go.Figure(go.Bar(
                x=corr_err["avg_error"], y=corr_err["corridor"], orientation="h",
                marker_color=corr_colors,
                text=[f"{v:+.0f} min (n={n})"
                      for v, n in zip(corr_err["avg_error"], corr_err["n"])],
                textposition="outside",
                hovertemplate="%{y}: %{x:+.1f} min<extra></extra>",
            ))
            fig_corr.add_vline(x=0, line_color="#30363d", line_width=1)
            fig_corr.update_layout(**DARK,
                title=dict(text="Systematic corridor bias — informs retraining priority",
                           font=dict(size=11, color="#8b949e")))
            fig_corr.update_xaxes(title="Avg Error (min, + = over-predicting)")
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Resource adequacy + Tier distribution side-by-side ──
        ra_col, td_col = st.columns(2, gap="large")

        with ra_col:
            st.markdown('<div class="section-header">Resource Adequacy (Dispatcher Feedback)</div>',
                        unsafe_allow_html=True)
            for field, label, icon in [
                ("officers_adequate",  "Officers",   "👮"),
                ("barricades_adequate","Barricades", "🚧"),
                ("diversion_effective","Diversion",  "🗺️"),
            ]:
                if field in df.columns:
                    vals = pd.to_numeric(df[field], errors="coerce").dropna()
                    pct  = vals.mean() * 100 if not vals.empty else None
                else:
                    pct = None
                if pct is not None:
                    c = "good" if pct >= 75 else "warn" if pct >= 50 else "bad"
                    st.markdown(f"""<div class="metric-card" style="display:flex;align-items:center;gap:1rem">
                        <span style="font-size:1.6rem">{icon}</span>
                        <div>
                          <div class="label">{label}</div>
                          <div class="value {c}" style="font-size:1.4rem">{pct:.0f}% adequate</div>
                          <div class="sub">n = {len(vals)}</div>
                        </div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="metric-card">
                        <span style="font-size:1.2rem">{icon}</span>
                        <div class="label">{label}</div>
                        <div class="sub">no dispatcher feedback yet</div></div>""", unsafe_allow_html=True)

        with td_col:
            st.markdown('<div class="section-header">Predicted vs Actual Tier Distribution</div>',
                        unsafe_allow_html=True)
            tiers_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            pred_counts   = df["pred_tier"].value_counts().reindex(tiers_order, fill_value=0) if "pred_tier" in df.columns else pd.Series([0]*4, index=tiers_order)
            actual_counts = closed["actual_tier"].value_counts().reindex(tiers_order, fill_value=0) if ("actual_tier" in closed.columns and not closed.empty) else pd.Series([0]*4, index=tiers_order)

            fig_tier = go.Figure()
            fig_tier.add_trace(go.Bar(
                name="Predicted", x=tiers_order, y=pred_counts.values,
                marker_color=[TIER_COLORS[t] for t in tiers_order], opacity=0.9,
                text=pred_counts.values, textposition="outside",
            ))
            fig_tier.add_trace(go.Bar(
                name="Actual", x=tiers_order, y=actual_counts.values,
                marker_color=[TIER_COLORS[t] for t in tiers_order], opacity=0.4,
                text=actual_counts.values, textposition="outside",
            ))
            fig_tier.update_layout(**DARK, barmode="group",
                legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
                title=dict(text="Solid = predicted  ·  Faded = actual outcome",
                           font=dict(size=11, color="#8b949e")))
            st.plotly_chart(fig_tier, use_container_width=True)

        # ── Raw feedback log table ──
        st.markdown('<div class="section-header">Full Feedback Log</div>', unsafe_allow_html=True)
        display_cols = [c for c in [
            "event_id", "event_cause", "corridor",
            "pred_duration_mins", "actual_duration_mins", "duration_error_mins",
            "pred_tier", "actual_tier", "severity_correct", "tier_correct",
            "officers_adequate", "dispatcher_notes"
        ] if c in df.columns]

        def color_error(val):
            try:
                v = float(val)
                if abs(v) > 120: return "color: #f85149"
                if abs(v) > 30:  return "color: #d29922"
                return "color: #3fb950"
            except Exception:
                return ""

        styled = (df[display_cols].style
                  .map(color_error,
                       subset=["duration_error_mins"] if "duration_error_mins" in display_cols else []))
        st.dataframe(styled, use_container_width=True, hide_index=True)

    else:
        st.info("No closed events in the feedback log yet. Run the notebook demo cell to populate data.")
