import json
import time
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="AI Heart Pattern Checker",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #102033;
        --muted: #5f6f86;
        --line: #dbe7f3;
        --surface: #ffffff;
        --soft: #f4f9ff;
        --blue: #2563eb;
        --blue-soft: #dbeafe;
        --green: #1f7a4d;
        --green-soft: #e8f7ef;
        --yellow: #9a6700;
        --yellow-soft: #fff5cc;
        --red: #ba1a1a;
        --red-soft: #ffe8e8;
        --purple: #7c3aed;
        --purple-soft: #ede9fe;
        --pink: #ec4899;
        --pink-soft: #fce7f3;
        --orange: #f59e0b;
        --orange-soft: #fef3c7;
        --cyan: #06b6d4;
        --cyan-soft: #cffafe;
    }

    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 50%, #f5f3ff 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1160px;
        padding-top: 2.2rem;
        padding-bottom: 2.8rem;
    }

    /* Animated gradient hero */
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 50%, #f5f0ff 100%);
        border: 2px solid var(--line);
        border-radius: 18px;
        padding: 32px;
        box-shadow: 0 18px 45px rgba(37, 99, 235, 0.12);
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        animation: shimmer 3s ease-in-out infinite;
    }

    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(37, 99, 235, 0.03) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }

    .hero h1 {
        color: var(--ink);
        font-size: 2.3rem;
        line-height: 1.12;
        margin: 0 0 0.7rem;
        letter-spacing: 0;
        position: relative;
        z-index: 1;
        background: linear-gradient(135deg, #1a365d, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.55;
        max-width: 760px;
        margin: 0 0 0.9rem;
        position: relative;
        z-index: 1;
    }

    .research-note {
        display: inline-flex;
        color: #244a70;
        background: linear-gradient(135deg, var(--blue-soft), var(--purple-soft));
        border-radius: 999px;
        padding: 8px 18px;
        font-weight: 800;
        border: 1px solid rgba(37, 99, 235, 0.2);
        position: relative;
        z-index: 1;
        transition: all 0.3s ease;
    }

    .research-note:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 900;
        margin: 24px 0 12px;
        padding: 8px 0;
        border-bottom: 3px solid transparent;
        background: linear-gradient(135deg, var(--blue), var(--purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
    }

    .section-title::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, var(--blue), var(--purple));
        border-radius: 3px;
        -webkit-text-fill-color: initial;
    }

    .section-title span {
        background: none;
        -webkit-text-fill-color: initial;
        color: var(--ink);
    }

    .info-tip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--blue-soft), var(--purple-soft));
        color: var(--blue);
        font-size: 0.72rem;
        font-weight: 900;
        position: relative;
        cursor: help;
        transition: all 0.3s ease;
    }

    .info-tip:hover {
        transform: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    .info-tip .tooltip-text {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        z-index: 9999;
        left: 50%;
        top: 30px;
        transform: translateX(-50%);
        rotate: 0deg;
        animation: none;
        width: min(280px, 72vw);
        background: #1a365d;
        color: white;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 0.82rem;
        font-weight: 700;
        line-height: 1.35;
        box-shadow: 0 12px 28px rgba(16, 32, 51, 0.22);
        transition: visibility 0.2s ease, opacity 0.2s ease;
        pointer-events: none;
    }

    .info-tip:hover .tooltip-text,
    .info-tip:focus .tooltip-text {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%);
        rotate: 0deg;
    }

    .term-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 900;
        margin-bottom: 6px;
    }

    .term-help {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 12px;
    }

    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 12px;
        font-size: 0.82rem;
        font-weight: 900;
        margin: 4px 0 8px;
        transition: all 0.3s ease;
        cursor: default;
    }

    .badge:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .badge-normal {
        color: #0d6e3a;
        background: var(--green-soft);
        border: 1px solid #86d9a5;
    }

    .badge-borderline {
        color: #8a5d00;
        background: var(--yellow-soft);
        border: 1px solid #fcd34d;
    }

    .badge-abnormal {
        color: #9a0000;
        background: var(--red-soft);
        border: 1px solid #fca5a5;
    }

    .result-card {
        border-radius: 18px;
        border: 2px solid var(--line);
        padding: 24px;
        margin: 18px 0;
        box-shadow: 0 14px 34px rgba(16, 32, 51, 0.06);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }

    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 48px rgba(16, 32, 51, 0.12);
    }

    .result-normal {
        background: linear-gradient(135deg, #f0fdf4, #ffffff);
        border-color: #86d9a5;
    }

    .result-normal::before {
        content: '✓';
        position: absolute;
        top: -10px;
        right: -10px;
        font-size: 60px;
        opacity: 0.05;
        color: var(--green);
    }

    .result-alert {
        background: linear-gradient(135deg, #fef2f2, #ffffff);
        border-color: #fca5a5;
        animation: alertPulse 2s ease-in-out infinite;
    }

    @keyframes alertPulse {
        0%, 100% { border-color: #fca5a5; }
        50% { border-color: #ef4444; }
    }

    .result-alert::before {
        content: '⚠';
        position: absolute;
        top: -10px;
        right: -10px;
        font-size: 60px;
        opacity: 0.05;
        color: var(--red);
    }

    .result-kicker {
        color: var(--muted);
        font-weight: 900;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 1px;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
    }

    .result-title {
        color: #0a1628;
        font-size: 1.55rem;
        font-weight: 900;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
    }

    .result-copy {
        color: #1a2d44;
        font-size: 1rem;
        line-height: 1.55;
        max-width: 780px;
        position: relative;
        z-index: 1;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 2px solid var(--line);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 10px 24px rgba(16, 32, 51, 0.05);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.12);
        border-color: var(--blue);
    }

    div[data-testid="stMetric"] label {
        color: #0a1628 !important;
        font-weight: 700;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0a1628 !important;
        font-weight: 900;
        font-size: 1.8rem;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #2d3748 !important;
        font-weight: 600;
    }

    .stNumberInput label {
        color: var(--ink) !important;
        font-weight: 800;
    }

    .stNumberInput input {
        border-radius: 10px;
        border: 2px solid var(--line);
        transition: all 0.3s ease;
        background: #ffffff !important;
        color: #0a1628 !important;
        -webkit-text-fill-color: #0a1628 !important;
        font-weight: 600;
    }

    .stNumberInput [data-baseweb="input"],
    .stNumberInput [data-baseweb="input"] > div {
        background: #ffffff !important;
        color: #0a1628 !important;
        border-color: var(--line) !important;
    }

    .stNumberInput button {
        background: #ffffff !important;
        color: #0a1628 !important;
        border-color: var(--line) !important;
    }

    .stNumberInput button svg {
        fill: #0a1628 !important;
        color: #0a1628 !important;
    }

    .stNumberInput input:focus {
        border-color: var(--blue);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    .stNumberInput input:hover {
        border-color: var(--blue);
    }

    .stButton > button {
        width: 100%;
        min-height: 58px;
        border: 0;
        border-radius: 16px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        font-size: 1.05rem;
        font-weight: 900;
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.28);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: all 0.6s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(37, 99, 235, 0.35);
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    .stProgress > div {
        background: linear-gradient(90deg, var(--blue), var(--purple));
        border-radius: 999px;
        height: 8px !important;
        transition: width 0.5s ease;
    }

    /* Custom container styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        border-radius: 16px;
        background: var(--surface);
        padding: 16px;
        border: 2px solid var(--line);
        transition: all 0.3s ease;
    }

    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover {
        border-color: var(--blue);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.08);
    }

    /* Dialog styling */
    .stDialog > div {
        background: linear-gradient(135deg, #ffffff, #f8fbff);
        border-radius: 20px;
        border: 2px solid var(--line);
        box-shadow: 0 24px 64px rgba(16, 32, 51, 0.2);
    }

    /* Progress bar in range check */
    .stProgress > div > div {
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
        border-radius: 999px;
    }

    /* Status message animations */
    .stStatus {
        border-radius: 12px;
        border-left: 4px solid var(--blue);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stAlert {
        animation: fadeIn 0.5s ease;
        border-radius: 12px;
        border-left: 4px solid;
    }

    /* Expander styling */
    .stExpander {
        border-radius: 12px;
        border: 2px solid var(--line);
        transition: all 0.3s ease;
    }

    .stExpander:hover {
        border-color: var(--blue);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
    }

    .stExpander details {
        border-radius: 12px;
    }

    .stExpander details summary {
        font-weight: 700;
        color: var(--blue);
        padding: 8px 12px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--surface);
        border-radius: 12px;
        padding: 4px;
        border: 2px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 700;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--blue-soft);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--blue), var(--purple));
        color: white !important;
    }

    /* Spinner animation for analyze button */
    .stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #10b981, #059669);
    }

    /* Improve readability in result dialog */
    .stDialog .section-title,
    .stDialog .section-title span {
        background: none !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
    }

    .stDialog .section-title::after {
        background: linear-gradient(90deg, #60a5fa, #a78bfa) !important;
    }

    .stDialog .section-title .info-tip {
        background: #0f172a !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        text-shadow: none !important;
    }

    .stDialog .section-title .info-tip .tooltip-text {
        background: #0f172a !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #60a5fa;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.45);
        text-shadow: none !important;
        line-height: 1.35;
        top: 34px;
        width: min(340px, 78vw);
    }

    .stDialog [data-testid="stMetric"] label {
        color: #0a1628 !important;
        font-weight: 700 !important;
    }

    .stDialog [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0a1628 !important;
        font-weight: 900 !important;
        font-size: 2rem !important;
    }

    .stDialog [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }

    /* Make badge text more visible */
    .badge {
        font-weight: 900 !important;
    }

    .badge-normal {
        color: #0d6e3a !important;
        background: #d1fae5 !important;
    }

    .badge-borderline {
        color: #8a5d00 !important;
        background: #fef3c7 !important;
    }

    .badge-abnormal {
        color: #9a0000 !important;
        background: #fee2e2 !important;
    }

    /* Caption text visibility */
    .stCaption {
        color: #1a2d44 !important;
        font-weight: 500;
    }

    /* Ensure progress label visibility */
    .stProgress label {
        color: #0a1628 !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


FEATURE_COLUMNS = [
    "Heart_Rate",
    "Temperature",
    "Systolic_BP",
    "Diastolic_BP",
    "MAP",
    "Pulse_Pressure",
]

VITAL_RANGES = {
    "heart_rate": {"normal": (60, 100), "borderline": (50, 120), "scale": (40, 200), "unit": "bpm"},
    "temperature": {"normal": (36.1, 37.2), "borderline": (35.8, 38.0), "scale": (35.0, 42.0), "unit": "C"},
    "systolic": {"normal": (90, 120), "borderline": (80, 139), "scale": (80, 200), "unit": "mmHg"},
    "diastolic": {"normal": (60, 80), "borderline": (50, 89), "scale": (50, 130), "unit": "mmHg"},
    "map": {"normal": (70, 100), "borderline": (60, 110), "scale": (40, 140), "unit": "mmHg"},
    "pulse_pressure": {"normal": (30, 50), "borderline": (20, 60), "scale": (0, 100), "unit": "mmHg"},
}

TOOLTIPS = {
    "your_vitals": "Enter the basic measurements the AI needs to check your current heart pattern.",
    "calculated": "These values are calculated automatically from your blood pressure readings.",
    "range_overview": "Shows whether each measurement is low, normal, borderline, or abnormal.",
    "results": "Summarizes whether your current pattern looks normal or unusual.",
    "heart_rate": "The number of times your heart beats in one minute.",
    "temperature": "Measures your body's internal temperature.",
    "systolic": "The top blood pressure number. It measures pressure when your heart beats.",
    "diastolic": "The bottom blood pressure number. It measures pressure while your heart rests.",
    "map": "The average blood pressure flowing through your arteries during one heartbeat.",
    "pulse_pressure": "The difference between systolic and diastolic blood pressure.",
    "ai_risk": "Shows how unusual your heart pattern appears compared with healthy patterns learned by the AI model.",
    "ai_confidence": "Indicates how confident the AI is in its prediction based on the learned data.",
    "recommendations": "Suggested actions based on your current analysis. These recommendations do not replace medical advice.",
    "advanced": "Shows technical information used by the AI model.",
    "reconstruction_error": "The difference between your entered vital signs and what the AI expects from a normal heart pattern.",
    "threshold": "The maximum reconstruction error considered normal.",
    "normalized_features": "The values after scaling them into the format expected by the AI model.",
}


@st.cache_resource
def load_assets():
    app_dir = Path(__file__).resolve().parent
    scaler_path = app_dir / "unified_minmax_scaler.pkl"
    model_path = app_dir / "final_champion_ae.keras"
    threshold_path = app_dir / "ae_threshold.json"

    missing = [
        path.name
        for path in (scaler_path, model_path, threshold_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifact(s): " + ", ".join(missing)
        )

    scaler = joblib.load(scaler_path)
    model = tf.keras.models.load_model(model_path)
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]
    return scaler, model, threshold


def predict_risk(hr, temp, sbp, dbp, map_val, pulse_pressure):
    scaler, model, threshold = load_assets()
    input_data = pd.DataFrame(
        [[hr, temp, sbp, dbp, map_val, pulse_pressure]],
        columns=FEATURE_COLUMNS,
    )
    input_scaled = scaler.transform(input_data)
    reconstruction = model.predict(input_scaled, verbose=0)
    error = float(np.mean(np.power(input_scaled - reconstruction, 2)))
    scaled_features = pd.DataFrame(input_scaled, columns=FEATURE_COLUMNS)
    return error, threshold, error > threshold, scaled_features


def classify_value(value, range_key):
    low, high = VITAL_RANGES[range_key]["normal"]
    border_low, border_high = VITAL_RANGES[range_key]["borderline"]
    if low <= value <= high:
        return "Normal", "badge-normal"
    if border_low <= value <= border_high:
        return "Borderline", "badge-borderline"
    return "Abnormal", "badge-abnormal"


def format_value(value, decimals=0):
    if decimals:
        return f"{value:.{decimals}f}"
    return f"{value:.0f}"


def progress_percent(value, range_key):
    scale_low, scale_high = VITAL_RANGES[range_key]["scale"]
    percent = ((value - scale_low) / (scale_high - scale_low)) * 100
    return int(max(0, min(100, percent)))


def risk_score(error, threshold):
    if threshold <= 0:
        return 0
    return int(max(0, min(100, round((error / (threshold * 2)) * 100))))


def risk_label(score):
    if score < 35:
        return "Low Risk", "badge-normal"
    if score < 70:
        return "Moderate Risk", "badge-borderline"
    return "High Risk", "badge-abnormal"


def render_badge(text, css_class):
    st.markdown(
        f'<span class="badge {css_class}">{text}</span>',
        unsafe_allow_html=True,
    )


def section_title(icon, title, tooltip):
    icon_html = f"<span>{icon}</span>" if icon else ""
    tooltip_html = (
        '<span class="info-tip" tabindex="0">i'
        f'<span class="tooltip-text">{escape(tooltip)}</span></span>'
    )
    st.markdown(
        f'<div class="section-title">{icon_html}<span>{title}</span>{tooltip_html}</div>',
        unsafe_allow_html=True,
    )


def render_vital_card(title, normal_text, tooltip_key, input_key, **input_kwargs):
    with st.container(border=True):
        st.markdown(f'<div class="term-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="term-help">{normal_text}</div>', unsafe_allow_html=True)
        value = st.number_input(
            label=title,
            key=input_key,
            label_visibility="collapsed",
            help=TOOLTIPS[tooltip_key],
            **input_kwargs,
        )
        return value


def render_range_check(label, value, range_key, decimals=0):
    status, css_class = classify_value(value, range_key)
    unit = VITAL_RANGES[range_key]["unit"]
    low, high = VITAL_RANGES[range_key]["normal"]
    st.markdown(f'<div class="term-title">{label}</div>', unsafe_allow_html=True)
    render_badge(status, css_class)
    st.progress(progress_percent(value, range_key))
    st.caption(
        f"Current: {format_value(value, decimals)} {unit} | Normal: "
        f"{format_value(low, decimals)}-{format_value(high, decimals)} {unit}"
    )


def render_result(result):
    error = result["error"]
    threshold = result["threshold"]
    is_high_risk = result["is_high_risk"]
    score = risk_score(error, threshold)
    label, badge_class = risk_label(score)

    if is_high_risk:
        card_class = "result-alert"
        title = "⚠️ Unusual Pattern Detected"
        explanation = (
            "The entered vital signs differ considerably from normal physiological "
            "patterns learned by the AI model."
        )
        recommendations = [
            "🔄 Repeat the measurement after resting.",
            "🚨 If symptoms such as chest pain, shortness of breath, or dizziness occur, seek immediate medical attention.",
            "👨‍⚕️ Consult a healthcare professional."
        ]
    else:
        card_class = "result-normal"
        title = "✅ Normal Pattern"
        explanation = (
            "Your entered vital signs closely match the healthy patterns learned by "
            "the AI model."
        )
        recommendations = [
            "💪 Continue maintaining a healthy lifestyle.",
            "📊 Monitor your vital signs regularly.",
            "👨‍⚕️ Seek medical advice if symptoms develop."
        ]

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-kicker">Overall Status</div>
            <div class="result-title">{title}</div>
            <div class="result-copy">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    meter_col, advice_col = st.columns([0.95, 1.05], gap="large")
    with meter_col:
        section_title("", "AI Confidence Meter", TOOLTIPS["ai_confidence"])
        st.metric("AI Risk Score", f"{score}%", label, help=TOOLTIPS["ai_risk"])
        render_badge(label, badge_class)
        st.progress(score)
        st.caption("🟢 Green = Lower risk | 🟡 Yellow = Moderate risk | 🔴 Red = Higher risk")

    with advice_col:
        section_title("", "Recommendations", TOOLTIPS["recommendations"])
        for item in recommendations:
            st.write(item)

    section_title("", "Advanced Analysis", TOOLTIPS["advanced"])
    with st.expander("🔬 Show technical details", expanded=False):
        adv_col_1, adv_col_2, adv_col_3, adv_col_4 = st.columns(4)
        adv_col_1.metric("Reconstruction Error", f"{error:.8f}", help=TOOLTIPS["reconstruction_error"])
        adv_col_2.metric("Threshold", f"{threshold:.8f}", help=TOOLTIPS["threshold"])
        adv_col_3.metric("MAP", f"{result['map_val']:.1f} mmHg", help=TOOLTIPS["map"])
        adv_col_4.metric("Pulse Pressure", f"{result['pulse_pressure']:.1f} mmHg", help=TOOLTIPS["pulse_pressure"])
        st.markdown("**Normalized Features**")
        st.dataframe(result["scaled_features"], width="stretch", hide_index=True)


@st.dialog("📊 Results Dashboard", width="large")
def show_result_dialog():
    render_result(st.session_state["last_result"])
    if st.button("✕ Close", width="stretch"):
        st.session_state["show_result_dialog"] = False
        st.rerun()


st.markdown(
    """
    <div class="hero">
        <h1>❤️ AI Heart Pattern Checker</h1>
        <p>Enter your vital signs to analyse whether your current heart pattern appears normal or unusual.</p>
        <div class="research-note">
            🔬 This system is intended for research purposes only and does not provide medical diagnosis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section_title("", "Your Vital Signs", TOOLTIPS["your_vitals"])
vital_col_1, vital_col_2 = st.columns(2, gap="large")

with vital_col_1:
    hr = render_vital_card(
        "Heart Rate",
        "Normal resting heart rate: 60-100 bpm",
        "heart_rate",
        "heart_rate",
        min_value=40,
        max_value=200,
        value=75,
        step=1,
    )
    sbp = render_vital_card(
        "Systolic Blood Pressure",
        "Normal: 90-120 mmHg",
        "systolic",
        "systolic_bp",
        min_value=80,
        max_value=200,
        value=120,
        step=1,
    )

with vital_col_2:
    temp = render_vital_card(
        "Body Temperature",
        "Normal: 36.1-37.2 C",
        "temperature",
        "temperature",
        min_value=35.0,
        max_value=42.0,
        value=36.8,
        step=0.1,
    )
    dbp = render_vital_card(
        "Diastolic Blood Pressure",
        "Normal: 60-80 mmHg",
        "diastolic",
        "diastolic_bp",
        min_value=50,
        max_value=130,
        value=80,
        step=1,
    )
map_val = (sbp + 2 * dbp) / 3
pulse_pressure = sbp - dbp

button_left, button_mid, button_right = st.columns([0.18, 0.64, 0.18])
with button_mid:
    analyze = st.button("🚀 Analyse Heart Pattern", width="stretch")

if analyze:
    try:
        with st.status("🔍 Analysing your vital signs...", expanded=True) as status:
            st.write("🫀 Checking cardiovascular pattern...")
            time.sleep(0.25)
            st.write("🧮 Calculating anomaly score...")
            error, threshold, is_high_risk, scaled_features = predict_risk(
                hr,
                temp,
                sbp,
                dbp,
                map_val,
                pulse_pressure,
            )
            time.sleep(0.25)
            status.update(label="✅ Analysis complete", state="complete", expanded=False)

        st.session_state["last_result"] = {
            "error": error,
            "threshold": threshold,
            "is_high_risk": is_high_risk,
            "scaled_features": scaled_features,
            "map_val": map_val,
            "pulse_pressure": pulse_pressure,
        }
        st.session_state["show_result_dialog"] = True
    except Exception as e:
        st.error(
            "❌ Error loading AI model: "
            f"{e}. Please ensure final_champion_ae.keras, "
            "unified_minmax_scaler.pkl, and ae_threshold.json are in 05_Web_Prototype."
        )

if st.session_state.get("show_result_dialog") and "last_result" in st.session_state:
    show_result_dialog()

section_title("", "Automatically Calculated", TOOLTIPS["calculated"])
calc_col_1, calc_col_2 = st.columns(2, gap="large")
with calc_col_1:
    st.metric(
        "Mean Arterial Pressure (MAP)",
        f"{map_val:.0f} mmHg",
        "Normal: 70-100 mmHg",
        help=TOOLTIPS["map"],
    )
    render_badge(*classify_value(map_val, "map"))
with calc_col_2:
    st.metric(
        "Pulse Pressure",
        f"{pulse_pressure:.0f} mmHg",
        "Normal: 30-50 mmHg",
        help=TOOLTIPS["pulse_pressure"],
    )
    render_badge(*classify_value(pulse_pressure, "pulse_pressure"))

section_title("", "Health Range Overview", TOOLTIPS["range_overview"])
range_col_1, range_col_2, range_col_3 = st.columns(3, gap="large")
with range_col_1:
    render_range_check("Heart Rate", hr, "heart_rate")
    render_range_check("Body Temperature", temp, "temperature", decimals=1)
with range_col_2:
    render_range_check("Systolic Blood Pressure", sbp, "systolic")
    render_range_check("Diastolic Blood Pressure", dbp, "diastolic")
with range_col_3:
    render_range_check("MAP", map_val, "map")
    render_range_check("Pulse Pressure", pulse_pressure, "pulse_pressure")
