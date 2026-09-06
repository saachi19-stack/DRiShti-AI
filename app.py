import os
import sys
import json
import base64
from datetime import datetime

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.pdfgen import canvas

# ============================================================
# MODEL PACKAGE LOCATION
# ============================================================
# inference.py, metadata.json, class_names.json and
# drishti_ai_efficientnet.keras all live together inside
# the drishti_ai_explanation/ folder, not next to app.py.

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "DRiShti_AI_explanation")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from DRiShti_AI_explanation.inference import DrishtiAI


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DRiShti-AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_NAME = "DRiShti-AI"
TAGLINE = "Explainable AI for Diabetic Retinopathy Screening in Rural India"

REPORT_DIR = "reports"
HISTORY_FILE = "screening_history.json"

os.makedirs(REPORT_DIR, exist_ok=True)

# Severity -> color mapping (traffic-light style, used everywhere:
# badges, probability bars, chart colors). Purely a display layer —
# does not touch model outputs.
SEVERITY_COLORS = {
    "No DR": "#27AE60",
    "Mild": "#A9CB3C",
    "Moderate": "#F5B041",
    "Severe": "#E67E22",
    "Proliferative DR": "#E74C3C",
}

PRIORITY_COLORS = {
    "ROUTINE": "#27AE60",
    "LOW": "#27AE60",
    "PRIORITY": "#F5B041",
    "MEDIUM": "#F5B041",
    "URGENT": "#E74C3C",
    "HIGH": "#E74C3C",
}


def color_for_priority(priority_text):
    priority_text = (priority_text or "").upper()
    for key, color in PRIORITY_COLORS.items():
        if key in priority_text:
            return color
    return "#7F8C8D"


# ============================================================
# CUSTOM CSS — vibrant medical theme
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #F4F9FA 0%, #EDF6F6 100%);
}

/* ---------- Hero banner ---------- */
.hero-banner {
    background: linear-gradient(120deg, #0F5C5C 0%, #146B8C 55%, #5B4FCF 100%);
    padding: 34px 40px;
    border-radius: 22px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(15, 92, 92, 0.25);
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    font-size: 17px;
    font-weight: 400;
    opacity: 0.92;
    margin-top: 6px;
}

.hero-pill {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.4);
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 14px;
    margin-right: 8px;
}

/* ---------- Cards ---------- */
.app-card {
    background: white;
    border-radius: 18px;
    padding: 24px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(20, 60, 70, 0.08);
    border: 1px solid #E7F0F0;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #0F5C5C;
    margin-top: 0px;
    margin-bottom: 14px;
    border-left: 5px solid #5B4FCF;
    padding-left: 12px;
}

.small-note {
    font-size: 13px;
    color: #6B7A7A;
}

/* ---------- Badges ---------- */
.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    color: white;
    font-weight: 700;
    font-size: 14px;
}

/* ---------- Probability bars ---------- */
.prob-row {
    margin-bottom: 12px;
}

.prob-label {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 4px;
    color: #234;
}

.prob-track {
    background: #EAF1F1;
    border-radius: 999px;
    height: 16px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 999px;
}

/* ---------- Workflow steps ---------- */
.step-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #F4FBFB;
    border: 1px solid #DCEEEE;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14px;
}

.step-num {
    background: #0F5C5C;
    color: white;
    width: 26px;
    height: 26px;
    min-width: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 13px;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(120deg, #0F5C5C, #146B8C);
    color: white !important;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    padding: 8px 18px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(120deg, #0C4747, #0F5A72);
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(15,92,92,0.3);
}

.stDownloadButton > button {
    background: linear-gradient(120deg, #E8785A, #D65D40) !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 14px 10px;
    border: 1px solid #E7F0F0;
    box-shadow: 0 2px 8px rgba(20,60,70,0.05);
}

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #7A8A8A;
    font-size: 13px;
    margin-top: 30px;
    padding: 16px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(record):
    history = load_history()
    history.append(record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


# ============================================================
# UI HELPERS
# ============================================================

def severity_badge(class_name):
    color = SEVERITY_COLORS.get(class_name, "#7F8C8D")
    return f'<span class="badge" style="background:{color};">{class_name}</span>'


def priority_badge(priority_text):
    color = color_for_priority(priority_text)
    return f'<span class="badge" style="background:{color};">{priority_text}</span>'


def render_probability_bars(probability_data):
    for class_name, probability in probability_data.items():
        color = SEVERITY_COLORS.get(class_name, "#5B4FCF")
        pct = probability * 100
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label">
                    <span>{class_name}</span>
                    <span>{pct:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{pct}%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# IMAGE QUALITY VISUALIZATION
# ============================================================

def create_quality_chart(quality):
    labels = ["Brightness", "Contrast", "Sharpness"]
    values = [
        quality["brightness_score"],
        quality["contrast_score"],
        quality["sharpness_score"]
    ]

    bar_colors = []
    for v in values:
        if v >= 70:
            bar_colors.append("#27AE60")
        elif v >= 40:
            bar_colors.append("#F5B041")
        else:
            bar_colors.append("#E74C3C")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("Retinal Image Quality Assessment")
    ax.spines[['top', 'right']].set_visible(False)

    for i, value in enumerate(values):
        ax.text(i, value + 2, f"{value:.1f}", ha="center", fontweight="bold")

    plt.tight_layout()
    return fig


def create_history_charts(history):
    class_counts = {}
    priority_counts = {}

    for item in history:
        cls = item.get("prediction", "Unknown")
        pr = item.get("referral_priority", "Unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1
        priority_counts[pr] = priority_counts.get(pr, 0) + 1

    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    if class_counts:
        labels = list(class_counts.keys())
        values = list(class_counts.values())
        bar_colors = [SEVERITY_COLORS.get(l, "#5B4FCF") for l in labels]
        ax1.bar(labels, values, color=bar_colors)
        ax1.set_title("Screenings by Severity Class")
        ax1.set_ylabel("Count")
        plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
    else:
        ax1.text(0.5, 0.5, "No data yet", ha="center", va="center")
        ax1.axis("off")
    ax1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    if priority_counts:
        labels = list(priority_counts.keys())
        values = list(priority_counts.values())
        bar_colors = [color_for_priority(l) for l in labels]
        ax2.bar(labels, values, color=bar_colors)
        ax2.set_title("Screenings by Referral Priority")
        ax2.set_ylabel("Count")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    else:
        ax2.text(0.5, 0.5, "No data yet", ha="center", va="center")
        ax2.axis("off")
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    return fig1, fig2


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(image, heatmap):
    image = np.array(image.convert("RGB"))
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.60, colored_heatmap, 0.40, 0)
    return overlay


# ============================================================
# PDF REPORT
# ============================================================

def wrap_paragraph(c, text, x, y, max_len, font="Helvetica", size=10, leading=16):
    c.setFont(font, size)
    words = text.split()
    current_line = ""
    for word in words:
        test_line = current_line + " " + word
        if len(test_line) > max_len:
            c.drawString(x, y, current_line.strip())
            y -= leading
            current_line = word
        else:
            current_line = test_line
    if current_line:
        c.drawString(x, y, current_line.strip())
        y -= leading
    return y


def generate_pdf(patient_name, patient_id, age, sex, result, image_quality):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_name = patient_name.replace(" ", "_")
    filename = os.path.join(
        REPORT_DIR,
        f"DRiShti_AI_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 50

    # Colored header band
    c.setFillColor(rl_colors.HexColor("#0F5C5C"))
    c.rect(0, height - 90, width, 90, fill=1, stroke=0)

    c.setFillColor(rl_colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 45, "DRiShti-AI")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 65, TAGLINE)

    c.setFillColor(rl_colors.black)
    y = height - 115

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Patient Information")
    y -= 25

    c.setFont("Helvetica", 11)
    for line in [
        f"Name: {patient_name}",
        f"Patient ID: {patient_id}",
        f"Age: {age}",
        f"Sex: {sex}",
        f"Screening Time: {timestamp}"
    ]:
        c.drawString(60, y, line)
        y -= 18
    y -= 15

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "AI Screening Result")
    y -= 25

    prediction = result["prediction"]
    c.setFont("Helvetica", 11)
    for line in [
        f"Predicted Class: {prediction['class_name']}",
        f"AI Confidence: {prediction['confidence'] * 100:.2f}%",
        f"Image Quality: {image_quality['status']}",
        f"Image Quality Score: {image_quality['percentage']:.2f}%",
        f"AI Reliability: {result['final_trust']['status']}",
        f"Reliability Score: {result['final_trust']['percentage']:.2f}%",
        f"Referral Priority: {result['referral']['priority']}",
        f"Urgency: {result['referral']['urgency_level']}"
    ]:
        c.drawString(60, y, line)
        y -= 18
    y -= 15

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Recommended Action")
    y -= 25
    y = wrap_paragraph(c, result["referral"]["action"], 60, y, max_len=85)
    y -= 10

    explanation = result.get("explanation")
    if explanation:
        if y < 150:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "AI Explanation (Plain Language)")
        y -= 22

        simple = explanation.get("simple", {})
        y = wrap_paragraph(c, simple.get("summary", ""), 60, y, max_len=90, size=10, leading=15)
        y -= 8
        y = wrap_paragraph(
            c, simple.get("recommendation_explanation", ""), 60, y,
            max_len=90, font="Helvetica-Oblique", size=9, leading=14
        )
        y -= 15

        technical = explanation.get("technical", {})
        decision_basis = technical.get("decision_basis", "")
        if decision_basis:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y, "Decision Basis:")
            y -= 16
            y = wrap_paragraph(c, decision_basis, 60, y, max_len=95, size=9, leading=13)

    y -= 15
    if y < 90:
        c.showPage()
        y = height - 50

    disclaimer = (
        "Disclaimer: DRiShti-AI is an AI-assisted screening prototype. "
        "It does not replace professional medical examination, "
        "clinical judgment, or diagnosis by a qualified healthcare professional."
    )
    y = wrap_paragraph(c, disclaimer, 50, y, max_len=100, font="Helvetica-Oblique", size=8, leading=13)

    c.save()
    return filename


# ============================================================
# LOAD AI MODEL
# ============================================================

@st.cache_resource
def load_ai_model():
    return DrishtiAI(MODEL_DIR)


try:
    ai = load_ai_model()
    model_loaded = True
except Exception as e:
    ai = None
    model_loaded = False
    model_error = str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 👁️ DRiShti-AI")
    st.caption("AI-assisted retinal screening, built for rural India")

    st.divider()

    st.markdown("### 🌐 Connectivity")
    connectivity = st.selectbox(
        "Current mode",
        ["Online", "Limited Connectivity", "Offline / Local"]
    )
    st.caption(
        "The full screening pipeline — capture, quality check, "
        "classification, Grad-CAM, referral, report — runs locally. "
        "Only history sync needs connectivity."
    )

    st.divider()

    st.markdown("### 🔄 Screening Workflow")
    workflow_steps = [
        "Image Capture / Upload",
        "Image Quality Check",
        "AI DR Screening (5-class)",
        "Explainable AI (Grad-CAM)",
        "Referral Prioritization",
    ]
    for i, step in enumerate(workflow_steps, start=1):
        st.markdown(
            f"""<div class="step-chip"><div class="step-num">{i}</div>{step}</div>""",
            unsafe_allow_html=True
        )

    st.divider()

    with st.expander("ℹ️ About this prototype"):
        st.write(
            "**DRiShti-AI** screens retinal fundus images for diabetic "
            "retinopathy across 5 severity levels, explains its reasoning "
            "with Grad-CAM heatmaps, and prioritizes referrals — designed "
            "for use by non-specialist health workers at rural PHCs, "
            "with or without internet access."
        )
        st.caption("SIH26038 • Smart India Hackathon 2026 • Team: [Add team name]")

    with st.expander("🩺 What is Diabetic Retinopathy?"):
        st.write(
            "Diabetic retinopathy (DR) is damage to the retina's blood "
            "vessels caused by long-term diabetes. It's often symptomless "
            "in early stages — by the time vision problems appear, damage "
            "can already be advanced. Early screening allows early treatment."
        )

    st.divider()

    if model_loaded:
        st.success("✅ AI Model Loaded")
    else:
        st.error("❌ AI Model Failed to Load")
        st.caption(model_error)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">👁️ DRiShti-AI</div>
        <div class="hero-subtitle">{TAGLINE}</div>
        <div>
            <span class="hero-pill">🩺 5-Class Severity</span>
            <span class="hero-pill">🔥 Grad-CAM Explainable</span>
            <span class="hero-pill">📡 Offline-First</span>
            <span class="hero-pill">🚨 Trust-Aware Referral</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

tab_screen, tab_dashboard, tab_history, tab_about = st.tabs(
    ["🔍 New Screening", "📊 Dashboard", "📋 History", "❓ About & FAQ"]
)


# ============================================================
# TAB 1 — NEW SCREENING
# ============================================================

with tab_screen:

    if "analysis_result" in st.session_state:
        reset_col, _ = st.columns([1, 3])
        with reset_col:
            if st.button("🔄 Start New Screening"):
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("analysis_image", None)
                st.rerun()

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        patient_name = st.text_input("Patient Name", placeholder="Enter name")
    with col2:
        patient_id = st.text_input("Patient ID", placeholder="e.g. DR-001")
    with col3:
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
    with col4:
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📷 Retinal Image</div>', unsafe_allow_html=True)
    st.write(
        "Upload a clear, well-lit fundus photograph. The AI will check "
        "image quality first — blurry or poorly lit images will be "
        "flagged for a retake before screening runs."
    )

    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image",
        type=["jpg", "jpeg", "png"]
    )

    sample_dir = "sample_images"
    if os.path.isdir(sample_dir):
        sample_files = [
            f for f in os.listdir(sample_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if sample_files:
            st.markdown("**Or try a sample image:**")
            selected_sample = st.selectbox("Sample images", ["None"] + sample_files)
            if selected_sample != "None" and not uploaded_file:
                uploaded_file = open(os.path.join(sample_dir, selected_sample), "rb")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Retinal Image", use_container_width=True)
        with col2:
            st.info(
                "Image received successfully. The AI will first assess "
                "image quality before performing diabetic retinopathy screening."
            )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        analyze_button = st.button(
            "🔍 Analyze Retinal Image", type="primary", use_container_width=True
        )

        if analyze_button:
            if not model_loaded:
                st.error("AI model could not be loaded.")
                st.stop()

            temp_path = os.path.join(REPORT_DIR, "temp_screening_image.png")
            image.save(temp_path)

            try:
                with st.spinner("Running DRiShti-AI screening..."):
                    result = ai.analyze(temp_path, generate_gradcam=True)

                st.session_state["analysis_result"] = result
                st.session_state["analysis_image"] = image

            except Exception as e:
                st.error("Analysis failed.")
                st.code(str(e))

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "analysis_result" in st.session_state:

        result = st.session_state["analysis_result"]
        image = st.session_state["analysis_image"]

        prediction = result["prediction"]
        uncertainty = result["uncertainty"]
        model_trust = result["model_trust"]
        image_quality = result["image_quality"]
        final_trust = result["final_trust"]
        referral = result["referral"]
        explanation = result.get("explanation", {})
        simple_explanation = explanation.get("simple", {})
        technical_explanation = explanation.get("technical", {})

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 AI Screening Result</div>', unsafe_allow_html=True)

        st.markdown(
            f"Predicted Severity: {severity_badge(prediction['class_name'])} "
            f"&nbsp;&nbsp; Referral: {priority_badge(referral['priority'])}",
            unsafe_allow_html=True
        )
        st.write("")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AI Confidence", f"{prediction['confidence'] * 100:.1f}%")
        with col2:
            st.metric("AI Reliability", final_trust["status"])
        with col3:
            st.metric("Reliability Score", f"{final_trust['percentage']:.1f}%")
        with col4:
            st.metric("Image Quality", image_quality["status"])
        st.markdown('</div>', unsafe_allow_html=True)

        if simple_explanation:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">🗣️ What This Means (Plain Language)</div>',
                unsafe_allow_html=True
            )
            st.info(simple_explanation.get("summary", ""))

            reason_cols = st.columns(2)
            with reason_cols[0]:
                st.write(f"🔹 {simple_explanation.get('confidence_reason', '')}")
                st.write(f"🔹 {simple_explanation.get('uncertainty_reason', '')}")
            with reason_cols[1]:
                st.write(f"🔹 {simple_explanation.get('image_quality_reason', '')}")
                st.write(f"🔹 {simple_explanation.get('trust_reason', '')}")

            st.success(simple_explanation.get("recommendation_explanation", ""))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Class Probabilities</div>', unsafe_allow_html=True)
        render_probability_bars(prediction["all_probabilities"])

        prob_summary_lines = [
            f"{cls}: {p*100:.2f}%" for cls, p in prediction["all_probabilities"].items()
        ]
        prob_text = (
            f"DRiShti-AI Screening — Class Probabilities\n"
            f"Patient: {patient_name or 'Not provided'} ({patient_id or 'N/A'})\n"
            f"Predicted: {prediction['class_name']} ({prediction['confidence']*100:.2f}%)\n\n"
            + "\n".join(prob_summary_lines)
        )
        st.download_button(
            "📥 Download Probabilities (.txt)",
            data=prob_text,
            file_name="drishti_ai_probabilities.txt",
            mime="text/plain"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📷 Image Quality Assessment</div>', unsafe_allow_html=True)
        quality_col1, quality_col2 = st.columns(2)
        with quality_col1:
            st.metric("Overall Quality", f"{image_quality['percentage']:.1f}%")
            st.write(f"Brightness: {image_quality['brightness']:.2f}")
            st.write(f"Contrast: {image_quality['contrast']:.2f}")
            st.write(f"Sharpness: {image_quality['sharpness']:.2f}")
        with quality_col2:
            fig = create_quality_chart(image_quality)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧮 Prediction Uncertainty</div>', unsafe_allow_html=True)
        u1, u2, u3 = st.columns(3)
        with u1:
            st.metric("Confidence", f"{uncertainty['confidence'] * 100:.1f}%")
        with u2:
            st.metric("Prediction Margin", f"{uncertainty['margin'] * 100:.1f}%")
        with u3:
            st.metric("Normalized Entropy", f"{uncertainty['entropy']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔥 Explainable AI — Grad-CAM</div>', unsafe_allow_html=True)
        st.write(
            "The highlighted regions represent image areas that "
            "contributed strongly to the model's prediction."
        )

        heatmap = result.get("gradcam_heatmap")
        if heatmap is not None:
            overlay = create_gradcam_overlay(image, heatmap)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Fundus Image", use_container_width=True)
            with col2:
                st.image(overlay, caption="Grad-CAM Explainability Heatmap", use_container_width=True)
        else:
            st.warning("Grad-CAM could not be generated.")

        with st.expander("ℹ️ What does the Grad-CAM heatmap mean?"):
            st.write(
                "Grad-CAM (Gradient-weighted Class Activation Mapping) shows "
                "which regions of the retina most influenced the AI's decision. "
                "Warmer colors (red/yellow) mark areas the model focused on most; "
                "cooler colors (blue) mark areas that mattered less. This lets a "
                "health worker or doctor visually sanity-check the AI's reasoning "
                "instead of trusting a black-box score."
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if technical_explanation:
            with st.expander("🧑‍⚕️ Technical Decision Explanation (for clinical review)"):
                tech_pred = technical_explanation.get("prediction", {})
                tech_uncertainty = technical_explanation.get("uncertainty", {})
                tech_quality = technical_explanation.get("image_quality", {})
                tech_model_trust = technical_explanation.get("model_trust", {})
                tech_final_trust = technical_explanation.get("final_trust", {})
                tech_referral = technical_explanation.get("referral", {})

                st.write(
                    f"**Prediction:** {tech_pred.get('class', '')} "
                    f"({tech_pred.get('confidence_percentage', 0):.2f}%) — "
                    f"{tech_pred.get('interpretation', '')}"
                )
                st.write(
                    f"**Prediction Margin:** "
                    f"{tech_uncertainty.get('prediction_margin_percentage', 0):.2f}% — "
                    f"{tech_uncertainty.get('margin_interpretation', '')}"
                )
                st.write(
                    f"**Normalized Entropy:** "
                    f"{tech_uncertainty.get('normalized_entropy', 0):.3f} — "
                    f"{tech_uncertainty.get('entropy_interpretation', '')}"
                )
                st.write(
                    f"**Image Quality:** {tech_quality.get('score_percentage', 0):.2f}% "
                    f"({tech_quality.get('status', '')})"
                )
                st.write(
                    f"**Model Trust:** {tech_model_trust.get('score_percentage', 0):.2f}% "
                    f"({tech_model_trust.get('status', '')})"
                )
                st.write(
                    f"**Final Trust:** {tech_final_trust.get('score_percentage', 0):.2f}% "
                    f"({tech_final_trust.get('status', '')})"
                )
                st.write(
                    f"**Referral:** {tech_referral.get('priority', '')} — "
                    f"Urgency: {tech_referral.get('urgency_level', '')}"
                )
                st.write(f"**Decision Basis:** {technical_explanation.get('decision_basis', '')}")

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚨 Risk & Referral Prioritization</div>', unsafe_allow_html=True)
        st.markdown(
            f"**Referral Priority:** {priority_badge(referral['priority'])}",
            unsafe_allow_html=True
        )
        r1, r2 = st.columns(2)
        with r1:
            st.write(f"**Urgency Level:** {referral['urgency_level']}")
            st.write(f"**Reason:** {referral['reason']}")
        with r2:
            st.write(f"**Recommended Action:** {referral['action']}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🛡️ AI Reliability Assessment</div>', unsafe_allow_html=True)
        st.write(f"Model Trust: **{model_trust['status']}** ({model_trust['percentage']:.2f}%)")
        st.write(f"Final Reliability: **{final_trust['percentage']:.2f}%**")
        st.info(final_trust["recommendation"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📄 Screening Report</div>', unsafe_allow_html=True)

        report_col1, report_col2 = st.columns(2)
        with report_col1:
            st.write(f"**Patient:** {patient_name or 'Not provided'}")
            st.write(f"**Patient ID:** {patient_id or 'Not provided'}")
            st.write(f"**Predicted Severity:** {prediction['class_name']}")
            st.write(f"**Confidence:** {prediction['confidence'] * 100:.2f}%")
        with report_col2:
            st.write(f"**Referral:** {referral['priority']}")
            st.write(f"**Urgency:** {referral['urgency_level']}")
            st.write(f"**AI Reliability:** {final_trust['status']}")
            st.write(f"**Connectivity:** {connectivity}")

        record = {
            "timestamp": datetime.now().isoformat(),
            "patient_name": patient_name,
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "prediction": prediction["class_name"],
            "confidence": prediction["confidence"],
            "image_quality": image_quality["status"],
            "image_quality_score": image_quality["percentage"],
            "ai_reliability": final_trust["status"],
            "reliability_score": final_trust["percentage"],
            "referral_priority": referral["priority"],
            "urgency": referral["urgency_level"],
            "decision_basis": technical_explanation.get("decision_basis", ""),
            "connectivity": connectivity
        }

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("💾 Save Screening Record", use_container_width=True):
                save_history(record)
                st.success("Screening record saved locally.")

        with btn_col2:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                pdf_path = generate_pdf(
                    patient_name or "Unknown", patient_id or "N/A",
                    age, sex, result, image_quality
                )
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state["pdf_name"] = os.path.basename(pdf_path)

        if "pdf_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Screening Report",
                data=st.session_state["pdf_bytes"],
                file_name=st.session_state["pdf_name"],
                mime="application/pdf",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚀 Ready to Screen</div>', unsafe_allow_html=True)
        st.write(
            "Fill in patient details, upload a fundus image above (or pick a "
            "sample image), then click **Analyze Retinal Image** to run the "
            "full pipeline: quality check → 5-class DR screening → Grad-CAM "
            "explanation → trust scoring → referral prioritization."
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# TAB 2 — DASHBOARD
# ============================================================

with tab_dashboard:

    history = load_history()
    total_screened = len(history)
    high_priority = sum(
        1 for item in history
        if "PRIORITY" in item.get("referral_priority", "")
        or "URGENT" in item.get("referral_priority", "")
    )
    urgent_cases = sum(
        1 for item in history if "URGENT" in item.get("referral_priority", "")
    )

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Screening Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Screened", total_screened)
    with col2:
        st.metric("Priority Cases", high_priority)
    with col3:
        st.metric("Urgent Cases", urgent_cases)
    with col4:
        st.metric("Local Records", total_screened)

    if st.button("🔄 Refresh Dashboard"):
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Trends</div>', unsafe_allow_html=True)
    fig1, fig2 = create_history_charts(history)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
    with chart_col2:
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    st.caption(
        "Charts are generated from your saved local screening history. "
        "Save a screening record after analysis to see it reflected here."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# TAB 3 — HISTORY
# ============================================================

with tab_history:

    history = load_history()

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Local Screening History</div>', unsafe_allow_html=True)

    if history:
        st.caption(f"Showing the {min(10, len(history))} most recent of {len(history)} saved records.")

        for item in reversed(history[-10:]):
            badge_html = severity_badge(item.get("prediction", "N/A"))
            with st.expander(f"{item.get('patient_name', 'Unknown')} — {item.get('prediction', 'N/A')}"):
                st.markdown(badge_html, unsafe_allow_html=True)
                st.write(f"**Date:** {item.get('timestamp', '')}")
                st.write(f"**Patient ID:** {item.get('patient_id', 'N/A')}")
                st.write(f"**Confidence:** {item.get('confidence', 0) * 100:.2f}%")
                st.markdown(
                    f"**Referral:** {priority_badge(item.get('referral_priority', 'N/A'))}",
                    unsafe_allow_html=True
                )
                if item.get("decision_basis"):
                    st.write(f"**Decision Basis:** {item.get('decision_basis', '')}")
                st.write(f"**Connectivity:** {item.get('connectivity', 'N/A')}")

        st.divider()
        confirm_clear = st.checkbox("I understand this will permanently delete all local history")
        if st.button("🗑️ Clear All History", disabled=not confirm_clear):
            clear_history()
            st.success("Screening history cleared.")
            st.rerun()
    else:
        st.info("No screening records saved yet. Analyze an image and click 'Save Screening Record' to start building history.")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# TAB 4 — ABOUT & FAQ
# ============================================================

with tab_about:

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 About DRiShti-AI</div>', unsafe_allow_html=True)
    st.write(
        "**DRiShti-AI** is an AI-assisted screening tool built for the "
        "Smart India Hackathon 2026 (Problem Statement SIH26038), addressing "
        "diabetic retinopathy screening in rural India where ophthalmologist "
        "access is limited. It analyzes retinal fundus images, classifies "
        "diabetic retinopathy severity into 5 classes, explains its "
        "reasoning visually, and recommends referral urgency — all "
        "designed to run locally, without needing constant internet access."
    )
    st.write("**Organization:** MathWorks &nbsp;|&nbsp; **Theme:** Healthcare / MedTech")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 The 5 Severity Classes</div>', unsafe_allow_html=True)
    class_descriptions = {
        "No DR": "No visible retinal damage.",
        "Mild": "Early, small lesions (microaneurysms) beginning to appear.",
        "Moderate": "More lesions present, but still limited in extent.",
        "Severe": "Extensive lesions with high risk of progression.",
        "Proliferative DR": "Abnormal new blood vessel growth — most advanced, highest urgency.",
    }
    for cls, desc in class_descriptions.items():
        st.markdown(f"{severity_badge(cls)} &nbsp; {desc}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">❓ Frequently Asked Questions</div>', unsafe_allow_html=True)

    with st.expander("Can this replace an ophthalmologist?"):
        st.write(
            "No. DRiShti-AI is explicitly a screening and referral-support "
            "tool. Final diagnosis and treatment decisions remain with a "
            "qualified ophthalmologist — the AI helps prioritize who needs "
            "to see one, and how urgently."
        )

    with st.expander("What happens if the uploaded image is poor quality?"):
        st.write(
            "The image-quality assessment stage flags it before "
            "classification runs, so the health worker is prompted to "
            "recapture the image rather than receiving an unreliable "
            "prediction on a bad image."
        )

    with st.expander("Does this work without internet?"):
        st.write(
            "Yes. The full pipeline — capture, quality check, "
            "classification, Grad-CAM, risk scoring, and report generation "
            "— runs locally on the device. Only syncing screening history "
            "to a central system needs connectivity, and that sync is "
            "opportunistic, not required."
        )

    with st.expander("What happens to patient data?"):
        st.write(
            "Screening records are stored locally by default in this "
            "prototype. Encryption, access control, and formal "
            "anonymization are planned safeguards for real deployment "
            "and are not yet implemented at the prototype stage."
        )

    with st.expander("How is this different from a generic DR classifier?"):
        st.write(
            "Many generic tools focus primarily on binary classification "
            "(DR / no DR). DRiShti-AI combines 5-class severity, an "
            "image-quality gate, Grad-CAM explainability, trust-aware "
            "referral prioritization, and an offline-first workflow — "
            "built around the full screening-to-referral journey, not "
            "just a single prediction."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Disclaimer & Limitations</div>', unsafe_allow_html=True)
    st.warning(
        "DRiShti-AI is an AI-assisted screening prototype. It does not "
        "replace ophthalmologist diagnosis, clinical judgment, or "
        "professional medical examination. Model performance depends on "
        "training data diversity, and this prototype has not undergone "
        "formal clinical validation."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DRiShti-AI • SIH26038 • Smart India Hackathon 2026<br>
        Explainable AI for Diabetic Retinopathy Screening in Rural India
    </div>
    """,
    unsafe_allow_html=True
)