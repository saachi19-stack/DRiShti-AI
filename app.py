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
from reportlab.pdfgen import canvas

# ============================================================
# MODEL PACKAGE LOCATION
# ============================================================
# inference.py, metadata.json, class_names.json and
# drishti_ai_efficientnet.keras all live together inside
# the drishti_ai_explanation/ folder, not next to app.py.
# Add that folder to sys.path so "from inference import DrishtiAI"
# can find it, and point DrishtiAI at the same folder to load
# its model/metadata/class files.

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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
}

.result-box {
    padding: 22px;
    border-radius: 15px;
    margin: 10px 0;
    border: 1px solid #ddd;
}

.metric-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #ddd;
    text-align: center;
}

.small-note {
    font-size: 13px;
    color: #666;
}

.explanation-box {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #cfe3e3;
    background-color: #f4fbfb;
    margin: 10px 0;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


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


# ============================================================
# IMAGE QUALITY VISUALIZATION
# ============================================================

def create_quality_chart(quality):

    labels = [
        "Brightness",
        "Contrast",
        "Sharpness"
    ]

    values = [
        quality["brightness_score"],
        quality["contrast_score"],
        quality["sharpness_score"]
    ]

    fig, ax = plt.subplots(figsize=(7, 3))

    ax.bar(labels, values)

    ax.set_ylim(0, 100)

    ax.set_ylabel("Score")

    ax.set_title("Retinal Image Quality Assessment")

    for i, value in enumerate(values):
        ax.text(
            i,
            value + 2,
            f"{value:.1f}",
            ha="center"
        )

    plt.tight_layout()

    return fig


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(image, heatmap):

    image = np.array(image.convert("RGB"))

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image,
        0.60,
        colored_heatmap,
        0.40,
        0
    )

    return overlay


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf(
    patient_name,
    patient_id,
    age,
    sex,
    result,
    image_quality
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    safe_name = patient_name.replace(
        " ",
        "_"
    )

    filename = os.path.join(
        REPORT_DIR,
        f"DRiShti_AI_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    c = canvas.Canvas(
        filename,
        pagesize=A4
    )

    width, height = A4

    y = height - 50

    # Header
    c.setFont(
        "Helvetica-Bold",
        22
    )

    c.drawString(
        50,
        y,
        "DRiShti-AI"
    )

    y -= 25

    c.setFont(
        "Helvetica",
        10
    )

    c.drawString(
        50,
        y,
        TAGLINE
    )

    y -= 40

    # Patient information
    c.setFont(
        "Helvetica-Bold",
        14
    )

    c.drawString(
        50,
        y,
        "Patient Information"
    )

    y -= 25

    c.setFont(
        "Helvetica",
        11
    )

    patient_details = [
        f"Name: {patient_name}",
        f"Patient ID: {patient_id}",
        f"Age: {age}",
        f"Sex: {sex}",
        f"Screening Time: {timestamp}"
    ]

    for line in patient_details:

        c.drawString(
            60,
            y,
            line
        )

        y -= 18

    y -= 15

    # AI result
    c.setFont(
        "Helvetica-Bold",
        14
    )

    c.drawString(
        50,
        y,
        "AI Screening Result"
    )

    y -= 25

    c.setFont(
        "Helvetica",
        11
    )

    prediction = result["prediction"]

    lines = [
        f"Predicted Class: {prediction['class_name']}",
        f"AI Confidence: {prediction['confidence'] * 100:.2f}%",
        f"Image Quality: {image_quality['status']}",
        f"Image Quality Score: {image_quality['percentage']:.2f}%",
        f"AI Reliability: {result['final_trust']['status']}",
        f"Reliability Score: {result['final_trust']['percentage']:.2f}%",
        f"Referral Priority: {result['referral']['priority']}",
        f"Urgency: {result['referral']['urgency_level']}"
    ]

    for line in lines:

        c.drawString(
            60,
            y,
            line
        )

        y -= 18

    y -= 15

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

    # Recommendation
    c.setFont(
        "Helvetica-Bold",
        14
    )

    c.drawString(
        50,
        y,
        "Recommended Action"
    )

    y -= 25

    recommendation = result[
        "referral"
    ]["action"]

    y = wrap_paragraph(
        c,
        recommendation,
        60,
        y,
        max_len=85,
        font="Helvetica",
        size=10,
        leading=16
    )

    y -= 10

    # --------------------------------------------------------
    # EXPLAINABLE AI SUMMARY (new)
    # --------------------------------------------------------

    explanation = result.get("explanation")

    if explanation:

        if y < 150:
            c.showPage()
            y = height - 50

        c.setFont(
            "Helvetica-Bold",
            14
        )

        c.drawString(
            50,
            y,
            "AI Explanation (Plain Language)"
        )

        y -= 22

        simple = explanation.get("simple", {})

        y = wrap_paragraph(
            c,
            simple.get("summary", ""),
            60,
            y,
            max_len=90,
            font="Helvetica",
            size=10,
            leading=15
        )

        y -= 8

        y = wrap_paragraph(
            c,
            simple.get("recommendation_explanation", ""),
            60,
            y,
            max_len=90,
            font="Helvetica-Oblique",
            size=9,
            leading=14
        )

        y -= 15

        technical = explanation.get("technical", {})

        decision_basis = technical.get("decision_basis", "")

        if decision_basis:

            c.setFont(
                "Helvetica-Bold",
                11
            )

            c.drawString(
                50,
                y,
                "Decision Basis:"
            )

            y -= 16

            y = wrap_paragraph(
                c,
                decision_basis,
                60,
                y,
                max_len=95,
                font="Helvetica",
                size=9,
                leading=13
            )

    y -= 15

    if y < 90:
        c.showPage()
        y = height - 50

    # Disclaimer
    c.setFont(
        "Helvetica-Oblique",
        8
    )

    disclaimer = (
        "Disclaimer: DRiShti-AI is an AI-assisted screening prototype. "
        "It does not replace professional medical examination, "
        "clinical judgment, or diagnosis by a qualified healthcare professional."
    )

    y = wrap_paragraph(
        c,
        disclaimer,
        50,
        y,
        max_len=100,
        font="Helvetica-Oblique",
        size=8,
        leading=13
    )

    c.save()

    return filename


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 👁️ DRiShti-AI"
    )

    st.caption(
        "AI-assisted retinal screening"
    )

    st.divider()

    st.markdown(
        "### 🌐 Connectivity"
    )

    connectivity = st.selectbox(
        "Current mode",
        [
            "Online",
            "Limited Connectivity",
            "Offline / Local"
        ]
    )

    st.divider()

    st.markdown(
        "### 🔄 Screening Workflow"
    )

    st.write("1️⃣ Image Capture / Upload")
    st.write("2️⃣ Image Quality Check")
    st.write("3️⃣ AI DR Screening")
    st.write("4️⃣ Explainable AI (Grad-CAM + Plain-Language)")
    st.write("5️⃣ Referral Prioritization")

    st.divider()

    if model_loaded:

        st.success(
            "AI Model Loaded"
        )

    else:

        st.error(
            "AI Model Failed to Load"
        )

        st.caption(
            model_error
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">👁️ DRiShti-AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="subtitle">{TAGLINE}</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

history = load_history()

total_screened = len(history)

high_priority = sum(
    1
    for item in history
    if "PRIORITY" in item.get(
        "referral_priority",
        ""
    )
    or "URGENT" in item.get(
        "referral_priority",
        ""
    )
)

urgent_cases = sum(
    1
    for item in history
    if "URGENT" in item.get(
        "referral_priority",
        ""
    )
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Screened",
        total_screened
    )

with col2:

    st.metric(
        "Priority Cases",
        high_priority
    )

with col3:

    st.metric(
        "Urgent Cases",
        urgent_cases
    )

with col4:

    st.metric(
        "Local Records",
        total_screened
    )


st.divider()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter name"
    )

with col2:

    patient_id = st.text_input(
        "Patient ID",
        placeholder="e.g. DR-001"
    )

with col3:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40
    )

with col4:

    sex = st.selectbox(
        "Sex",
        [
            "Male",
            "Female",
            "Other"
        ]
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📷 Retinal Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Retinal Image",
            use_container_width=True
        )

    with col2:

        st.info(
            "Image received successfully. "
            "The AI will first assess image quality "
            "before performing diabetic retinopathy screening."
        )


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file:

    st.divider()

    analyze_button = st.button(
        "🔍 Analyze Retinal Image",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not model_loaded:

            st.error(
                "AI model could not be loaded."
            )

            st.stop()

        # Save temporary image
        temp_path = os.path.join(
            REPORT_DIR,
            "temp_screening_image.png"
        )

        image.save(
            temp_path
        )

        try:

            with st.spinner(
                "Running DRiShti-AI screening..."
            ):

                result = ai.analyze(
                    temp_path,
                    generate_gradcam=True
                )

            st.session_state[
                "analysis_result"
            ] = result

            st.session_state[
                "analysis_image"
            ] = image

        except Exception as e:

            st.error(
                "Analysis failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# DISPLAY RESULT
# ============================================================

if "analysis_result" in st.session_state:

    result = st.session_state[
        "analysis_result"
    ]

    image = st.session_state[
        "analysis_image"
    ]

    prediction = result[
        "prediction"
    ]

    uncertainty = result[
        "uncertainty"
    ]

    model_trust = result[
        "model_trust"
    ]

    image_quality = result[
        "image_quality"
    ]

    final_trust = result[
        "final_trust"
    ]

    referral = result[
        "referral"
    ]

    explanation = result.get(
        "explanation",
        {}
    )

    simple_explanation = explanation.get(
        "simple",
        {}
    )

    technical_explanation = explanation.get(
        "technical",
        {}
    )

    st.divider()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🧠 AI Screening Result</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Predicted Severity",
            prediction["class_name"]
        )

    with col2:

        st.metric(
            "AI Confidence",
            f"{prediction['confidence'] * 100:.1f}%"
        )

    with col3:

        st.metric(
            "AI Reliability",
            final_trust["status"]
        )

    with col4:

        st.metric(
            "Reliability Score",
            f"{final_trust['percentage']:.1f}%"
        )


    # --------------------------------------------------------
    # PLAIN-LANGUAGE EXPLANATION (new)
    # --------------------------------------------------------

    if simple_explanation:

        st.markdown(
            '<div class="section-title">🗣️ What This Means (Plain Language)</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="explanation-box">{simple_explanation.get("summary", "")}</div>',
            unsafe_allow_html=True
        )

        reason_cols = st.columns(2)

        with reason_cols[0]:

            st.write(
                f"🔹 {simple_explanation.get('confidence_reason', '')}"
            )

            st.write(
                f"🔹 {simple_explanation.get('uncertainty_reason', '')}"
            )

        with reason_cols[1]:

            st.write(
                f"🔹 {simple_explanation.get('image_quality_reason', '')}"
            )

            st.write(
                f"🔹 {simple_explanation.get('trust_reason', '')}"
            )

        st.info(
            simple_explanation.get(
                "recommendation_explanation",
                ""
            )
        )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Class Probabilities"
    )

    probability_data = prediction[
        "all_probabilities"
    ]

    for class_name, probability in probability_data.items():

        st.write(
            f"**{class_name}** — {probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    st.markdown(
        "### 📷 Image Quality Assessment"
    )

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:

        st.metric(
            "Quality Status",
            image_quality["status"]
        )

        st.metric(
            "Overall Quality",
            f"{image_quality['percentage']:.1f}%"
        )

        st.write(
            f"Brightness: {image_quality['brightness']:.2f}"
        )

        st.write(
            f"Contrast: {image_quality['contrast']:.2f}"
        )

        st.write(
            f"Sharpness: {image_quality['sharpness']:.2f}"
        )

    with quality_col2:

        fig = create_quality_chart(
            image_quality
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    st.markdown(
        "### 🧮 Prediction Uncertainty"
    )

    u1, u2, u3 = st.columns(3)

    with u1:

        st.metric(
            "Confidence",
            f"{uncertainty['confidence'] * 100:.1f}%"
        )

    with u2:

        st.metric(
            "Prediction Margin",
            f"{uncertainty['margin'] * 100:.1f}%"
        )

    with u3:

        st.metric(
            "Normalized Entropy",
            f"{uncertainty['entropy']:.3f}"
        )


    # --------------------------------------------------------
    # EXPLAINABLE AI
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔥 Explainable AI — Grad-CAM</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The highlighted regions represent image areas "
        "that contributed strongly to the model's prediction."
    )

    heatmap = result.get(
        "gradcam_heatmap"
    )

    if heatmap is not None:

        overlay = create_gradcam_overlay(
            image,
            heatmap
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original Fundus Image",
                use_container_width=True
            )

        with col2:

            st.image(
                overlay,
                caption="Grad-CAM Explainability Heatmap",
                use_container_width=True
            )

    else:

        st.warning(
            "Grad-CAM could not be generated."
        )


    # --------------------------------------------------------
    # TECHNICAL EXPLANATION (new, for clinicians)
    # --------------------------------------------------------

    if technical_explanation:

        with st.expander(
            "🧑‍⚕️ Technical Decision Explanation (for clinical review)"
        ):

            tech_pred = technical_explanation.get(
                "prediction", {}
            )

            tech_uncertainty = technical_explanation.get(
                "uncertainty", {}
            )

            tech_quality = technical_explanation.get(
                "image_quality", {}
            )

            tech_model_trust = technical_explanation.get(
                "model_trust", {}
            )

            tech_final_trust = technical_explanation.get(
                "final_trust", {}
            )

            tech_referral = technical_explanation.get(
                "referral", {}
            )

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
                f"**Image Quality:** "
                f"{tech_quality.get('score_percentage', 0):.2f}% "
                f"({tech_quality.get('status', '')})"
            )

            st.write(
                f"**Model Trust:** "
                f"{tech_model_trust.get('score_percentage', 0):.2f}% "
                f"({tech_model_trust.get('status', '')})"
            )

            st.write(
                f"**Final Trust:** "
                f"{tech_final_trust.get('score_percentage', 0):.2f}% "
                f"({tech_final_trust.get('status', '')})"
            )

            st.write(
                f"**Referral:** {tech_referral.get('priority', '')} — "
                f"Urgency: {tech_referral.get('urgency_level', '')}"
            )

            st.write(
                f"**Decision Basis:** "
                f"{technical_explanation.get('decision_basis', '')}"
            )


    # --------------------------------------------------------
    # RISK & REFERRAL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🚨 Risk & Referral Prioritization</div>',
        unsafe_allow_html=True
    )

    st.warning(
        f"**Referral Priority:** {referral['priority']}"
    )

    r1, r2 = st.columns(2)

    with r1:

        st.write(
            f"**Urgency Level:** {referral['urgency_level']}"
        )

        st.write(
            f"**Reason:** {referral['reason']}"
        )

    with r2:

        st.write(
            f"**Recommended Action:** {referral['action']}"
        )


    # --------------------------------------------------------
    # AI RELIABILITY
    # --------------------------------------------------------

    st.markdown(
        "### 🛡️ AI Reliability Assessment"
    )

    st.write(
        f"Model Trust: **{model_trust['status']}**"
    )

    st.write(
        f"Model Trust Score: "
        f"**{model_trust['percentage']:.2f}%**"
    )

    st.write(
        f"Final Reliability: "
        f"**{final_trust['percentage']:.2f}%**"
    )

    st.info(
        final_trust["recommendation"]
    )


    # --------------------------------------------------------
    # SCREENING REPORT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📄 Screening Report</div>',
        unsafe_allow_html=True
    )

    report_col1, report_col2 = st.columns(2)

    with report_col1:

        st.write(
            f"**Patient:** {patient_name or 'Not provided'}"
        )

        st.write(
            f"**Patient ID:** {patient_id or 'Not provided'}"
        )

        st.write(
            f"**Predicted Severity:** {prediction['class_name']}"
        )

        st.write(
            f"**Confidence:** "
            f"{prediction['confidence'] * 100:.2f}%"
        )

        st.write(
            f"**Image Quality:** "
            f"{image_quality['status']}"
        )

    with report_col2:

        st.write(
            f"**Referral:** {referral['priority']}"
        )

        st.write(
            f"**Urgency:** {referral['urgency_level']}"
        )

        st.write(
            f"**AI Reliability:** {final_trust['status']}"
        )

        st.write(
            f"**Connectivity:** {connectivity}"
        )


    # --------------------------------------------------------
    # SAVE RECORD
    # --------------------------------------------------------

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

        "decision_basis": technical_explanation.get(
            "decision_basis",
            ""
        ),

        "connectivity": connectivity
    }

    if st.button(
        "💾 Save Screening Record"
    ):

        save_history(
            record
        )

        st.success(
            "Screening record saved locally."
        )


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if st.button(
        "📄 Generate PDF Report"
    ):

        pdf_path = generate_pdf(
            patient_name or "Unknown",
            patient_id or "N/A",
            age,
            sex,
            result,
            image_quality
        )

        with open(
            pdf_path,
            "rb"
        ) as f:

            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download Screening Report",
            data=pdf_bytes,
            file_name=os.path.basename(
                pdf_path
            ),
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================
# SCREENING HISTORY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📋 Local Screening History</div>',
    unsafe_allow_html=True
)

history = load_history()

if history:

    for item in reversed(history[-10:]):

        with st.expander(
            f"{item.get('patient_name', 'Unknown')} — "
            f"{item.get('prediction', 'N/A')}"
        ):

            st.write(
                f"**Date:** {item.get('timestamp', '')}"
            )

            st.write(
                f"**Patient ID:** {item.get('patient_id', 'N/A')}"
            )

            st.write(
                f"**Prediction:** {item.get('prediction', 'N/A')}"
            )

            st.write(
                f"**Confidence:** "
                f"{item.get('confidence', 0) * 100:.2f}%"
            )

            st.write(
                f"**Referral:** "
                f"{item.get('referral_priority', 'N/A')}"
            )

            if item.get("decision_basis"):

                st.write(
                    f"**Decision Basis:** "
                    f"{item.get('decision_basis', '')}"
                )

            st.write(
                f"**Connectivity:** "
                f"{item.get('connectivity', 'N/A')}"
            )

else:

    st.info(
        "No screening records saved yet."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ DRiShti-AI is an AI-assisted screening prototype. "
    "It does not replace ophthalmologist diagnosis, "
    "clinical judgment, or professional medical examination."
)


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