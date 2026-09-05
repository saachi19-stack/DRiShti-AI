import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ============================================================
# DRiShti-AI
# Explainable AI for Diabetic Retinopathy Screening
# ============================================================

st.set_page_config(
    page_title="DRiShti-AI | Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FOLDERS
# ============================================================
import os

st.write("CURRENT FOLDER:", os.getcwd())
st.write("REPORTS:", os.path.abspath("reports"))
st.write("IS REPORTS A FOLDER:", os.path.isdir("reports"))
st.write("IS REPORTS A FILE:", os.path.isfile("reports"))
os.makedirs("reports", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)

HISTORY_FILE = "screening_history.json"

# ============================================================
# SESSION STATE
# ============================================================

if "screenings" not in st.session_state:
    st.session_state.screenings = []

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = {}

# Load previous local history
if not st.session_state.screenings and os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r") as f:
            st.session_state.screenings = json.load(f)
    except:
        st.session_state.screenings = []


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #eaf2ff, #f8fbff);
    border: 1px solid #dbe7f5;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 18px;
    color: #536273;
}

.step-box {
    padding: 18px;
    border-radius: 14px;
    background: white;
    border: 1px solid #e2e8f0;
    text-align: center;
    min-height: 110px;
}

.metric-box {
    padding: 18px;
    border-radius: 14px;
    background: white;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
}

.result-box {
    padding: 25px;
    border-radius: 16px;
    background: white;
    border: 1px solid #dbe3ec;
}

.warning-box {
    padding: 15px;
    border-radius: 12px;
    background: #fff8e6;
    border: 1px solid #f0d48a;
}

.info-box {
    padding: 15px;
    border-radius: 12px;
    background: #eef6ff;
    border: 1px solid #cfe2ff;
}

.footer {
    text-align: center;
    padding: 25px;
    color: #718096;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCTIONS
# ============================================================

def assess_image_quality(image):
    """
    Basic retinal-image quality assessment.
    Uses blur, brightness and contrast.
    """

    img = np.array(image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Blur score
    if blur_score >= 150:
        blur_quality = 100
    elif blur_score >= 80:
        blur_quality = 75
    elif blur_score >= 40:
        blur_quality = 50
    else:
        blur_quality = 20

    # Brightness
    if 70 <= brightness <= 190:
        brightness_quality = 100
    elif 45 <= brightness <= 220:
        brightness_quality = 70
    else:
        brightness_quality = 35

    # Contrast
    if contrast >= 45:
        contrast_quality = 100
    elif contrast >= 25:
        contrast_quality = 70
    else:
        contrast_quality = 35

    quality_score = int(
        0.5 * blur_quality +
        0.25 * brightness_quality +
        0.25 * contrast_quality
    )

    if quality_score >= 70:
        status = "Good"
    else:
        status = "Needs Review"

    return {
        "score": quality_score,
        "status": status,
        "blur": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2)
    }


def demo_prediction():
    """
    TEMPORARY DEMO ONLY.

    This will be replaced with the actual trained ML model.
    """

    severity = "Moderate Diabetic Retinopathy"
    confidence = 91.0

    return severity, confidence


def generate_demo_heatmap(image):
    """
    TEMPORARY visualization.

    NOT actual Grad-CAM.
    Will be replaced after the trained model is connected.
    """

    img = np.array(image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    gray = cv2.resize(gray, (512, 512))

    # Demo visualization
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = cv2.resize(img, (512, 512))

    overlay = cv2.addWeighted(
        original.astype(np.uint8),
        0.55,
        heatmap.astype(np.uint8),
        0.45,
        0
    )

    return overlay


def get_risk_information(severity):

    severity_lower = severity.lower()

    if "no dr" in severity_lower:
        return (
            "LOW",
            "Routine",
            "No diabetic retinopathy detected by the AI screening model."
        )

    elif "mild" in severity_lower:
        return (
            "MODERATE",
            "Follow-up",
            "Follow-up ophthalmic evaluation is recommended."
        )

    elif "moderate" in severity_lower:
        return (
            "HIGH",
            "High Priority",
            "Ophthalmic evaluation should be prioritized."
        )

    elif "severe" in severity_lower:
        return (
            "VERY HIGH",
            "Urgent",
            "Urgent specialist evaluation is recommended."
        )

    elif "proliferative" in severity_lower:
        return (
            "VERY HIGH",
            "Urgent",
            "Urgent specialist evaluation is recommended."
        )

    return (
        "REVIEW",
        "Manual Review",
        "Professional review is recommended."
    )


def save_screening(record):

    st.session_state.screenings.append(record)

    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(
                st.session_state.screenings,
                f,
                indent=4
            )
    except Exception as e:
        st.warning(f"Could not save local history: {e}")


def create_pdf(record, heatmap_path=None):

    filename = (
        f"reports/"
        f"DRiShti_AI_{record['patient_id']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    c = canvas.Canvas(filename, pagesize=A4)

    width, height = A4

    y = height - 60

    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, y, "DRiShti-AI")

    y -= 25

    c.setFont("Helvetica", 11)
    c.drawString(
        50,
        y,
        "Explainable AI for Diabetic Retinopathy Screening"
    )

    y -= 45

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Screening Report")

    y -= 30

    c.setFont("Helvetica", 11)

    details = [
        f"Patient ID: {record['patient_id']}",
        f"Age: {record['age']}",
        f"Diabetes Duration: {record['diabetes_duration']} years",
        f"Date: {record['date']}",
        "",
        f"Image Quality: {record['quality_status']}",
        f"Quality Score: {record['quality_score']}/100",
        "",
        f"AI Screening Result: {record['severity']}",
        f"Model Confidence: {record['confidence']:.2f}%",
        f"Risk Level: {record['risk']}",
        f"Referral Priority: {record['referral']}",
    ]

    for line in details:

        if line == "":
            y -= 10
            continue

        c.drawString(55, y, line)
        y -= 20

    y -= 15

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Referral Guidance")

    y -= 20

    c.setFont("Helvetica", 10)

    guidance = record["guidance"]

    # Wrap guidance
    words = guidance.split()
    line = ""

    for word in words:

        if len(line + " " + word) > 80:
            c.drawString(55, y, line)
            y -= 15
            line = word
        else:
            line += " " + word

    if line:
        c.drawString(55, y, line)

    y -= 40

    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Important Disclaimer")

    y -= 18

    c.setFont("Helvetica", 9)

    disclaimer = (
        "This system is an AI-assisted screening prototype and "
        "does not replace examination or diagnosis by a qualified "
        "healthcare professional."
    )

    words = disclaimer.split()
    line = ""

    for word in words:

        if len(line + " " + word) > 90:
            c.drawString(55, y, line)
            y -= 13
            line = word
        else:
            line += " " + word

    if line:
        c.drawString(55, y, line)

    y -= 35

    c.drawString(
        50,
        y,
        "DRiShti-AI | SIH26038"
    )

    c.save()

    return filename


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 👁️ DRiShti-AI")

    st.caption(
        "Explainable AI for Diabetic Retinopathy Screening"
    )

    st.divider()

    st.markdown("### Connectivity")

    connectivity = st.toggle(
        "Limited Connectivity Mode",
        value=False
    )

    if connectivity:
        st.warning(
            "Offline-first mode active. "
            "Screening records will be stored locally."
        )
    else:
        st.success("Online mode")

    st.divider()

    st.markdown("### Screening Workflow")

    st.markdown("""
    **1.** Retinal Image  
    ↓  
    **2.** Image Quality Check  
    ↓  
    **3.** DR Severity Screening  
    ↓  
    **4.** Explainable AI  
    ↓  
    **5.** Risk & Referral
    """)

    st.divider()

    st.markdown("### About")

    st.info(
        "DRiShti-AI is designed as an AI-assisted "
        "screening support tool for diabetic retinopathy."
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
👁️ DRiShti-AI
</div>

<div class="hero-subtitle">
Explainable AI for Diabetic Retinopathy Screening in Rural India
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

total = len(st.session_state.screenings)

high_priority = sum(
    1 for x in st.session_state.screenings
    if x.get("risk") in ["HIGH", "VERY HIGH"]
)

urgent = sum(
    1 for x in st.session_state.screenings
    if x.get("referral") == "Urgent"
)

pending = (
    total if connectivity else 0
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-number">{total}</div>
        <div class="metric-label">Total Screened</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-number">{high_priority}</div>
        <div class="metric-label">High Priority</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-number">{urgent}</div>
        <div class="metric-label">Urgent Referrals</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-box">
        <div class="metric-number">{pending}</div>
        <div class="metric-label">Local Records</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# WORKFLOW
# ============================================================

st.markdown("### 🔄 Screening Workflow")

steps = [
    ("📷", "Capture", "Upload retinal image"),
    ("🔍", "Quality", "Check image quality"),
    ("🧠", "AI Screening", "Predict DR severity"),
    ("💡", "Explain", "Visualize AI reasoning"),
    ("🏥", "Refer", "Prioritize follow-up")
]

cols = st.columns(5)

for col, step in zip(cols, steps):

    with col:

        st.markdown(
            f"""
            <div class="step-box">
            <div style="font-size:30px">{step[0]}</div>
            <b>{step[1]}</b>
            <br>
            <small>{step[2]}</small>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# PATIENT DETAILS
# ============================================================

st.markdown("### 👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    patient_id = st.text_input(
        "Patient ID",
        placeholder="Example: DR-001"
    )

with col2:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

with col3:

    diabetes_duration = st.number_input(
        "Diabetes Duration (years)",
        min_value=0,
        max_value=80,
        value=5
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("### 📷 Retinal Image")

uploaded_file = st.file_uploader(
    "Upload a fundus / retinal image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Retinal Image",
            use_container_width=True
        )

    with col2:

        st.markdown("#### 🔍 Image Quality Assessment")

        quality = assess_image_quality(image)

        q1, q2, q3 = st.columns(3)

        with q1:
            st.metric(
                "Quality Score",
                f"{quality['score']}/100"
            )

        with q2:
            st.metric(
                "Brightness",
                quality["brightness"]
            )

        with q3:
            st.metric(
                "Contrast",
                quality["contrast"]
            )

        if quality["status"] == "Good":

            st.success(
                "Image quality is suitable for screening."
            )

        else:

            st.warning(
                "Image may need to be retaken or reviewed."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    analyze = st.button(
        "🧠 Analyze Retinal Image",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze:

        if not patient_id.strip():

            st.error("Please enter a Patient ID.")

        else:

            with st.spinner("Analyzing retinal image..."):

                # TEMPORARY DEMO MODEL
                severity, confidence = demo_prediction()

                risk, referral, guidance = get_risk_information(
                    severity
                )

                heatmap = generate_demo_heatmap(image)

                record = {
                    "patient_id": patient_id,
                    "age": int(age),
                    "diabetes_duration": int(diabetes_duration),
                    "date": datetime.now().strftime(
                        "%d-%m-%Y %H:%M"
                    ),
                    "severity": severity,
                    "confidence": confidence,
                    "risk": risk,
                    "referral": referral,
                    "guidance": guidance,
                    "quality_score": quality["score"],
                    "quality_status": quality["status"]
                }

                st.session_state.analysis_data = {
                    "record": record,
                    "heatmap": heatmap
                }

                st.session_state.analysis_done = True

                save_screening(record)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_done:

    data = st.session_state.analysis_data
    record = data["record"]
    heatmap = data["heatmap"]

    st.divider()

    st.markdown("## 🧠 AI Screening Result")

    st.warning(
        "DEMO MODE — The current prediction is a placeholder. "
        "The trained DR model will replace this result."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="result-box">
            <h4>Detected Severity</h4>
            <h2>{record['severity']}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="result-box">
            <h4>Model Confidence</h4>
            <h2>{record['confidence']:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="result-box">
            <h4>Risk Level</h4>
            <h2>{record['risk']}</h2>
            <b>{record['referral']}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.markdown("### 💡 Explainable AI")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            heatmap,
            caption="Demo visualization — will be replaced by actual Grad-CAM",
            use_container_width=True
        )

    with col2:

        st.markdown("""
        #### Why Explainability?

        DRiShti-AI is designed to provide a visual explanation
        alongside its screening prediction.

        **Actual implementation:**

        `Retinal Image`
        ↓

        `AI Model`
        ↓

        `Prediction`
        +
        `Grad-CAM`
        ↓

        `Highlighted Retinal Regions`

        This helps healthcare workers understand which
        retinal regions contributed to the model's prediction.
        """)

        st.info(
            "The current heatmap is a prototype visualization. "
            "Actual Grad-CAM will be connected with the trained "
            "model."
        )


    # ========================================================
    # RISK / REFERRAL
    # ========================================================

    st.markdown("### 🏥 Risk & Referral Prioritization")

    if record["referral"] == "Urgent":

        st.error(
            f"🚨 {record['referral']}: {record['guidance']}"
        )

    elif record["referral"] == "High Priority":

        st.warning(
            f"⚠️ {record['referral']}: {record['guidance']}"
        )

    else:

        st.info(
            f"ℹ️ {record['referral']}: {record['guidance']}"
        )


    # ========================================================
    # REPORT
    # ========================================================

    st.markdown("### 📄 Screening Report")

    report_df = pd.DataFrame({
        "Parameter": [
            "Patient ID",
            "Age",
            "Diabetes Duration",
            "Image Quality",
            "Quality Score",
            "AI Screening Result",
            "Confidence",
            "Risk Level",
            "Referral Priority"
        ],
        "Result": [
            record["patient_id"],
            record["age"],
            f"{record['diabetes_duration']} years",
            record["quality_status"],
            f"{record['quality_score']}/100",
            record["severity"],
            f"{record['confidence']:.1f}%",
            record["risk"],
            record["referral"]
        ]
    })

    st.table(report_df)

    pdf_file = create_pdf(record)

    with open(pdf_file, "rb") as f:

        st.download_button(
            "⬇️ Download Screening Report",
            data=f,
            file_name=os.path.basename(pdf_file),
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================
# SCREENING HISTORY
# ============================================================

st.divider()

st.markdown("## 📋 Screening History")

if st.session_state.screenings:

    history_df = pd.DataFrame(
        st.session_state.screenings
    )

    display_columns = [
        "patient_id",
        "age",
        "severity",
        "risk",
        "referral",
        "quality_status",
        "date"
    ]

    available_columns = [
        c for c in display_columns
        if c in history_df.columns
    ]

    st.dataframe(
        history_df[available_columns],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No screening records yet. "
        "Complete a screening to see it here."
    )


# ============================================================
# RURAL / OFFLINE FEATURE
# ============================================================

st.divider()

st.markdown("## 🌐 Rural & Low-Connectivity Support")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    ### 📡 Limited Connectivity

    DRiShti-AI is designed around a low-connectivity workflow:

    - Screening can continue during limited connectivity.
    - Records can be stored locally.
    - Results can be reviewed later.
    - Data can be synchronized when connectivity returns.
    """)

with col2:

    if connectivity:

        st.warning(
            "🔴 LIMITED CONNECTIVITY — "
            "Local storage mode active."
        )

        if st.button("🔄 Simulate Sync"):

            st.success(
                "Local screening records synchronized successfully."
            )

    else:

        st.success(
            "🟢 CONNECTED — System ready for synchronization."
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.markdown("""
<div class="warning-box">

<b>⚠️ Medical Disclaimer</b>

<br><br>

DRiShti-AI is an AI-assisted screening prototype intended
to support diabetic retinopathy screening workflows.

It does <b>not</b> replace professional medical examination,
clinical judgment, or diagnosis by a qualified healthcare
professional.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<b>DRiShti-AI</b> |
Explainable AI for Diabetic Retinopathy Screening |
SIH26038

<br><br>

Built for Smart India Hackathon 2026

</div>
""", unsafe_allow_html=True)