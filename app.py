# app.py — Banana Ripeness Classifier
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="BananaIQ — Ripeness Classifier",
    page_icon="🔬",
    layout="centered"
)

# -----------------------------
# Custom CSS — luxury dark editorial aesthetic
# -----------------------------
# Non-blocking font preload — injected before CSS so browser fetches fonts in parallel
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap" media="print" onload="this.media='all'">
<noscript><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap"></noscript>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0c0f14;
    color: #e8e4dc;
}

/* Full-page gradient background */
.stApp {
    background: linear-gradient(160deg, #0c0f14 0%, #141a26 45%, #1a2235 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 720px;
    padding: 3rem 2rem 4rem;
}

/* ── Custom header ── */
.hero-label {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #8a9ab5;
    margin-bottom: 0.6rem;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 400;
    line-height: 1.12;
    color: #f0ece3;
    margin: 0 0 0.5rem;
    letter-spacing: -0.01em;
}

.hero-title em {
    font-style: italic;
    color: #c8b97a;
}

.hero-sub {
    font-size: 0.92rem;
    color: #b0bdd0;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.01em;
}

/* ── Divider ── */
.thin-rule {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 2rem 0;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 2px;
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(200, 185, 122, 0.4);
}
[data-testid="stFileUploader"] label {
    color: #8a9ab5 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stFileDropzone"] {
    background: transparent !important;
}

/* ── Uploaded image ── */
[data-testid="stImage"] img {
    border-radius: 2px;
    border: 1px solid rgba(255,255,255,0.07);
}

/* ── Result card ── */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 2px;
    padding: 2rem 2.2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #c8b97a, #8a7a4a);
}

.result-status {
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
    font-size: 0.68rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #c0cfe0;
    margin-bottom: 0.4rem;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    color: #f0ece3;
    margin: 0 0 0.25rem;
    line-height: 1.1;
}
.result-confidence {
    font-size: 0.8rem;
    color: #c8b97a;
    letter-spacing: 0.1em;
    font-weight: 500;
    text-transform: uppercase;
}
.result-advice {
    margin-top: 1.2rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.1);
    font-size: 0.92rem;
    color: #b0bdd0;
    font-weight: 300;
    line-height: 1.65;
}

/* ── Warning card ── */
.warn-card {
    background: rgba(200, 130, 60, 0.07);
    border: 1px solid rgba(200, 130, 60, 0.25);
    border-radius: 2px;
    padding: 1rem 1.4rem;
    font-size: 0.82rem;
    color: #c8955a;
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
}

/* ── Confidence bars ── */
.conf-section-title {
    font-size: 0.68rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #c0cfe0;
    margin-bottom: 1.1rem;
    font-weight: 500;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}
.conf-name {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #b0bdd0;
    width: 78px;
    flex-shrink: 0;
    font-weight: 500;
}
.conf-track {
    flex: 1;
    height: 2px;
    background: rgba(255,255,255,0.12);
    border-radius: 1px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #c8b97a, #e8d89a);
    border-radius: 1px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.conf-fill-dim {
    background: rgba(255,255,255,0.25);
}
.conf-pct {
    font-size: 0.75rem;
    color: #b0bdd0;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
    font-weight: 400;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #8a9ab5 !important;
    font-size: 0.8rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CLASS_NAMES — alphabetical, matches Keras indices
# 0=overripe, 1=ripe, 2=rotten, 3=unripe
# -----------------------------
MODEL_PATH = "model/banana_cnn_model.h5"
IMG_SIZE = 128
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]

CLASS_ADVICE = {
    "unripe":   "Needs more time. Leave at room temperature for 2–4 days.",
    "ripe":     "Perfect. This banana is ready to eat.",
    "overripe": "Best used for smoothies, banana bread, or baking.",
    "rotten":   "This banana has gone bad. Do not eat it.",
}

DISPLAY_ORDER = ["unripe", "ripe", "overripe", "rotten"]
CONFIDENCE_THRESHOLD = 0.60

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_banana_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at `{MODEL_PATH}`. Please place it there and restart.")
        st.stop()
    return load_model(MODEL_PATH)

model = load_banana_model()

assert model.output_shape[-1] == len(CLASS_NAMES), (
    f"Model has {model.output_shape[-1]} outputs but CLASS_NAMES has {len(CLASS_NAMES)} entries."
)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<h1 class="hero-title">BananaIQ -<br><em>A Robust</em> Banana Ripeness Classifier</h1>
<p class="hero-sub">Upload a photograph to assess the ripeness stage of your banana.</p>
<hr class="thin-rule">
""", unsafe_allow_html=True)

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, use_container_width=True)

    # Preprocess
    img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analysing..."):
        predictions = model.predict(img_array, verbose=0)[0]

    class_index = int(np.argmax(predictions))
    class_label = CLASS_NAMES[class_index]
    confidence  = float(predictions[class_index])

    st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)

    # Low confidence warning
    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown(f"""
        <div class="warn-card">
            ⚠ &nbsp;Low confidence &nbsp;({confidence*100:.1f}%) — try a clearer, well-lit photograph.
        </div>
        """, unsafe_allow_html=True)

    # Result card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-status">Classification result</div>
        <div class="result-label">{class_label.capitalize()}</div>
        <div class="result-confidence">{confidence*100:.1f}% confidence</div>
        <div class="result-advice">{CLASS_ADVICE[class_label]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence breakdown
    st.markdown('<p class="conf-section-title">Confidence breakdown</p>', unsafe_allow_html=True)

    conf_rows_html = ""
    for name in DISPLAY_ORDER:
        idx  = CLASS_NAMES.index(name)
        pct  = float(predictions[idx]) * 100
        is_top = (name == class_label)
        fill_class = "conf-fill" if is_top else "conf-fill conf-fill-dim"
        conf_rows_html += f"""
        <div class="conf-row">
            <span class="conf-name">{name.capitalize()}</span>
            <div class="conf-track">
                <div class="{fill_class}" style="width:{pct:.1f}%"></div>
            </div>
            <span class="conf-pct">{pct:.0f}%</span>
        </div>
        """
    st.markdown(conf_rows_html, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="
        border: 1px dashed rgba(255,255,255,0.35);
        border-radius: 2px;
        padding: 3.5rem 2rem;
        text-align: center;
        color: #8a9ab5;
        font-size: 0.8rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-weight: 400;
        margin-top: 0.5rem;
    ">
        No image selected
    </div>
    """, unsafe_allow_html=True)