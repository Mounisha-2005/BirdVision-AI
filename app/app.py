import os
import json
import textwrap
from html import escape

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# app.py is inside:
# C:\Users\Dell\Desktop\Bird_Classification_CNN\app
#
# Therefore ".." points to:
# C:\Users\Dell\Desktop\Bird_Classification_CNN

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_bird_cnn.keras"
)

CENTROIDS_PATH = os.path.join(
    BASE_DIR, "models", "class_centroids.npy"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "models", "class_names.json"
)

OPEN_SET_PATH = os.path.join(
    BASE_DIR, "models", "optimized_open_set.json"
)


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BirdVision AI",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 3. HTML RENDERING HELPER
# ============================================================
#
# IMPORTANT:
# This fixes the problem where Streamlit displays:
#
# <div class="hero">
# ...
# </div>
#
# as literal text.
#
# Python indentation is removed using textwrap.dedent()
# before the HTML is passed to Streamlit.
# ============================================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


# ============================================================
# 4. CUSTOM KERAS FUNCTION
# ============================================================

def clip_to_unit_range(x):
    return tf.clip_by_value(x, 0.0, 1.0)


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "clip_to_unit_range": clip_to_unit_range
            },
            safe_mode=False,
            compile=False
        )

    except TypeError:
        # Compatibility fallback for older TensorFlow/Keras
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "clip_to_unit_range": clip_to_unit_range
            },
            compile=False
        )

    return model


# ============================================================
# 6. LOAD SUPPORTING DATA
# ============================================================

@st.cache_data
def load_supporting_data():

    if not os.path.exists(CENTROIDS_PATH):
        raise FileNotFoundError(
            f"Centroid file not found:\n{CENTROIDS_PATH}"
        )

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    if not os.path.exists(OPEN_SET_PATH):
        raise FileNotFoundError(
            f"Open-set configuration not found:\n{OPEN_SET_PATH}"
        )

    centroids = np.load(CENTROIDS_PATH)

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        class_names = json.load(file)

    with open(
        OPEN_SET_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        open_set_config = json.load(file)

    # --------------------------------------------------------
    # Handle class_names saved either as a list or dictionary
    # --------------------------------------------------------

    if isinstance(class_names, dict):

        try:
            class_names = [
                class_names[str(i)]
                for i in range(len(class_names))
            ]

        except Exception:
            class_names = list(class_names.values())

    class_names = [
        str(name)
        for name in class_names
    ]

    # --------------------------------------------------------
    # Get best threshold rule
    # --------------------------------------------------------

    best_rule = open_set_config.get(
        "best_rule",
        {}
    )

    def get_threshold(
        keys,
        default_value
    ):

        for key in keys:

            if key in best_rule:

                try:
                    return float(best_rule[key])
                except Exception:
                    pass

        return float(default_value)

    confidence_threshold = get_threshold(
        [
            "confidence_threshold",
            "confidence",
            "min_confidence"
        ],
        0.50
    )

    similarity_threshold = get_threshold(
        [
            "similarity_threshold",
            "similarity",
            "min_similarity"
        ],
        0.50
    )

    margin_threshold = get_threshold(
        [
            "margin_threshold",
            "margin",
            "min_margin"
        ],
        0.10
    )

    return (
        centroids,
        class_names,
        open_set_config,
        confidence_threshold,
        similarity_threshold,
        margin_threshold
    )


# ============================================================
# 7. FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def build_feature_extractor(_model):

    try:

        dense_layer = _model.get_layer("dense")

        feature_extractor = tf.keras.Model(
            inputs=_model.inputs,
            outputs=dense_layer.output
        )

        return feature_extractor

    except Exception as error:

        raise RuntimeError(
            "Could not create the feature extractor from "
            "the 'dense' layer.\n\n"
            f"Original error: {error}"
        )


# ============================================================
# 8. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # The trained model expects raw pixel values
    # in the 0-255 range.
    #
    # Therefore we DO NOT divide by 255 here.

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# 9. COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vector,
    matrix
):

    vector = np.asarray(
        vector,
        dtype=np.float32
    ).reshape(-1)

    matrix = np.asarray(
        matrix,
        dtype=np.float32
    )

    vector_norm = np.linalg.norm(vector)

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1
    )

    denominator = (
        vector_norm *
        matrix_norms
    )

    denominator = np.where(
        denominator == 0,
        1e-8,
        denominator
    )

    similarities = (
        matrix @ vector
    ) / denominator

    return similarities


# ============================================================
# 10. BIRD PREDICTION
# ============================================================

def predict_bird(
    image,
    model,
    centroids,
    class_names,
    feature_extractor,
    confidence_threshold,
    similarity_threshold,
    margin_threshold
):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_array = preprocess_image(image)

    # --------------------------------------------------------
    # CNN prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Sort predictions
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    top1_index = int(
        sorted_indices[0]
    )

    top2_index = int(
        sorted_indices[1]
    )

    confidence = float(
        probabilities[top1_index]
    )

    second_confidence = float(
        probabilities[top2_index]
    )

    margin = (
        confidence -
        second_confidence
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if top1_index >= len(class_names):

        raise ValueError(
            "Number of CNN output classes does not match "
            "class_names.json."
        )

    # --------------------------------------------------------
    # CNN predicted class
    # --------------------------------------------------------

    predicted_name = class_names[
        top1_index
    ]

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    features = feature_extractor.predict(
        image_array,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Centroid similarity
    # --------------------------------------------------------

    if centroids.ndim != 2:

        raise ValueError(
            "class_centroids.npy must contain a 2D array."
        )

    if centroids.shape[0] != len(class_names):

        raise ValueError(
            "Number of centroids does not match "
            "number of classes."
        )

    similarities = cosine_similarity(
        features,
        centroids
    )

    nearest_centroid_index = int(
        np.argmax(similarities)
    )

    nearest_similarity = float(
        similarities[
            nearest_centroid_index
        ]
    )

    centroid_name = class_names[
        nearest_centroid_index
    ]

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    agreement = (
        top1_index ==
        nearest_centroid_index
    )

    # --------------------------------------------------------
    # Open-set decision
    # --------------------------------------------------------

    confidence_pass = (
        confidence >=
        confidence_threshold
    )

    similarity_pass = (
        nearest_similarity >=
        similarity_threshold
    )

    margin_pass = (
        margin >=
        margin_threshold
    )

    accepted = (
        confidence_pass
        and similarity_pass
        and margin_pass
        and agreement
    )

    # --------------------------------------------------------
    # Top 3
    # --------------------------------------------------------

    top_predictions = []

    for rank, index in enumerate(
        sorted_indices[:3],
        start=1
    ):

        index = int(index)

        top_predictions.append(
            {
                "rank": rank,
                "class_name": class_names[index],
                "confidence": float(
                    probabilities[index]
                )
            }
        )

    return {

        "accepted": accepted,

        "predicted_name": predicted_name,

        "centroid_name": centroid_name,

        "confidence": confidence,

        "second_confidence":
            second_confidence,

        "margin": margin,

        "similarity":
            nearest_similarity,

        "agreement": agreement,

        "confidence_pass":
            confidence_pass,

        "similarity_pass":
            similarity_pass,

        "margin_pass":
            margin_pass,

        "top_predictions":
            top_predictions,

        "confidence_threshold":
            confidence_threshold,

        "similarity_threshold":
            similarity_threshold,

        "margin_threshold":
            margin_threshold
    }


# ============================================================
# 11. LOAD EVERYTHING
# ============================================================

try:

    model = load_model()

    (
        centroids,
        class_names,
        open_set_config,
        CONFIDENCE_THRESHOLD,
        SIMILARITY_THRESHOLD,
        MARGIN_THRESHOLD
    ) = load_supporting_data()

    feature_extractor = build_feature_extractor(
        model
    )

except Exception as error:

    st.error(
        "BirdVision could not load the model."
    )

    st.code(
        str(error)
    )

    st.info(
        "Check that the required files exist inside "
        "the models folder."
    )

    st.stop()


# ============================================================
# 12. GLOBAL CSS
# ============================================================

render_html("""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap'
);

:root {
    --bg: #07090d;
    --panel: #0d1118;
    --panel2: #111722;
    --border: #222b38;
    --text: #edf2f7;
    --muted: #8994a3;
    --green: #62e6a4;
    --green2: #2ebd7d;
    --red: #ff6b7a;
    --yellow: #f4d35e;
    --blue: #72b7ff;
    --cyan: #5ee7f2;
}

html {
    scroll-behavior: smooth;
}

body {
    background: var(--bg);
}

.stApp {
    background:
        radial-gradient(
            circle at 20% 0%,
            rgba(98, 230, 164, 0.08),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 10%,
            rgba(114, 183, 255, 0.06),
            transparent 25%
        ),
        var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: rgba(7, 9, 13, 0.90);
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
}

a {
    text-decoration: none !important;
}

.top-nav {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 20px;
    margin-bottom: 45px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: rgba(9, 12, 17, 0.92);
    backdrop-filter: blur(18px);
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 800;
    letter-spacing: -0.4px;
}

.brand-icon {
    width: 35px;
    height: 35px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: rgba(98, 230, 164, 0.12);
    border: 1px solid rgba(98, 230, 164, 0.25);
}

.nav-links {
    display: flex;
    gap: 24px;
}

.nav-links a {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    transition: 0.2s;
}

.nav-links a:hover {
    color: var(--green);
}

.eyebrow {
    color: var(--green);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

.hero {
    padding: 45px 0 60px;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1.5fr 0.8fr;
    gap: 35px;
    align-items: center;
}

.hero h1 {
    font-size: clamp(42px, 7vw, 82px);
    line-height: 0.95;
    letter-spacing: -5px;
    margin: 0 0 25px;
    font-weight: 800;
}

.hero h1 span {
    color: var(--green);
}

.hero-description {
    max-width: 680px;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.8;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 25px;
    padding: 9px 13px;
    border: 1px solid rgba(98, 230, 164, 0.25);
    border-radius: 100px;
    color: var(--green);
    background: rgba(98, 230, 164, 0.06);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
}

.hero-card {
    padding: 28px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background:
        linear-gradient(
            145deg,
            rgba(98, 230, 164, 0.06),
            rgba(114, 183, 255, 0.02)
        ),
        var(--panel);
}

.hero-card-title {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    margin-bottom: 20px;
}

.signal {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
}

.signal:last-child {
    border-bottom: none;
}

.signal-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 12px rgba(98, 230, 164, 0.6);
}

.signal-label {
    flex: 1;
    color: var(--muted);
    font-size: 13px;
}

.signal-value {
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
}

.section {
    margin-top: 80px;
    scroll-margin-top: 100px;
}

.section-number {
    color: var(--green);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
}

.section-title {
    font-size: 34px;
    letter-spacing: -1.5px;
    margin: 0 0 12px;
}

.section-description {
    color: var(--muted);
    line-height: 1.7;
    margin-bottom: 30px;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}

.card {
    padding: 25px;
    min-height: 160px;
    border: 1px solid var(--border);
    border-radius: 15px;
    background: var(--panel);
}

.card-number {
    color: var(--green);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    margin-bottom: 20px;
}

.card h3 {
    margin: 0 0 10px;
    font-size: 18px;
}

.card p {
    color: var(--muted);
    line-height: 1.6;
    font-size: 13px;
    margin: 0;
}

.pipeline {
    display: grid;
    grid-template-columns:
        repeat(5, 1fr);
    gap: 10px;
}

.pipeline-step {
    position: relative;
    padding: 20px;
    min-height: 125px;
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 13px;
}

.pipeline-step:not(:last-child)::after {
    content: '→';
    position: absolute;
    right: -14px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--green);
    z-index: 2;
    font-size: 18px;
}

.pipeline-index {
    color: var(--green);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    margin-bottom: 15px;
}

.pipeline-step strong {
    font-size: 14px;
}

.pipeline-step p {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.5;
    margin-top: 8px;
}

.stat-grid {
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 12px;
}

.stat-card {
    padding: 25px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--panel);
}

.stat-label {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stat-value {
    margin-top: 10px;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
}

.stat-value.green {
    color: var(--green);
}

.stat-value.blue {
    color: var(--blue);
}

.stat-value.yellow {
    color: var(--yellow);
}

.stat-value.cyan {
    color: var(--cyan);
}

.inference-box {
    padding: 28px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: var(--panel);
}

.upload-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 5px;
}

.upload-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 20px;
}

.result-success {
    padding: 30px;
    margin-top: 25px;
    border: 1px solid rgba(98, 230, 164, 0.30);
    border-radius: 16px;
    background: rgba(98, 230, 164, 0.06);
}

.result-unknown {
    padding: 30px;
    margin-top: 25px;
    border: 1px solid rgba(255, 107, 122, 0.35);
    border-radius: 16px;
    background: rgba(255, 107, 122, 0.06);
}

.result-status {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.result-name {
    font-size: 35px;
    line-height: 1.1;
    margin: 10px 0 25px;
    font-weight: 800;
}

.result-success .result-name {
    color: var(--green);
}

.result-unknown .result-name {
    color: var(--red);
}

.measure-grid {
    display: grid;
    grid-template-columns:
        repeat(3, 1fr);
    gap: 12px;
}

.measure {
    padding: 17px;
    border: 1px solid var(--border);
    border-radius: 11px;
    background: rgba(0,0,0,0.15);
}

.measure-label {
    color: var(--muted);
    font-size: 11px;
}

.measure-value {
    margin-top: 7px;
    font-family: 'DM Mono', monospace;
    font-size: 19px;
    font-weight: 500;
}

.check-grid {
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 10px;
    margin-top: 18px;
}

.check {
    padding: 15px;
    border: 1px solid var(--border);
    border-radius: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
}

.check.pass {
    color: var(--green);
    border-color: rgba(98, 230, 164, 0.25);
}

.check.fail {
    color: var(--red);
    border-color: rgba(255, 107, 122, 0.25);
}

.class-grid {
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 8px;
}

.class-card {
    padding: 13px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--panel);
    color: var(--muted);
    font-size: 11px;
    transition: 0.2s;
}

.class-card:hover {
    border-color: rgba(98, 230, 164, 0.35);
    color: var(--green);
}

.class-number {
    color: var(--green);
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    margin-bottom: 6px;
}

.note {
    padding: 20px;
    margin-top: 20px;
    border-left: 2px solid var(--green);
    background: rgba(98, 230, 164, 0.04);
    color: var(--muted);
    line-height: 1.7;
    font-size: 13px;
}

.footer {
    margin-top: 100px;
    padding-top: 30px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
}

div[data-testid="stFileUploader"] {
    border: 1px dashed #303b4a;
    border-radius: 13px;
    padding: 8px;
    background: rgba(255,255,255,0.015);
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(98, 230, 164, 0.5);
}

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 10px;
    border: 1px solid rgba(98, 230, 164, 0.35);
    background: rgba(98, 230, 164, 0.10);
    color: var(--green);
    font-weight: 700;
}

.stButton > button:hover {
    border-color: var(--green);
    background: rgba(98, 230, 164, 0.17);
    color: white;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
}

@media (max-width: 900px) {

    .hero-grid {
        grid-template-columns: 1fr;
    }

    .card-grid {
        grid-template-columns: 1fr;
    }

    .pipeline {
        grid-template-columns: 1fr;
    }

    .pipeline-step:not(:last-child)::after {
        content: '↓';
        right: 50%;
        top: auto;
        bottom: -19px;
        transform: translateX(50%);
    }

    .stat-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .class-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .nav-links {
        display: none;
    }
}

</style>
""")


# ============================================================
# 13. TOP NAVIGATION
# ============================================================

render_html("""
<nav class="top-nav">

    <div class="brand">
        <div class="brand-icon">🐦</div>
        <div>BirdVision AI</div>
    </div>

    <div class="nav-links">
        <a href="#overview">OVERVIEW</a>
        <a href="#pipeline">PIPELINE</a>
        <a href="#model">MODEL</a>
        <a href="#inference">CLASSIFY</a>
        <a href="#classes">CLASSES</a>
    </div>

</nav>
""")


# ============================================================
# 14. HERO SECTION
# ============================================================

render_html(f"""
<section class="hero">

    <div class="hero-grid">

        <div>

            <div class="eyebrow">
                COMPUTER VISION · CNN · OPEN SET
            </div>

            <h1>
                Identify<br>
                <span>the bird.</span>
            </h1>

            <div class="hero-description">
                A deep-learning based bird classification
                system that combines CNN predictions with
                feature-space similarity to distinguish
                recognized species from unknown inputs.
            </div>

            <div class="hero-badge">
                ● MODEL ONLINE · {len(class_names)} SPECIES
            </div>

        </div>

        <div class="hero-card">

            <div class="hero-card-title">
                SYSTEM STATUS
            </div>

            <div class="signal">
                <div class="signal-dot"></div>
                <div class="signal-label">CNN classifier</div>
                <div class="signal-value">READY</div>
            </div>

            <div class="signal">
                <div class="signal-dot"></div>
                <div class="signal-label">Feature centroids</div>
                <div class="signal-value">READY</div>
            </div>

            <div class="signal">
                <div class="signal-dot"></div>
                <div class="signal-label">Open-set detector</div>
                <div class="signal-value">ACTIVE</div>
            </div>

            <div class="signal">
                <div class="signal-dot"></div>
                <div class="signal-label">Inference size</div>
                <div class="signal-value">224 × 224</div>
            </div>

        </div>

    </div>

</section>
""")


# ============================================================
# 15. OVERVIEW
# ============================================================

render_html("""
<section class="section" id="overview">

    <div class="section-number">01 · OVERVIEW</div>

    <h2 class="section-title">
        From image to species
    </h2>

    <p class="section-description">
        The application accepts a bird image, processes it
        through the trained convolutional neural network,
        extracts deep features and validates the prediction
        using centroid similarity and decision thresholds.
    </p>

    <div class="card-grid">

        <div class="card">
            <div class="card-number">01</div>
            <h3>Image Input</h3>
            <p>
                Upload a JPG, JPEG, PNG or WEBP bird image.
                The image is converted to RGB and resized
                to the model input size.
            </p>
        </div>

        <div class="card">
            <div class="card-number">02</div>
            <h3>CNN Prediction</h3>
            <p>
                The trained CNN generates class probabilities
                for all supported bird species.
            </p>
        </div>

        <div class="card">
            <div class="card-number">03</div>
            <h3>Open-set Validation</h3>
            <p>
                Confidence, prediction margin, feature
                similarity and CNN-centroid agreement are
                jointly evaluated.
            </p>
        </div>

    </div>

</section>
""")


# ============================================================
# 16. PIPELINE
# ============================================================

render_html("""
<section class="section" id="pipeline">

    <div class="section-number">02 · PIPELINE</div>

    <h2 class="section-title">
        Inference architecture
    </h2>

    <p class="section-description">
        The complete prediction flow used by BirdVision AI.
    </p>

    <div class="pipeline">

        <div class="pipeline-step">
            <div class="pipeline-index">STEP 01</div>
            <strong>Upload</strong>
            <p>Bird image enters the application.</p>
        </div>

        <div class="pipeline-step">
            <div class="pipeline-index">STEP 02</div>
            <strong>Preprocess</strong>
            <p>RGB conversion and 224×224 resizing.</p>
        </div>

        <div class="pipeline-step">
            <div class="pipeline-index">STEP 03</div>
            <strong>CNN</strong>
            <p>Model produces species probabilities.</p>
        </div>

        <div class="pipeline-step">
            <div class="pipeline-index">STEP 04</div>
            <strong>Feature Space</strong>
            <p>Deep representation is compared with centroids.</p>
        </div>

        <div class="pipeline-step">
            <div class="pipeline-index">STEP 05</div>
            <strong>Decision</strong>
            <p>Known or unknown classification.</p>
        </div>

    </div>

</section>
""")


# ============================================================
# 17. MODEL INFORMATION
# ============================================================

render_html("""
<section class="section" id="model">

    <div class="section-number">03 · MODEL</div>

    <h2 class="section-title">
        Model intelligence
    </h2>

    <p class="section-description">
        Core configuration of the trained bird classification
        system currently loaded by the application.
    </p>

</section>
""")

# Dynamic statistics

render_html(f"""
<div class="stat-grid">

    <div class="stat-card">
        <div class="stat-label">
            Classes
        </div>
        <div class="stat-value green">
            {len(class_names)}
        </div>
    </div>

    <div class="stat-card">
        <div class="stat-label">
            Input
        </div>
        <div class="stat-value blue">
            224²
        </div>
    </div>

    <div class="stat-card">
        <div class="stat-label">
            CNN
        </div>
        <div class="stat-value yellow">
            ACTIVE
        </div>
    </div>

    <div class="stat-card">
        <div class="stat-label">
            Open Set
        </div>
        <div class="stat-value cyan">
            ON
        </div>
    </div>

</div>
""")


# ============================================================
# 18. THRESHOLD INFORMATION
# ============================================================

render_html(f"""
<div class="note">

    <strong>Decision thresholds</strong><br><br>

    Confidence ≥
    {CONFIDENCE_THRESHOLD:.4f}
    &nbsp; · &nbsp;

    Similarity ≥
    {SIMILARITY_THRESHOLD:.4f}
    &nbsp; · &nbsp;

    Margin ≥
    {MARGIN_THRESHOLD:.4f}
    &nbsp; · &nbsp;

    CNN/centroid agreement required.

</div>
""")


# ============================================================
# 19. LIVE INFERENCE
# ============================================================

render_html("""
<section class="section" id="inference">

    <div class="section-number">04 · LIVE INFERENCE</div>

    <h2 class="section-title">
        Classify a bird
    </h2>

    <p class="section-description">
        Upload an image and run the trained CNN together
        with the open-set validation layer.
    </p>

</section>
""")


# ============================================================
# 20. UPLOAD AREA
# ============================================================

upload_col, image_col = st.columns(
    [1, 1],
    gap="large"
)

with upload_col:

    render_html("""
    <div class="inference-box">

        <div class="upload-title">
            Upload bird image
        </div>

        <div class="upload-subtitle">
            Supported formats: JPG, JPEG, PNG, WEBP
        </div>

    </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        label_visibility="collapsed"
    )

    classify_button = st.button(
        "🐦  CLASSIFY IMAGE",
        use_container_width=True
    )


# ============================================================
# 21. IMAGE PREVIEW
# ============================================================

image = None

with image_col:

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            )

            st.image(
                image,
                caption="Input image",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Could not open image: {error}"
            )

    else:

        render_html("""
        <div class="inference-box"
             style="min-height:280px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    text-align:center;">

            <div>
                <div style="font-size:45px;">
                    🐦
                </div>

                <div style="color:#8994a3;
                            margin-top:15px;">
                    Upload an image to begin inference
                </div>
            </div>

        </div>
        """)


# ============================================================
# 22. RUN CLASSIFICATION
# ============================================================

if classify_button:

    if image is None:

        st.warning(
            "Please upload a bird image first."
        )

    else:

        with st.spinner(
            "Running CNN inference..."
        ):

            try:

                result = predict_bird(
                    image=image,
                    model=model,
                    centroids=centroids,
                    class_names=class_names,
                    feature_extractor=feature_extractor,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    margin_threshold=MARGIN_THRESHOLD
                )

                st.session_state[
                    "prediction_result"
                ] = result

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    str(error)
                )


# ============================================================
# 23. DISPLAY RESULT
# ============================================================

if (
    "prediction_result"
    in st.session_state
):

    result = st.session_state[
        "prediction_result"
    ]

    predicted_name = escape(
        result["predicted_name"]
    )

    centroid_name = escape(
        result["centroid_name"]
    )

    confidence_percent = (
        result["confidence"] * 100
    )

    similarity_percent = (
        result["similarity"] * 100
    )

    margin_percent = (
        result["margin"] * 100
    )

    if result["accepted"]:

        render_html(f"""
        <div class="result-success">

            <div class="result-status">
                ✓ RECOGNIZED SPECIES
            </div>

            <div class="result-name">
                {predicted_name}
            </div>

            <div class="measure-grid">

                <div class="measure">
                    <div class="measure-label">
                        CNN confidence
                    </div>
                    <div class="measure-value">
                        {confidence_percent:.2f}%
                    </div>
                </div>

                <div class="measure">
                    <div class="measure-label">
                        Centroid similarity
                    </div>
                    <div class="measure-value">
                        {similarity_percent:.2f}%
                    </div>
                </div>

                <div class="measure">
                    <div class="measure-label">
                        Prediction margin
                    </div>
                    <div class="measure-value">
                        {margin_percent:.2f}%
                    </div>
                </div>

            </div>

        </div>
        """)

    else:

        render_html(f"""
        <div class="result-unknown">

            <div class="result-status">
                ! OPEN-SET REJECTION
            </div>

            <div class="result-name">
                Unknown / Uncertain
            </div>

            <p style="color:#8994a3;
                      line-height:1.7;">
                The CNN predicted
                <strong>{predicted_name}</strong>,
                but the complete decision rule did not
                accept the image as a sufficiently reliable
                known species.
            </p>

        </div>
        """)


    # ========================================================
    # DECISION CHECKS
    # ========================================================

    render_html(f"""
    <div class="check-grid">

        <div class="check
            {'pass' if result['confidence_pass'] else 'fail'}">

            {'✓' if result['confidence_pass'] else '✕'}
            Confidence
        </div>

        <div class="check
            {'pass' if result['similarity_pass'] else 'fail'}">

            {'✓' if result['similarity_pass'] else '✕'}
            Similarity
        </div>

        <div class="check
            {'pass' if result['margin_pass'] else 'fail'}">

            {'✓' if result['margin_pass'] else '✕'}
            Margin
        </div>

        <div class="check
            {'pass' if result['agreement'] else 'fail'}">

            {'✓' if result['agreement'] else '✕'}
            Agreement
        </div>

    </div>
    """)


    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    with st.expander(
        "View detailed prediction analysis"
    ):

        st.write(
            "### CNN / Centroid Analysis"
        )

        detail_col1, detail_col2 = st.columns(
            2
        )

        with detail_col1:

            st.write(
                f"**CNN prediction:** "
                f"{result['predicted_name']}"
            )

            st.write(
                f"**Nearest centroid:** "
                f"{result['centroid_name']}"
            )

            st.write(
                f"**Agreement:** "
                f"{'YES' if result['agreement'] else 'NO'}"
            )

        with detail_col2:

            st.write(
                f"**Confidence:** "
                f"{result['confidence']:.6f}"
            )

            st.write(
                f"**Similarity:** "
                f"{result['similarity']:.6f}"
            )

            st.write(
                f"**Margin:** "
                f"{result['margin']:.6f}"
            )

        st.write(
            "### Top 3 CNN Predictions"
        )

        for prediction in result[
            "top_predictions"
        ]:

            name = escape(
                prediction["class_name"]
            )

            confidence = (
                prediction["confidence"]
                * 100
            )

            render_html(f"""
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                padding:12px 14px;
                margin:7px 0;
                border:1px solid #222b38;
                border-radius:9px;
                background:#0d1118;
            ">

                <span style="
                    color:#edf2f7;
                    font-size:13px;
                ">
                    #{prediction["rank"]}
                    &nbsp; {name}
                </span>

                <span style="
                    color:#62e6a4;
                    font-family:'DM Mono',monospace;
                    font-size:12px;
                ">
                    {confidence:.2f}%
                </span>

            </div>
            """)


# ============================================================
# 24. CLASS LIST
# ============================================================

render_html("""
<section class="section" id="classes">

    <div class="section-number">05 · CLASS SPACE</div>

    <h2 class="section-title">
        Supported species
    </h2>

    <p class="section-description">
        Bird species available in the trained classifier.
    </p>

</section>
""")


# Build class cards dynamically

class_cards = []

for index, class_name in enumerate(
    class_names,
    start=1
):

    safe_name = escape(
        str(class_name)
    )

    class_cards.append(
        f"""
        <div class="class-card">

            <div class="class-number">
                {index:02d}
            </div>

            {safe_name}

        </div>
        """
    )


render_html(
    '<div class="class-grid">'
    + "".join(class_cards)
    + '</div>'
)


# ============================================================
# 25. SYSTEM NOTES
# ============================================================

render_html("""
<section class="section">

    <div class="section-number">
        06 · SYSTEM NOTES
    </div>

    <h2 class="section-title">
        How the decision works
    </h2>

    <p class="section-description">
        BirdVision does not rely only on the highest CNN
        probability. The prediction is checked using multiple
        independent signals.
    </p>

    <div class="card-grid">

        <div class="card">
            <div class="card-number">SIGNAL 01</div>

            <h3>Confidence</h3>

            <p>
                Measures how strongly the CNN favors the
                top predicted bird species.
            </p>
        </div>

        <div class="card">
            <div class="card-number">SIGNAL 02</div>

            <h3>Similarity</h3>

            <p>
                Compares the extracted feature representation
                against the learned class centroids.
            </p>
        </div>

        <div class="card">
            <div class="card-number">SIGNAL 03</div>

            <h3>Margin</h3>

            <p>
                Measures the difference between the first and
                second highest CNN predictions.
            </p>
        </div>

    </div>

</section>
""")


# ============================================================
# 26. FINAL DECISION RULE
# ============================================================

render_html("""
<section class="section">

    <div class="section-number">
        07 · DECISION RULE
    </div>

    <h2 class="section-title">
        Four signals. One decision.
    </h2>

</section>
""")


render_html(f"""
<div class="note">

    <strong>Image accepted as a known bird when:</strong>

    <br><br>

    CNN confidence ≥
    <strong>{CONFIDENCE_THRESHOLD:.4f}</strong>

    <br>

    AND centroid similarity ≥
    <strong>{SIMILARITY_THRESHOLD:.4f}</strong>

    <br>

    AND prediction margin ≥
    <strong>{MARGIN_THRESHOLD:.4f}</strong>

    <br>

    AND CNN prediction agrees with the nearest centroid.

</div>
""")


# ============================================================
# 27. FOOTER
# ============================================================

render_html(f"""
<div class="footer">

    <div>
        BIRDVISION AI · CNN CLASSIFICATION
    </div>

    <div>
        {len(class_names)} CLASSES · 224×224
    </div>

</div>
""")