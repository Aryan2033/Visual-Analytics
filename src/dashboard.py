"""
Streamlit Dashboard for MVTec XAI Anomaly Detection.

Flow:
  1. User uploads a bottle image (or picks a sample).
  2. Both models (ResNet-50 + Custom 2D CNN) predict normal vs anomalous.
  3. Grad-CAM and LIME explain WHY each model made its decision.
  4. Everything is shown side-by-side on a single page.
"""

import sys
from pathlib import Path

# Add src/ directory to system path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image

from config import DATA_DIR, CHECKPOINTS_DIR, IMAGE_SIZE
from predict import predict_image, CLASS_NAMES
from gradcam_explain import compute_gradcam
from lime_explain import compute_lime


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MVTec XAI — Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header banner */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #6c757d;
        margin-top: 0;
    }

    /* Prediction cards */
    .pred-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 2px solid #e9ecef;
        margin-bottom: 16px;
    }
    .pred-card-normal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-color: #28a745;
    }
    .pred-card-anomalous {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-color: #dc3545;
    }
    .pred-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6c757d;
        margin-bottom: 8px;
    }
    .pred-result {
        font-size: 1.6rem;
        font-weight: 800;
    }
    .pred-result-normal { color: #155724; }
    .pred-result-anomalous { color: #721c24; }
    .pred-conf {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .pred-conf-normal { color: #28a745; }
    .pred-conf-anomalous { color: #dc3545; }
    .pred-meta {
        font-size: 12px;
        color: #888;
        margin-top: 6px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #343a40;
        margin-top: 12px;
        margin-bottom: 4px;
        padding-bottom: 8px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }

    /* Explanation text */
    .explain-text {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 14px;
        color: #495057;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> Path:
    temp_dir = Path("outputs/temp")
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_path = temp_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def get_thresholded_gradcam_overlay(image_path: Path, heatmap: np.ndarray, threshold_pct: int) -> np.ndarray:
    pil_image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(pil_image)
    thr = np.percentile(heatmap, threshold_pct)
    binary_mask = (heatmap >= thr).astype(np.uint8)
    mask_overlay = img_np.copy()
    mask_overlay[binary_mask == 1] = [230, 57, 70]
    alpha = 0.45
    blended = cv2.addWeighted(mask_overlay, alpha, img_np, 1 - alpha, 0)
    return blended


@st.cache_data
def run_cached_gradcam(_image_path_str: str, model_type: str):
    return compute_gradcam(Path(_image_path_str), model_type=model_type)


@st.cache_data
def run_cached_lime(_image_path_str: str, num_samples: int, num_features: int, model_type: str):
    return compute_lime(Path(_image_path_str), num_samples=num_samples, num_features=num_features, model_type=model_type)


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("<p class='hero-title'>🔍 MVTec XAI — Anomaly Detection</p>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Upload a bottle image → Both models predict Normal or Anomalous → See <b>why</b> with Grad-CAM & LIME explanations</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR — Image Input + XAI Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 📷 Image Input")
input_mode = st.sidebar.radio("Choose source:", ["Upload Image", "Pick from Test Set"], label_visibility="collapsed")
image_path = None

if input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload a bottle image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_path = save_uploaded_file(uploaded_file)
else:
    bottle_test_dir = DATA_DIR / "bottle" / "test"
    if bottle_test_dir.exists():
        categories = sorted([d.name for d in bottle_test_dir.iterdir() if d.is_dir()])
        selected_category = st.sidebar.selectbox("Defect Category", categories)
        category_dir = bottle_test_dir / selected_category
        images = sorted([f.name for f in category_dir.glob("*.png")])
        selected_image = st.sidebar.selectbox("Image", images)
        image_path = category_dir / selected_image
    else:
        st.sidebar.error("Dataset not found under data/bottle/test/")

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Explainability Settings")
gradcam_threshold = st.sidebar.slider("Grad-CAM Highlight Percentile", 50, 95, 75, step=5,
                                       help="Higher = stricter focus on hottest regions")
lime_samples = st.sidebar.slider("LIME Perturbations", 200, 1000, 500, step=100,
                                  help="More samples = more accurate but slower")
lime_features = st.sidebar.slider("LIME Top Superpixels", 1, 10, 4,
                                   help="Number of important regions to highlight")

# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
if image_path is None:
    st.markdown("---")
    st.info("👈 **Upload a bottle image** or **pick one from the test set** in the sidebar to get started.")
    st.markdown("""
    ### How it works
    1. **Upload** a bottle image (normal or defective)
    2. **Both models** (ResNet-50 & Custom 2D CNN) will classify it
    3. **Grad-CAM** shows *where* the model looked (heatmap)
    4. **LIME** shows *which regions* influenced the decision (superpixels)
    """)
else:
    # =======================================================================
    # SECTION 1: Input Image Preview
    # =======================================================================
    col_preview, col_spacer, col_info = st.columns([1.5, 0.2, 3])

    with col_preview:
        st.image(str(image_path), use_container_width=True)

    with col_info:
        st.markdown(f"**File:** `{image_path.name}`")
        if input_mode == "Pick from Test Set":
            if selected_category == "good":
                st.success("📋 **Ground Truth:** Normal (good)")
            else:
                st.error(f"📋 **Ground Truth:** Anomalous — *{selected_category}*")
        else:
            st.info("📋 Uploaded image — ground truth unknown")

    st.markdown("---")

    # =======================================================================
    # SECTION 2: Both Models' Predictions — Side by Side
    # =======================================================================
    st.markdown("<p class='section-header'>📊 Model Predictions</p>", unsafe_allow_html=True)

    col_resnet, col_cnn = st.columns(2)

    predictions = {}
    for col, m_type, m_name in [
        (col_resnet, "resnet50", "ResNet-50 (Transfer Learning)"),
        (col_cnn, "cnn", "Custom 2D CNN (From Scratch)")
    ]:
        with col:
            start_time = time.time()
            try:
                pred_class, confidence = predict_image(image_path, model_type=m_type)
                latency = (time.time() - start_time) * 1000
                predictions[m_type] = (pred_class, confidence, latency)

                is_anomalous = pred_class == "anomalous"
                card_class = "pred-card-anomalous" if is_anomalous else "pred-card-normal"
                result_class = "pred-result-anomalous" if is_anomalous else "pred-result-normal"
                conf_class = "pred-conf-anomalous" if is_anomalous else "pred-conf-normal"
                icon = "🚨" if is_anomalous else "✅"

                st.markdown(f"""
                <div class='pred-card {card_class}'>
                    <div class='pred-label'>{m_name}</div>
                    <div class='pred-result {result_class}'>{icon} {pred_class.upper()}</div>
                    <div class='pred-conf {conf_class}'>{confidence:.1%}</div>
                    <div class='pred-meta'>confidence · {latency:.0f}ms latency</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ {m_name}: {e}")

    st.markdown("---")

    # =======================================================================
    # SECTION 3: WHY is it anomalous? — Grad-CAM Explanations
    # =======================================================================
    st.markdown("<p class='section-header'>📍 Why this prediction? — Grad-CAM</p>", unsafe_allow_html=True)
    st.markdown("""<div class='explain-text'>
        <b>Grad-CAM</b> looks <i>inside</i> the model to find which spatial regions of the image
        most strongly activated the predicted class. Red/warm areas = high importance.
    </div>""", unsafe_allow_html=True)

    col_gc_resnet, col_gc_cnn = st.columns(2)
    image_path_str = str(image_path)

    for col, m_type, m_name in [
        (col_gc_resnet, "resnet50", "ResNet-50"),
        (col_gc_cnn, "cnn", "Custom 2D CNN")
    ]:
        with col:
            if m_type not in predictions:
                continue
            st.markdown(f"**{m_name}**")
            with st.spinner(f"Computing Grad-CAM for {m_name}..."):
                try:
                    heatmap, overlay, _, _ = run_cached_gradcam(image_path_str, model_type=m_type)
                    thresholded = get_thresholded_gradcam_overlay(image_path, heatmap, gradcam_threshold)

                    sub1, sub2 = st.columns(2)
                    with sub1:
                        st.image(overlay, use_container_width=True, caption="Heatmap Overlay")
                    with sub2:
                        st.image(thresholded, use_container_width=True, caption=f"Top {100-gradcam_threshold}% Focus")
                except Exception as e:
                    st.error(f"Grad-CAM error: {e}")

    st.markdown("---")

    # =======================================================================
    # SECTION 4: WHY is it anomalous? — LIME Explanations
    # =======================================================================
    st.markdown("<p class='section-header'>🧩 Why this prediction? — LIME</p>", unsafe_allow_html=True)
    st.markdown(f"""<div class='explain-text'>
        <b>LIME</b> treats the model as a black box. It hides parts of the image ({lime_samples}× perturbations)
        and watches how the prediction changes, revealing which regions matter most. Yellow boundaries = important superpixels.
    </div>""", unsafe_allow_html=True)

    col_lime_resnet, col_lime_cnn = st.columns(2)

    for col, m_type, m_name in [
        (col_lime_resnet, "resnet50", "ResNet-50"),
        (col_lime_cnn, "cnn", "Custom 2D CNN")
    ]:
        with col:
            if m_type not in predictions:
                continue
            st.markdown(f"**{m_name}**")
            with st.spinner(f"Running LIME for {m_name} ({lime_samples} perturbations)..."):
                try:
                    _, lime_overlay, _, _ = run_cached_lime(
                        image_path_str,
                        num_samples=lime_samples,
                        num_features=lime_features,
                        model_type=m_type
                    )
                    st.image(lime_overlay, use_container_width=True,
                             caption=f"Top {lime_features} important superpixels")
                except Exception as e:
                    st.error(f"LIME error: {e}")

    st.markdown("---")

    # =======================================================================
    # SECTION 5: Model Comparison Table
    # =======================================================================
    st.markdown("<p class='section-header'>📈 Model Comparison</p>", unsafe_allow_html=True)

    # Load CNN val accuracy from checkpoint
    cnn_val_acc = "75.60%"
    try:
        import torch
        cnn_ckpt = CHECKPOINTS_DIR / "best_model_cnn.pt"
        if cnn_ckpt.exists():
            payload = torch.load(cnn_ckpt, map_location="cpu", weights_only=False)
            val_acc = payload.get("val_acc", None)
            if val_acc:
                cnn_val_acc = f"{val_acc:.2%}"
    except Exception:
        pass

    resnet_latency = f"{predictions['resnet50'][2]:.0f} ms" if 'resnet50' in predictions else "—"
    cnn_latency = f"{predictions['cnn'][2]:.0f} ms" if 'cnn' in predictions else "—"

    comparison_data = {
        "": ["Parameters", "Backbone", "Validation Accuracy", "Inference Latency"],
        "ResNet-50": ["23,512,130", "ImageNet V2 (pretrained)", "97.59%", resnet_latency],
        "Custom 2D CNN": ["106,306", "None (from scratch)", cnn_val_acc, cnn_latency],
    }
    st.table(comparison_data)

    # =======================================================================
    # FOOTER
    # =======================================================================
    st.markdown("---")
    st.caption("MVTec XAI — Visual Analytics Project, Summer 2026 · ResNet-50 (Transfer Learning) vs Custom 2D CNN (Scratch)")
