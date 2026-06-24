"""
Streamlit Dashboard for MVTec XAI Anomaly Detection.

This dashboard does ZERO heavy ML computation itself.
All predictions, Grad-CAM, and LIME are computed in a subprocess
(compute_xai.py) to avoid Python 3.13 + loky segfaults on macOS.
The dashboard only loads and displays saved images/results.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import DATA_DIR, CHECKPOINTS_DIR, IMAGE_SIZE

# Path to the python interpreter and XAI computation script
PYTHON = sys.executable
COMPUTE_SCRIPT = str(SRC_DIR / "compute_xai.py")
TEMP_DIR = Path("outputs/temp")
XAI_OUTPUT_DIR = TEMP_DIR / "xai_results"


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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0; padding-bottom: 0;
    }
    .pred-card {
        border-radius: 8px; padding: 12px;
        text-align: center; border: 2px solid #e9ecef; margin-bottom: 8px;
    }
    .pred-card-normal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-color: #28a745;
    }
    .pred-card-anomalous {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-color: #dc3545;
    }
    .pred-label { font-size: 14px; font-weight: 700; color: #343a40; margin-bottom: 4px; }
    .pred-result { font-size: 1.4rem; font-weight: 800; }
    .pred-result-normal { color: #155724; }
    .pred-result-anomalous { color: #721c24; }
    .pred-meta { font-size: 11px; color: #666; margin-top: 2px; }
    .streamlit-expanderHeader { font-size: 14px; font-weight: 600; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> Path:
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    temp_path = TEMP_DIR / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def get_thresholded_overlay(image_path: Path, heatmap: np.ndarray, threshold_pct: int) -> np.ndarray:
    pil_image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(pil_image)
    thr = np.percentile(heatmap, threshold_pct)
    binary_mask = (heatmap >= thr).astype(np.uint8)
    mask_overlay = img_np.copy()
    mask_overlay[binary_mask == 1] = [230, 57, 70]
    blended = cv2.addWeighted(mask_overlay, 0.45, img_np, 0.55, 0)
    return blended


def run_xai_pipeline(image_path: Path, gradcam_threshold: int, lime_samples: int, lime_features: int):
    """Run ALL XAI computations in a subprocess. Returns results dict."""
    XAI_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    result = subprocess.run(
        [PYTHON, COMPUTE_SCRIPT,
         str(image_path),
         str(gradcam_threshold),
         str(lime_samples),
         str(lime_features),
         str(XAI_OUTPUT_DIR)],
        capture_output=True,
        text=True,
        cwd=str(SRC_DIR.parent),
        timeout=120,
    )

    results_file = XAI_OUTPUT_DIR / "results.json"
    if not results_file.exists():
        raise RuntimeError(f"XAI computation failed.\nstderr: {result.stderr}\nstdout: {result.stdout}")

    with open(results_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("<p class='hero-title'>🔍 MVTec XAI — Explainable Anomaly Detection</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 📷 Image Input")
input_mode = st.sidebar.radio("Choose source:", ["Pick from Test Set", "Upload Image"], label_visibility="collapsed")
image_path = None

if input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload a bottle image", type=["png", "jpg", "jpeg"])
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
st.sidebar.markdown("## ⚙️ XAI Settings")
gradcam_threshold = st.sidebar.slider("Grad-CAM Threshold (%)", 50, 95, 75, step=5)
lime_samples = st.sidebar.slider("LIME Perturbations", 200, 1000, 500, step=100)
lime_features = st.sidebar.slider("LIME Top Superpixels", 1, 10, 4)

# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
if image_path is None:
    st.info("👈 **Select an image** in the sidebar to run the pipeline.")
else:
    # Run ALL computations in a subprocess
    with st.spinner("⏳ Running both models + Grad-CAM + LIME... (this takes ~15 seconds)"):
        try:
            results = run_xai_pipeline(image_path, gradcam_threshold, lime_samples, lime_features)
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
            st.stop()

    # ==========================================
    # 3-Column Layout: [Input] [ResNet-50] [CNN]
    # ==========================================
    col_input, col_resnet, col_cnn = st.columns([1, 1.5, 1.5], gap="large")

    # COLUMN 1: Input Image
    with col_input:
        st.image(str(image_path), width="stretch", caption="Input Image")
        if input_mode == "Pick from Test Set":
            if selected_category == "good":
                st.success("📋 **Truth:** Normal")
            else:
                st.error(f"📋 **Truth:** Anomalous ({selected_category})")

    # COLUMNS 2 & 3: Model Results
    models_config = [
        (col_resnet, "resnet50", "ResNet-50 (Transfer Learning)"),
        (col_cnn, "cnn", "Custom 2D CNN (From Scratch)")
    ]

    for col, m_type, m_name in models_config:
        with col:
            r = results[m_type]
            pred_class = r["pred_class"]
            confidence = r["confidence"]
            latency = r["latency"]

            # --- Prediction Card ---
            is_anomalous = pred_class == "anomalous"
            card_class = "pred-card-anomalous" if is_anomalous else "pred-card-normal"
            result_class = "pred-result-anomalous" if is_anomalous else "pred-result-normal"
            icon = "🚨" if is_anomalous else "✅"

            st.markdown(f"""
            <div class='pred-card {card_class}'>
                <div class='pred-label'>{m_name}</div>
                <div class='pred-result {result_class}'>{icon} {pred_class.upper()} ({confidence:.1%})</div>
                <div class='pred-meta'>{latency:.0f}ms latency</div>
            </div>
            """, unsafe_allow_html=True)

            # --- Grad-CAM Expander ---
            with st.expander("📍 Grad-CAM Explanation (Gradients)", expanded=True):
                gradcam_path = XAI_OUTPUT_DIR / f"gradcam_{m_type}.png"
                heatmap_path = XAI_OUTPUT_DIR / f"heatmap_{m_type}.npy"

                if gradcam_path.exists() and heatmap_path.exists():
                    overlay = np.array(Image.open(gradcam_path))
                    heatmap = np.load(heatmap_path)
                    thresholded = get_thresholded_overlay(image_path, heatmap, gradcam_threshold)

                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.image(overlay, width="stretch", caption="Overlay")
                    with gc2:
                        st.image(thresholded, width="stretch", caption=f"Top {100-gradcam_threshold}% Focus")
                else:
                    st.error("Grad-CAM output not found.")

            # --- LIME Expander ---
            with st.expander("🧩 LIME Explanation (Perturbation)", expanded=False):
                lime_path = XAI_OUTPUT_DIR / f"lime_{m_type}.png"
                if lime_path.exists():
                    lime_img = np.array(Image.open(lime_path))
                    st.image(lime_img, width="stretch",
                             caption=f"Top {lime_features} important regions")
                else:
                    st.error("LIME output not found.")

    # =======================================================================
    # BOTTOM: Model Comparison Table
    # =======================================================================
    with st.expander("📈 View Model Metrics & Comparison", expanded=False):
        import torch
        cnn_val_acc = "75.60%"
        try:
            cnn_ckpt = CHECKPOINTS_DIR / "best_model_cnn.pt"
            if cnn_ckpt.exists():
                payload = torch.load(cnn_ckpt, map_location="cpu", weights_only=False)
                val_acc = payload.get("val_acc", None)
                if val_acc:
                    cnn_val_acc = f"{val_acc:.2%}"
        except Exception:
            pass

        resnet_r = results.get("resnet50", {})
        cnn_r = results.get("cnn", {})

        comparison_data = {
            "Metric": ["Parameters", "Backbone", "Validation Accuracy", "Inference Latency"],
            "ResNet-50": ["23,512,130", "ImageNet V2 (pretrained)", "97.59%",
                          f"{resnet_r.get('latency', 0):.0f} ms"],
            "Custom 2D CNN": ["106,306", "None (from scratch)", cnn_val_acc,
                              f"{cnn_r.get('latency', 0):.0f} ms"],
        }
        st.table(comparison_data)
