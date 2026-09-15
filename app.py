"""Chest X-ray Diagnostic Portal — Normal vs Pneumonia
Part B of the AkiraChix DAS Medical Imaging ML Project.
Loads the trained model (chest_xray_model.json) and runs predictions in pure NumPy
(no TensorFlow/PyTorch needed).
Run with:  streamlit run app.py
"""
import json
import io
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Chest X-ray Diagnostic Portal", page_icon="🩺", layout="centered")
st.title("🩺 Chest X-ray Diagnostic Portal")
st.caption("Normal vs Pneumonia — AkiraChix DAS Medical Imaging ML Project")

st.warning(
    "⚠️ **Disclaimer:** This tool is an educational prototype and **NOT** a medical device. "
    "Its predictions must never be used for real clinical decisions. Always consult a qualified doctor."
)

IMG_SIZE = (64, 64)          
MODEL_PATH = "chest_xray_model.json"
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.35, 0.65


@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH) as f:
            m = json.load(f)
        return m, False
    except Exception:
        return None, True


model, is_dummy = load_model()


def predict(x):
    """x: float32 array shape (1, 4096), already standardized.
    Dense net: Linear-ReLU-Dropout-Linear-ReLU-Dropout-Linear-Sigmoid.
    Dropout is identity at inference."""
    w1, b1 = np.array(model["w1"]), np.array(model["b1"])
    w2, b2 = np.array(model["w2"]), np.array(model["b2"])
    w3, b3 = np.array(model["w3"]), np.array(model["b3"])
    h1 = np.maximum(0, x @ w1.T + b1)       
    h2 = np.maximum(0, h1 @ w2.T + b2)        
    z = h2 @ w3.T + b3                        
    return 1.0 / (1.0 + np.exp(-z))            


def preprocess(img: Image.Image) -> np.ndarray:
    """MUST match the notebook preprocessing exactly:
    grayscale -> resize 64x64 -> /255 -> standardize -> flatten to (1, 4096)."""
    img = img.convert("L").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - model["mu"]) / (model["sd"] + 1e-7)
    return arr.reshape(1, -1)



with st.sidebar:
    st.header(" Model info")
    if is_dummy:
        st.error("Model file not found! Place `chest_xray_model.json` next to this app.")
        st.caption("Predictions are disabled until the model is present.")
    else:
        st.markdown(
            f"- **Task:** Binary classification (Normal / Pneumonia)\n"
            f"- **Input layer:** {IMG_SIZE[0]}×{IMG_SIZE[1]} grayscale = {IMG_SIZE[0]*IMG_SIZE[1]:,} inputs\n"
            f"- **Hidden layers:** 64 (ReLU) → 16 (ReLU)\n"
            f"- **Output layer:** 1 neuron (sigmoid)\n"
            f"- **Preprocessing:** grayscale, resize {IMG_SIZE[0]}×{IMG_SIZE[1]}, /255, standardize"
        )

st.header(" Upload a chest X-ray")
uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img.convert("L"), caption=uploaded.name, width=320)

    if not is_dummy and st.button("🔍 Analyse X-ray", type="primary"):
        with st.spinner("Running model..."):
            x = preprocess(img)
            prob = float(predict(x)[0][0])

        if UNCERTAIN_LOW < prob < UNCERTAIN_HIGH:
            label = "UNSURE"
        else:
            label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"

        if label == "PNEUMONIA":
            st.error(f"### Prediction: **PNEUMONIA** ({prob:.1%} confidence)")
        elif label == "NORMAL":
            st.success(f"### Prediction: **NORMAL** ({1 - prob:.1%} confidence)")
        else:
            st.warning(f"### Uncertain prediction ({prob:.1%} pneumonia probability)")
            st.markdown(
                "The model is not confident either way. This is reported honestly — "
                "**please consult a medical professional.**"
            )

        st.progress(prob, text=f"Pneumonia probability: {prob:.1%}")

        st.markdown("### What does this mean?")
        if label == "PNEUMONIA":
            st.markdown(
                "The model detected visual patterns commonly associated with **pneumonia**, "
                "such as cloudy/white areas in the lungs. In a real clinical setting, a doctor "
                "would review this alongside symptoms and other tests."
            )
        elif label == "NORMAL":
            st.markdown(
                "The model did not detect the patterns it associates with **pneumonia**. "
                "This does not rule out other conditions."
            )

        with st.expander("Show technical details"):
            st.json({
                "model": "chest_xray_model.json (dense net)",
                "input_shape": [1, IMG_SIZE[0] * IMG_SIZE[1]],
                "preprocessing": f"grayscale, resize {IMG_SIZE[0]}×{IMG_SIZE[1]}, /255, standardize (mu={model['mu']:.4f}, sd={model['sd']:.4f})",
                "pneumonia_probability": round(prob, 4),
                "prediction": label,
                "uncertainty_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
            })
elif is_dummy:
    st.info(" Model file missing — see the sidebar notice.")
else:
    st.info(" Upload an X-ray image to begin.")
