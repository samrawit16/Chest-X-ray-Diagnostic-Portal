# Chest X-ray Diagnostic Portal 

An educational prototype for binary classification of chest X-rays (**Normal vs. Pneumonia**). This repository contains **Part B of the AkiraChix DAS Medical Imaging ML Project**, featuring an interactive web application that loads a pre-trained model architecture and executes deep learning inference using **pure NumPy**—entirely independent of heavy frameworks like TensorFlow or PyTorch.

---

##  Key Features

* **Pure NumPy Inference:** Runs a dense deep neural network forward pass (`Linear ➔ ReLU ➔ Linear ➔ ReLU ➔ Linear ➔ Sigmoid`) using only NumPy arrays.
* **Deterministic Preprocessing:** Handles image pipelines (`Grayscale ➔ Resize to 64x64 ➔ Normalization ➔ Standarization`) exactly mirroring the training notebook.
* **Honest Uncertainty Reporting:** Flags ambiguous scans falling inside an uncertainty threshold (35% - 65%) as `UNSURE` instead of forcing an unreliable diagnosis.
* **Streamlit UI:** A lightweight, interactive web interface for uploading X-ray images, analyzing results, and inspecting raw prediction states.

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **Framework:** Streamlit
* **Math & Image Ops:** NumPy, Pillow (PIL)
* **Model Format:** Custom JSON weights file

---

##  Getting Started

### 1. Prerequisites
Ensure you have Python installed on your system. Install the minimal package requirements via pip:

```bash
pip install streamlit numpy pillow
```

### 2. Model Weight Dependency
The application expects a JSON file containing the trained weights, biases, and standardization parameters (`mu` and `sd`) named exactly **`chest_xray_model.json`**. 

Place this file directly into the root folder alongside `app.py`.

```text
 your-repo-name/
├── 📄 app.py
└── 📄 chest_xray_model.json   <-- Core model file goes here
```

### 3. Launching the App
Run the local development server with Streamlit:

```bash
streamlit run app.py
```

Open the local network URL provided in your terminal (usually `http://localhost:8501`) to start uploading and evaluating X-rays.

---

## ⚙️ Model & Processing Matrix

| Layer / Parameter | Specification |
| :--- | :--- |
| **Input Shape** | 64 × 64 pixels (Grayscale) |
| **Flattened Features** | 4,096 Input Neurons |
| **Hidden Layer 1** | 64 Neurons with ReLU activation |
| **Hidden Layer 2** | 16 Neurons with ReLU activation |
| **Output Layer** | 1 Neuron with Sigmoid activation |
| **Uncertainty Band** | Predictions between `0.35` and `0.65` flag as `UNSURE` |

---

##  Medical Disclaimer

**This software is an educational prototype and is NOT a certified medical device.** 
The code and its predictions are meant solely for student machine learning demonstrations and must never be used to make real clinical decisions, diagnoses, or treatment plans. Always consult a qualified medical professional for health evaluations.
