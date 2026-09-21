# BAR-Net: A Boundary-Aware Refinement Network for Medical Image Segmentation

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Web-green)
![License](https://img.shields.io/badge/license-MIT-blue)

**MedVision AI Diagnostic Viewer** is the official web interface and PyTorch implementation of **BAR-Net**. This platform provides an ultra-fast, professional-grade clinical interface for uploading and diagnosing utilizing medical imaging datasets (both 2D `.npy` images and 3D `.h5` Volumetric Scans).

---

## 📋 Table of Contents
1. [Model Architecture](#-model-architecture)
2. [Project Structure](#-project-structure)
3. [Installation](#-installation)
4. [Dataset Preparation](#-dataset-preparation)
5. [How to Run (Web Interface)](#-how-to-run-web-interface)
6. [Training & Testing](#-training--testing)
7. [Authors & Non-Copyright](#-authors--non-copyright)

---

## 🧠 Model Architecture

**BAR-Net (Boundary-Aware Refinement Network)** is built upon a hybrid vision-transformer architecture optimized for volumetric medical imaging.

- **Core Encoder:** Uses a Swin Transformer backbone (Shifted Window attention) to capture long-range global dependencies without the heavy computational cost of traditional vision transformers.
- **Skip Connections & Expand Decoder:** Employs a U-Net-like symmetrical layout. Feature maps from the multi-scale Swin Encoder are projected across skip connections via CNN convolutions into an expansive decoder to restore the dense spatial resolution.
- **Boundary-Aware Refinement (BAR):** (Specific to this paper) Integrates multi-scale boundary information to force the model to cleanly delineate edges of difficult internal organs (like heart, kidney, liver, etc.) which traditional CNNs often blur.

---

## 📁 Project Structure

```text
BAR-Net/
│
├── checkpoints/             # Directory to store trained PyTorch model weights (.pth)
├── configs/                 # SwinTransformer configuration files
├── data_test/               # Temporary testing datasets and raw files
├── datasets/                # Contains dataloaders and pre-processed dataset folders
│   ├── dataset_synapse.py   # Dataset pipeline for loading .h5 and .npz
│
├── lists/                   # Text files mapping train/val split indices
├── networks/                # Deep Learning Architecture
│   ├── swin_transformer_unet_skip_expand_decoder_sys.py # Core BAR-Net Backbone Model
│
├── web/                     # Frontend and Backend for Clinical Viewer Web App
│   ├── app.py               # FastAPI server to handle AI Inference dynamically
│   └── static/              # HTML/CSS/JS frontend logic and styling elements
│
├── requirements.txt         # Project dependencies
├── train.py / trainer.py    # Main training loop scripts
├── test.py                  # Evaluation benchmark scripts
└── README.md                # Project documentation
```

---

## 🚀 Installation

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment (Anaconda or Venv).

1. Clone the repository:
   ```bash
   git clone [YOUR_GITHUB_LINK_HERE]
   cd BAR-Net
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For PyTorch, we recommend installing the CUDA version matching your GPU driver from [pytorch.org](https://pytorch.org/).)*

---

## 💾 Dataset Preparation

To train or reproduce our results, you need to download the medical datasets and structure them correctly.

1. **Download Dataset:**
   👉 *[Insert Dataset Download Link Here]*

2. **Extract and map paths:** 
   Place the volumetric images inside the preconfigured raw directory. Ensure that the dataset consists of `.npz` slices for training and `.npy.h5` volumes for validation.
   
   Expected dataset path structure:
   ```text
   BAR-Net/data/Synapse/
   │
   ├── train_npz/            # 2D training arrays (.npz)
   └── test_vol_h5/          # 3D validation volumes (.npy.h5)
   ```

*(You must update your paths in `train.sh` or when calling `train.py` directly).*

---

## 🖥️ How to Run (Web Interface)

The web system integrates the pre-trained `best_model.pth` and dynamically performs brain/clinical segmentation purely through a beautiful graphical interface.

1. Start the FastAPI backend server:
   ```bash
   python web/app.py
   ```
   
2. Access the Application:
   Open your preferred web browser and navigate to: **`http://localhost:8000`**

3. **Usage Flow**:
   - Upload any `.h5` 3D scan or 2D image via the Sidebar.
   - Wait 1-2 seconds for the automated pipeline to trigger inference.
   - Use the slider to explore individual transverse slices.
   - Use the *Clinical Observations* panel to write diagnosis.
   - Export an Ultra-HD Professional Report clicking `Save Report (PNG)`.

---

## 🔬 Training & Testing

**Training the model:**
```bash
python train.py --dataset Synapse --root_path data/Synapse --max_epochs 150 --output_dir ./checkpoints/
```

**Testing the model:**
```bash
python test.py --dataset Synapse --is_savenii --volume_path data/Synapse/test_vol_h5
```

---

## 🧑‍⚕️ Authors & Non-Copyright

**Research Project:** BARNet: A Boundary-Aware Refinement Network for Medical Image Segmentation  
**Conference/Year:** [Conference Name], 2026

**Authors:**
- Ngoc-Hai Le
- Gia-Kiet Tran
- Tam-Nhu Thai Ngoc
- Duy-Vinh Duong
- Van-Dung Hoang

*This project is completely open source and operates under a **No Copyright** model. It is intended for free academic usage, research scaling, and medical non-profit applications without licensing restrictions.*
