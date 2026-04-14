#  AI-Based Early Detection of Pancreatic Cancer from CT Scans

##  Overview

This project presents an end-to-end **AI-powered pipeline** for early detection of pancreatic cancer using CT scan data. It combines **deep learning**, **medical image processing**, and **radiomics-based machine learning** to identify abnormalities in pancreatic tissue.

The system is designed to:

* Segment the pancreas from CT scans
* Detect potential tumor regions
* Extract radiomics features
* Predict cancer risk using machine learning

---

## Project Architecture

```
CT Scan (.nii.gz)
        ↓
Preprocessing (Normalization, Slicing)
        ↓
U-Net Model → Pancreas Segmentation
        ↓
Tumor Detection Model
        ↓
Radiomics Feature Extraction
        ↓
XGBoost Classifier → Cancer Risk Prediction
```

---

##  Features

*  **Pancreas Segmentation** using U-Net (MONAI + PyTorch)
*  **Tumor Detection Model** for abnormal tissue identification
*  **Radiomics Feature Extraction** (texture, shape, intensity)
*  **Machine Learning Model** (XGBoost) for cancer risk prediction
*  End-to-end pipeline from raw CT scans to prediction

---

##  Project Structure

```
Pancreatic-Cancer-AI/
│
├── data_processing/        # Preprocessing & dataset creation
├── training/               # Model training scripts (U-Net, tumor model)
├── radiomics/              # Feature extraction + ML pipeline
├── inference/              # Full CT scan prediction pipeline
├── notebooks/              # Experimentation & visualization
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create virtual environment

```bash
python -m venv medical_ai
medical_ai\Scripts\activate   # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses the **Medical Segmentation Decathlon – Pancreas Dataset**.

🔗 Dataset link:
https://medicaldecathlon.com/

> ⚠️ Note: Due to size constraints, datasets are not included in this repository.

---

## 🧪 How to Run

### 1️⃣ Preprocess dataset

```bash
python data_processing/preprocess_dataset.py
```

### 2️⃣ Train segmentation model

```bash
python training/train_unet.py
```

### 3️⃣ Train tumor detection model

```bash
python training/train_tumor_model.py
```

### 4️⃣ Extract radiomics features

```bash
python radiomics/radiomics_feature_extraction.py
```

### 5️⃣ Train cancer risk predictor

```bash
python radiomics/train_xgboost_model.py
```

---

##  Results (Example)

* Accurate pancreas segmentation using U-Net
* Detection of abnormal regions in CT slices
* Radiomics features successfully extracted
* XGBoost model predicts cancer risk probability

---

##  Technologies Used

* Python
* PyTorch
* MONAI
* OpenCV
* Nibabel
* PyRadiomics
* Scikit-learn
* XGBoost

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.
It is **not intended for clinical use or medical diagnosis**.

---

## 📌 Future Improvements

* 3D U-Net for volumetric segmentation
* Transformer-based medical imaging models
* Integration with clinical metadata
* Deployment as a web-based diagnostic tool

---

## 🙌 Author

**Siju Johnson**
AI / Machine Learning Engineer


