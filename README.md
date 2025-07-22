# 🧠 Circular Failure Prediction in ICU Patients

This repository contains the code and models developed for the project:  
**"Predicting Circulatory Failure in ICU Patients with Deep Sequential Models"**.

> 🚑 The goal is to anticipate circulatory deterioration in critically ill patients up to **8 hours in advance**, using ICU data from the **MIMIC-IV** database.

---

## 📊 Overview

Traditional ICU alarms based on static thresholds are prone to high false positive rates, leading to alarm fatigue.  
We propose a machine learning-based early warning system using **deep learning models** (Transformer, GRU, and LSTM) to enhance predictive performance.

---

## 🧠 Models

We implemented and compared the following neural architectures:

- **SimpleTransformer** – best overall performance (AUROC ≈ 0.98, AUPRC ≈ 0.88)
- **LSTMNet**
- **GRUNet**

Pre-trained model weights (`.pt`) are available for 3 random seeds each.

---

## 🧪 Dataset

- The dataset used is **MIMIC-IV** (version 2.2), not publicly distributed here.
- Data was processed into 5-minute resolution windows with up to **7 days (2016 steps)** per ICU stay.
- Each timestep includes **620 features** from:
  - Vital signs
  - Lab tests
  - Medication (vasopressor usage excluded to avoid data leakage)

> ⚠️ Access to MIMIC-IV requires credentialed PhysioNet access and completion of human subjects research training.

---

## 📁 Repository Structure

```
CIRCULARFAILURE/
├── data/
│   ├── features_names.csv
│   ├── metrics_completas.csv
│   └── subject_stay_map.csv
├── figures/
│   └── [PR, ROC, ConfusionMatrix...]
├── models/
│   └── *.pt (pretrained models by seed)
├── notebooks/
│   ├── extract_stayids_and_split.ipynb
│   ├── preprocess.ipynb
│   ├── graphs.ipynb
│   └── models.ipynb
├── environment.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 🔁 Using `conda`
```bash
conda env create -f environment.yml
conda activate circulatory-failure
```

### 🐍 Or with `pip`
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

1. Ensure you have the processed HDF5 dataset (`dataset.h5`) in the `output/` or `data/` folder.
2. Run the preprocessing pipeline (`preprocess.ipynb`) if needed.
3. Run `models.ipynb` to load, train, or evaluate models.
4. Use `graphs.ipynb` to generate performance visualizations (PRC, ROC, etc.).

---

## 📈 Results Summary

| Model       | AUROC | AUPRC | F1    | Precision | Recall |
|-------------|-------|-------|-------|-----------|--------|
| Transformer | 0.976 | 0.884 | 0.804 | 0.844     | 0.768  |
| LSTM        | 0.969 | 0.858 | 0.745 | 0.839     | 0.671  |
| GRU         | 0.969 | 0.839 | 0.782 | 0.812     | 0.755  |

---

## 🔓 License

This code is licensed under the MIT License.  
**Note:** The dataset (MIMIC-IV) is not included and must be accessed through [PhysioNet](https://physionet.org/content/mimiciv/).

---