# 🌿 Greenhouse Gas Emission Source Classifier

A machine learning project that classifies emission sources (Agriculture, Industry, Energy, Transportation) based on CO₂, N₂O, and CH₄ measurements.

> **Funded by TÜBİTAK 2209-A** — Undergraduate Research Support Programme

---

## 📌 Problem Statement

Given greenhouse gas emission readings (CO₂, N₂O, CH₄), the model predicts which sector the emissions originate from:

| Class | Sector |
|-------|--------|
| 0 | Agriculture (Tarım) |
| 1 | Industry (Sanayi) |
| 2 | Energy Production (Enerji Üretimi) |
| 3 | Transportation (Ulaşım) |

---

## 🤖 Models & Results

Three classification algorithms were evaluated using **5-Fold Stratified Cross-Validation**:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM (RBF Kernel) | **99.2%** | - | - | - |
| KNN (k=2–12) | - | - | - | - |
| Naive Bayes | - | - | - | - |

> Best result: **SVM with RBF Kernel — 99.2% accuracy**

---

## 🗂️ Project Structure

```
emisyon-siniflandirma/
│
├── emisyon_tahmin.py        # Tkinter desktop GUI for real-time prediction
├── SMV.py                   # Model comparison & cross-validation script
├── dataset.xlsx             # Dataset with CO2, N2O, CH4 features
├── siniflandirma_sonuclari.xlsx  # Full model evaluation results
└── bg.png                   # GUI background image
```

---

## 🖥️ Desktop Application

A **Tkinter-based GUI** was built for real-time emission source prediction. Users enter CO₂, N₂O, and CH₄ values and the trained SVM model instantly predicts the emission sector.

![GUI Screenshot](bg.png)

---

## ⚙️ Installation & Usage

### Requirements
```bash
pip install pandas scikit-learn openpyxl matplotlib pillow
```

### Run the GUI App
```bash
python emisyon_tahmin.py
```

### Run Model Comparison
```bash
python SMV.py
```
> Outputs `siniflandirma_sonuclari.xlsx` with full cross-validation metrics.

---

## 🔬 Methodology

- **Preprocessing:** StandardScaler normalization via sklearn Pipeline
- **Validation:** StratifiedKFold (5 splits, random_state=42)
- **Metrics:** Accuracy, Precision (macro), Recall (macro), F1-Score (macro)
- **KNN sweep:** k values from 2 to 12 evaluated individually

---

## 🛠️ Tech Stack

- Python 3.x
- scikit-learn
- pandas / NumPy
- Matplotlib
- Tkinter + Pillow

---

## 👤 Author

**Ergün Hoşnut**  
Mathematics Student — Ege University, Faculty of Science  
Principal Investigator, TÜBİTAK 2209-A Project  
📧 ergunhosnut2001@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ergunhosnut9)
