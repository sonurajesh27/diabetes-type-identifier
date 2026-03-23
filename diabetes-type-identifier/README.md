# 🩺 AI Diabetes Type Identifier

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end AI-powered web application that classifies whether a patient has **Type 1 Diabetes**, **Type 2 Diabetes**, or **No Diabetes** based on clinical data — using Machine Learning with a clean, interactive frontend.

---

## 📸 Features

- 🔬 **Manual Entry** — Enter patient vitals (age, glucose, BMI, insulin, etc.) and get instant predictions
- 📂 **CSV Batch Upload** — Upload a CSV file to predict for multiple patients at once
- ⚙️ **Train / Retrain** — Train the model on your own labeled dataset directly from the UI
- 📊 **Confidence Scores** — Doughnut chart showing probability for each class
- 📈 **Feature Importance** — Bar chart showing which features influenced the prediction most
- 📄 **Export as PDF** — Print/save the prediction result as a PDF
- 🔄 **Drag & Drop** — Drag and drop CSV files for upload
- 🧠 **SMOTE** — Handles class imbalance automatically during training
- 🌐 **REST API** — Clean Flask API with `/predict`, `/train`, `/health`, `/features` endpoints

---

## 🧠 How It Works

### Structured Data Model
The app uses a **Random Forest** or **Decision Tree** classifier trained on clinical features:

| Feature | Description |
|---|---|
| Glucose | Plasma glucose concentration (mg/dL) |
| BMI | Body mass index (kg/m²) |
| Age | Patient age in years |
| Insulin | 2-hour serum insulin (μU/mL) |
| BloodPressure | Diastolic blood pressure (mmHg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Pregnancies | Number of pregnancies |
| DiabetesPedigreeFunction | Genetic diabetes risk score |

### Handling Imbalanced Data
- **SMOTE** (Synthetic Minority Oversampling Technique) is applied during training
- `class_weight="balanced"` is set on the classifier
- Evaluated using F1-score, Precision, Recall, and Confusion Matrix

### Preprocessing Pipeline
1. Replace biologically invalid zeros with `NaN`
2. Impute missing values using **median imputation**
3. Normalize features using **StandardScaler**

### Output Labels
| Code | Label |
|---|---|
| 0 | No Diabetes |
| 1 | Type 1 Diabetes |
| 2 | Type 2 Diabetes |

---

## 📁 Project Structure

```
diabetes-type-identifier/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── model.py            # ML model training & inference
│   ├── preprocessing.py    # Data preprocessing pipeline
│   ├── train.py            # Standalone CLI training script
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Main UI
│   ├── style.css           # Styling
│   └── script.js           # Frontend logic + Chart.js
├── data/
│   └── sample_diabetes.csv # Sample labeled dataset
├── models/                 # Saved model artifacts (auto-generated)
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or 3.12 (recommended)
- pip

### 1. Clone the repository
```bash
git clone https://github.com/sonurajesh27/diabetes-type-identifier.git
cd diabetes-type-identifier
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Train the model
```bash
python backend/train.py --model random_forest
```

### 4. Start the backend
```bash
python backend/app.py
```
Backend runs at `http://localhost:5000`

### 5. Open the frontend
```bash
python -m http.server 8080 --directory frontend
```
Then open `http://localhost:8080` in your browser.

---

## 🌐 API Reference

### `GET /health`
Returns backend and model status.
```json
{
  "status": "ok",
  "model_trained": true,
  "version": "1.0.0"
}
```

### `POST /train`
Train the model. Accepts `multipart/form-data`.

| Field | Type | Description |
|---|---|---|
| file | CSV (optional) | Labeled training data. Uses sample data if omitted. |
| model_type | string | `random_forest` or `decision_tree` |

```json
{
  "success": true,
  "metrics": {
    "f1_score": 0.91,
    "model_type": "random_forest",
    "training_samples": 1200
  }
}
```

### `POST /predict`
Predict diabetes type for a single patient (JSON) or batch (CSV file).

**Single patient (JSON):**
```json
{
  "Age": 45,
  "Glucose": 148,
  "BMI": 33.6,
  "Insulin": 0,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Pregnancies": 6,
  "DiabetesPedigreeFunction": 0.627
}
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "Type 2 Diabetes",
  "confidence": 0.87,
  "confidence_scores": {
    "No Diabetes": 0.05,
    "Type 1 Diabetes": 0.08,
    "Type 2 Diabetes": 0.87
  },
  "feature_importance": {
    "Glucose": 0.32,
    "BMI": 0.21,
    "Age": 0.18,
    "Insulin": 0.12,
    "BloodPressure": 0.09,
    "DiabetesPedigreeFunction": 0.05,
    "SkinThickness": 0.02,
    "Pregnancies": 0.01
  }
}
```

### `GET /features`
Returns expected feature names and label mapping.

---

## 📊 Dataset

The sample dataset is based on the [PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

For better accuracy, download the full dataset from Kaggle, add an `Outcome` column with values `0`, `1`, or `2`, and upload it via the **Train Model** tab in the UI.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Backend | Python, Flask, Flask-CORS |
| ML | Scikit-learn (Random Forest, Decision Tree) |
| Imbalance | imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| Serialization | Joblib |

---

## 📦 Requirements

```
flask==3.0.3
flask-cors==4.0.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
imbalanced-learn==0.12.3
joblib==1.4.2
werkzeug==3.0.3
```

---

## ⚠️ Disclaimer

This application is for **research and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Rajesh Sonu**
GitHub: [@sonurajesh27](https://github.com/sonurajesh27)
