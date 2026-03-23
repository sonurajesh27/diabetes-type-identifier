# AI Diabetes Type Identifier

An end-to-end web application that uses Machine Learning to classify whether a patient has **Type 1 Diabetes**, **Type 2 Diabetes**, or **No Diabetes** based on clinical data.

---

## Features

- Manual patient data entry (age, glucose, BMI, insulin, etc.)
- Batch prediction via CSV upload
- Train / retrain the model with your own labeled dataset
- Confidence scores for each class
- Feature importance bar chart
- Export prediction as PDF (browser print)
- Handles missing values, class imbalance (SMOTE), and feature scaling

---

## Dataset

Uses a format based on the [PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

Required columns:

| Column | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mmHg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| Outcome | 0 = No Diabetes, 1 = Type 1, 2 = Type 2 (training only) |

---

## Model

- **Random Forest** (default) or **Decision Tree**
- SMOTE for class imbalance handling
- StandardScaler + median imputation for preprocessing
- Evaluated with F1-score, precision, recall, confusion matrix

---

## Installation

```bash
cd diabetes-type-identifier/backend
pip install -r requirements.txt
```

---

## Running Locally

**1. Start the backend:**
```bash
cd diabetes-type-identifier/backend
python app.py
```
Backend runs at `http://localhost:5000`

**2. Open the frontend:**

Open `diabetes-type-identifier/frontend/index.html` in your browser, or serve it:
```bash
cd diabetes-type-identifier/frontend
python -m http.server 8080
```
Then visit `http://localhost:8080`

**3. Train the model (optional CLI):**
```bash
cd diabetes-type-identifier/backend
python train.py --model random_forest
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check backend + model status |
| POST | `/train` | Train model (form-data: `file`, `model_type`) |
| POST | `/predict` | Predict (JSON body or form-data `file` for batch) |
| GET | `/features` | List expected features and label map |

### Sample predict request
```json
POST /predict
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

### Sample response
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
    "Age": 0.18
  }
}
```

---

## Project Structure

```
diabetes-type-identifier/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── model.py            # ML model training & inference
│   ├── preprocessing.py    # Data preprocessing pipeline
│   ├── train.py            # Standalone training script
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   └── sample_diabetes.csv
├── models/                 # Saved model artifacts (git-ignored)
├── README.md
├── LICENSE
└── .gitignore
```

---

## Disclaimer

For research and educational purposes only. Not a substitute for professional medical advice.
