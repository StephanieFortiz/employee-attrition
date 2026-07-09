# 🔍 Employee Attrition Predictor

Predicts whether an employee is at risk of leaving the company,
based on the IBM HR Analytics dataset.

## 🚀 Live Demo
Try it on Render -> https://employee-attrition-dw7i.onrender.com

## 📊 Model
- **Algorithm:** Logistic Regression with class_weight={0:0.2, 1:0.8}
- **Features:** 14 selected from 43 original features
- **Test Accuracy:** 75.6%
- **Recall (attrition class):** 0.67
- **F1 Score (attrition class):** 0.47

## 🔑 Most influential features
- `TotalWorkingYears` — more experience = less likely to leave
- `JobRole_Laboratory Technician` — higher attrition risk
- `JobRole_Sales Representative` — higher attrition risk
- `MaritalStatus_Single` — higher attrition risk
- `OverTime` — working overtime increases risk

## 📁 Project Structure
employee-attrition/
├── app.py              ← Gradio app
├── model/              ← trained model and preprocessors
├── data/               ← IBM HR Analytics dataset
├── notebooks/          ← full analysis notebook
└── requirements.txt

## 🛠️ Run locally
```bash
git clone https://github.com/tu-usuario/employee-attrition
cd employee-attrition
pip install -r requirements.txt
python app.py
```

## 📚 Dataset
[IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — publicly available on Kaggle.