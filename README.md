# 🏥 Predictive Healthcare Intelligence System  

![Python](https://img.shields.io/badge/Python-3.10-blue)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)  
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)  
![Status](https://img.shields.io/badge/Status-Completed-success)  

---

## Overview  
This project presents a data-driven healthcare intelligence system designed to support early intervention and improve clinical decision-making.  

It combines machine learning and data analytics to identify high-risk patients, predict 30-day hospital readmissions, and forecast emergency demand. The aim is to shift healthcare systems from reactive care to proactive patient management.

---

##  Problem Statement  
Healthcare systems often face challenges such as:
- Late identification of high-risk patients  
- High rates of avoidable hospital readmissions  
- Poor visibility into future patient demand  

These issues lead to increased operational pressure and reduced quality of care. This project addresses these gaps using predictive modeling.

---

## Key Features  
- High-risk patient identification  
- 30-day readmission prediction  
- Emergency demand forecasting  
- End-to-end data cleaning and preprocessing pipeline  
- Feature engineering based on clinical and utilisation data  
- Model comparison and evaluation  

---

##  Target Engineering (High-Risk Definition)  
A patient is classified as high-risk if any of the following conditions are met:  
- Readmission within 30 days  
- Length of stay greater than 7 days  
- ICU admission  
- Three or more comorbidities  
- Three or more prior admissions  

This ensures both clinical severity and utilisation behaviour are captured.

---

## ⚙️ Tech Stack  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Jupyter Notebook / VS Code  
- MLflow
---

## Modeling Approach  

This project consists of **three predictive modeling tasks**:

### 1️⃣ High-Risk Patient Identification  
A classification model was developed to identify patients at risk of complications.

- Models used: Logistic Regression, Random Forest, XGBoost  
- Final model: XGBoost  

---

### 2️⃣ 30-Day Readmission Prediction  
A model was built to predict whether a patient is likely to be readmitted within 30 days.

- Focus: Early intervention and discharge planning  
- Evaluated using Precision, Recall, and F1-score  

---

### 3️⃣ Emergency Demand Forecasting  
A forecasting approach was used to analyse patterns in patient inflow.

- Focus: Predicting future demand trends  
- Supports hospital resource planning and staffing decisions  

---

## 📈 Key Insights  
- Patient history and utilisation patterns are strong predictors of risk  
- High-risk patients often exhibit repeated admissions and multiple comorbidities  
- Data quality was suitable for modeling after preprocessing  

---

## 🏥 Clinical Impact  
- Early identification of high-risk patients  
- Improved prioritisation of care  
- Reduction in avoidable readmissions  
- Better planning of hospital resources  
---

## 🤝 Contribution  
Contributions are welcome. Feel free to fork the repository and submit a pull request.

---

## 📄 License  
This project is for educational and research purposes.

---

## 👤 Author  
**Ekemini Mbabah**  
Data Scientist | Machine Learning Enthusiast  
