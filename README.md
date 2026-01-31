# Diabetes Prediction using KNN and Naive Bayes

This project demonstrates the use of **Machine Learning classification algorithms** to predict whether a person is diabetic based on medical attributes.  
The model is trained and evaluated using the **Diabetes dataset (`diabetes.csv`)** with **K-Nearest Neighbors (KNN)** and **Naive Bayes (NB)** algorithms.

---

## 📌 Project Overview

- Dataset: `diabetes.csv`
- Problem Type: Binary Classification
- Algorithms Used:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (NB)
- Goal: Predict diabetes outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 📂 Dataset Description

The dataset contains medical diagnostic measurements for patients.

### Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target:
- **Outcome**
  - `0` → Non-Diabetic  
  - `1` → Diabetic

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib / Seaborn (for visualization)
- Scikit-learn

---

## 🧠 Machine Learning Models

### 1️⃣ K-Nearest Neighbors (KNN)
- Distance-based algorithm
- Classifies data based on nearest neighbors
- Requires feature scaling

### 2️⃣ Naive Bayes (NB)
- Probabilistic classifier
- Assumes feature independence
- Fast and efficient for classification tasks

---

## 🔄 Workflow

1. Load and explore the dataset
2. Handle missing values (if any)
3. Split data into training and testing sets
4. Feature scaling (for KNN)
5. Train KNN and Naive Bayes models
6. Evaluate models using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
7. Compare model performance

---

## 📊 Model Evaluation

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

📌 Results & Conclusion

Both KNN and Naive Bayes models successfully classify diabetes cases.

KNN performance depends on the value of K and feature scaling.

Naive Bayes is faster and works well with smaller datasets.

Model comparison helps in selecting the best classifier for prediction.

🚀 Future Improvements

Hyperparameter tuning (GridSearchCV)

Add more ML models (Logistic Regression, Random Forest)

Deploy as a web app using Flask or Streamlit

Handle class imbalance

👤 Author

Sanket Mistari
Data Science & Machine Learning Enthusiast


