# 🚦 Accident Severity Prediction — Machine Learning Mini Project  
📚 **Subject:** MLT (Machine Learning Techniques)  
📊 **Mini Project — All 12 Practicals Completed**

---

## 🧠 Project Overview

This project predicts **traffic accident severity (1 → 4)** based on:

- Weather conditions  
- Road conditions  
- Timestamp  
- Short textual accident description

The output Severity levels are:

| Severity | Meaning |
|--------|--------|
| 1 | Very Low impact |
| 2 | Low impact |
| 3 | Moderate impact |
| 4 | High impact |

We use **Machine Learning + NLP + PCA + Boosting** to achieve strong performance.

---

# 🗂️ Practicals Completed (MLT Syllabus — Correct Order)

## ✔️ 1. Data Ingestion
- Loaded dataset from Kaggle: **US Accidents (2016–2023)**  
- 7M+ records, 47 features  
- Imported into Pandas DataFrame for processing

---

## ✔️ 2. Data Dictionary
- Feature metadata: type, subtype, description
- Marked relevance (Low / Medium / High)
- Dropped **20+ low relevance columns**
- Identified target variable: **Severity**

---

## ✔️ 3. Data Cleaning
Performed full quality cleaning:
- Removed NULL values
- Mode/Mean imputation
- Dropped duplicates
- Removed low variance columns  
- Filtered noisy weather conditions

---

## ✔️ 4. Exploratory Data Analysis (EDA)
Visual insights:
- Severity class distribution
- Weather vs severity trends
- Humidity, wind, rainfall patterns
- Correlation heatmaps

---

## ✔️ 5. Data Preprocessing
- Label Encoding
- Timestamp Feature Engineering:
  - Hour / Minute
- Outlier handling:
  - Winsorizer (IQR)
  - Quantile capping
- Balanced data:
  - Class-wise undersampling (~60k each)
- NLP embeddings:
  - **Sentence Transformer**
  - Model: `all-MiniLM-L6-v2` → 384D embeddings
- PCA:
  - PCA: 384 → 100 components
  - Dimensionality reduction
  - Variance explained study

---

## ✔️ 6. K-Nearest Neighbors (KNN)
- Baseline model
- Hyperparameter K tuning
- Observations:
  - Very slow on large dataset
  - Performed poorly with high-dimensional embeddings

---

## ✔️ 7. Naive Bayes
- Used Gaussian & Multinomial NB
- Required normal distributions
- Best for text-only feature
- Limited for full dataset

---

## ✔️ 8. Decision Tree & Random Forest
- Gini / Entropy splits
- Feature importance analysis
- Random Forest improved stability
- Handles categorical + numeric well

---

## ✔️ 9. Bagging Classifier
- Ensemble of decision trees
- Reduces variance
- Parallel bootstrap sampling
- Improves generalization

---

## ✔️ 10. Boosting
Used 3 methods:
- AdaBoost
- HistGradientBoosting
- LightGBM (Final)

Boosting improved class prediction:
- Especially for Severity = 3 and 4

---

## ✔️ 11. Stacking
- Multiple base models (DT, RF, XGB)
- Meta model on top
- Out-of-fold predictions
- Combined strengths of learners

---

## ✔️ 12. End-to-End ML Pipeline
  - Cleaning → Preprocess → Model predict
  - Saved preprocessing artifacts
  - Deployment with **Streamlit UI**

## 🧠 Model Explanation

The project predicts **accident severity (1–4)** using both **structured features** (weather + road) and **unstructured text** (Description column).


### Why this works
- **SentenceTransformer (all-MiniLM-L6-v2)** extracts semantic meaning from text  
  → understands "heavy fog", "dense rain", "collision"
- **PCA (384 → 100)** reduces noise and prevents overfitting
- **LightGBM** handles large tabular data, missing values, and mixed features efficiently

This combination gives:
✔ Better accuracy  
✔ Faster inference  
✔ Strong generalization  
✔ Balanced performance across all classes

---

## 📊 Model Evaluation

We evaluate using **multiclass metrics**:

- **Accuracy**
- **Precision / Recall / F1**
- **Macro AUC** → treats all classes equally
- **Weighted AUC** → weighted by class frequency
- **Confusion Matrix**

### What AUC means
- **Macro AUC** = average performance per class  
- **Weighted AUC** = average performance weighted by #samples

Result pattern (example):
- Classes 1 & 2 → easier to detect
- Class 3 & 4 → harder, but improved after boosting

---

## 🏁 Key Takeaways

- Text + numerical weather features give the best results  
- PCA reduces dimensionality and speeds the model  
- LightGBM outperforms KNN, Naive Bayes, and Decision Tree  
- Probability output helps interpret risk levels
