# iris-species-classification
Iris Species Classification using Morphological Features – Math Club Induction Task
# Iris Species Classification Using Morphological Features

## INFINITRIX – The Math Club  
### Data Science & AI Induction Task (2025–26)

**Submitted by:** Vijay Chamyal

---

## Problem Statement
This project addresses **Problem Statement 3: Iris Species Classification Using Morphological Features** as part of the INFINITRIX Math Club Induction Task.

The objective is to develop a supervised machine learning model that classifies iris flowers into three species—Setosa, Versicolor, and Virginica—using morphological measurements.

---

## Dataset
- Iris Dataset (Scikit-learn)
- 150 samples (50 per species)
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width


---

## Method Overview
- Exploratory data analysis was performed to study feature relationships.
- Data was split into **80% training** and **20% testing** sets.
- Feature scaling (standardization) was applied for linear models.

### Models Used
- **Logistic Regression (Linear Model)**
- **Decision Tree Classifier (Non-Linear Model)**
---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Key Results
- Logistic Regression Accuracy: **100%**
- Decision Tree Accuracy: **100%**
- Most class overlap occurs between *Versicolor* and *Virginica*.

---

## Bonus Objectives
- **Reduced Data Training:** Model performance dropped significantly when trained on only 10% of the data, highlighting the importance of sufficient training samples.
- **Error Analysis:** Misclassifications in reduced-data experiments occurred mainly in overlapping feature regions.
- **Linear vs Non-Linear Comparison:** Both models performed equally well, indicating near-linear separability of the dataset, with Decision Trees offering better interpretability.

