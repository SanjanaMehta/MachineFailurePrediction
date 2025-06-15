# 🏭 Machine Failure Prediction

## Overview

This project focuses on predicting machine failures using sensor data collected from manufacturing processes. It involves exploratory data analysis (EDA), feature engineering, and the development of classification models such as **Decision Trees** and **Random Forest Classifiers** to predict the likelihood of machine failures.

---

## 📂 Dataset

* **Rows:** 10,000 instances
* **Features:** 14 columns, including:

  * **Air temperature \[K]**
  * **Process temperature \[K]**
  * **Rotational speed \[rpm]**
  * **Torque \[Nm]**
  * **Tool wear \[min]**
  * **Target (Failure / No Failure)**
  * **Failure Type (Various failure modes)**

---

## 🔍 Problem Statement

To predict whether a machine will fail based on operational parameters and sensor readings in order to enable preventive maintenance and avoid unexpected downtime.

---

## 🔧 Steps Performed

### 1. **Exploratory Data Analysis (EDA)**

* Inspected feature distributions.
* Identified outliers using percentile limits for `Rotational speed [rpm]`.
* Created a new feature: **Temp difference = Process temp - Air temp**.
* Visualized failure distributions using count plots and heatmaps.

### 2. **Data Cleaning**

* Removed duplicates and inconsistent records (e.g., failures labeled as 'No Failure' when the target was 1).

### 3. **Feature Selection**

* Selected 5 important features based on correlation analysis:

  * Process temperature \[K]
  * Air temperature \[K]
  * Rotational speed \[rpm]
  * Torque \[Nm]
  * Tool wear \[min]

### 4. **Model Building**

* **Decision Tree Classifier**:

  * Trained with depth control (max\_depth=2).
  * Achieved basic predictive accuracy.
* **Random Forest Classifier**:

  * Trained with 100 estimators.
  * Higher predictive performance compared to the decision tree.
  * **Accuracy printed on test data**.

### 5. **Model Evaluation**

* **Confusion Matrix** visualized.
* Individual **Decision Trees from the Random Forest** plotted.
* **Partial Dependence Plots (PDPs)** used to interpret the influence of features on the prediction.

---

## 📈 Results

* **Random Forest Accuracy**: (Printed in notebook — e.g., \~98% depending on split)
* **Top Features**: Torque, Tool wear, Air temperature.

---

## 🚀 Conclusion

The Random Forest model showed high accuracy and robustness in predicting machine failures based on critical operational parameters. This model can assist manufacturers in predictive maintenance strategies, reducing downtime and maintenance costs.

---

## 📌 Requirements

```bash
pandas
numpy
seaborn
matplotlib
scikit-learn
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📚 Future Work

* Hyperparameter tuning.
* Cross-validation.
* Deployment using Flask/Django.
* Real-time data integration.

