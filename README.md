# Prima-Indian-Diabetes-dataset
# 🩺 Diabetes Risk Prediction Using Machine Learning: Pima Indians Healthcare Dataset

## Executive Summary

### Business Problem

Diabetes is a major global health challenge that requires early detection to prevent severe complications and improve patient outcomes. Traditional diagnosis often requires clinical testing and medical expertise, creating a need for data-driven tools that can assist healthcare professionals in identifying high-risk individuals.

This project develops an end-to-end machine learning classification system capable of predicting whether a patient is likely to have diabetes based on clinical measurements including glucose concentration, BMI, blood pressure, insulin levels, age, pregnancy history, and family diabetes history.

---

## Dataset Overview

The project uses the **Pima Indians Diabetes Dataset**, containing health records of 768 female patients of Pima Indian heritage with the objective of predicting diabetes status.

The dataset contains key medical features including:

- Pregnancies
- Glucose concentration
- Blood pressure
- Skin thickness
- Insulin levels
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age

The target variable is **Outcome**, where:
- 0 → Non-diabetic
- 1 → Diabetic

---

## Exploratory Data Analysis (EDA)

A comprehensive exploratory analysis was performed to understand the data quality, distributions, relationships between variables, and potential predictors of diabetes.

### Key Findings

- Several medical measurements contained unrealistic zero values, which were treated as missing data and cleaned appropriately.
- Glucose level showed the strongest relationship with diabetes diagnosis, with diabetic patients generally displaying significantly higher glucose concentrations.
- Higher BMI categories were associated with increased diabetes prevalence, highlighting the impact of obesity on diabetes risk.
- Older age groups showed higher diabetes occurrence compared with younger groups.
- Patients with higher pregnancy counts showed an increased prevalence of diabetes.
- Statistical hypothesis testing was used to determine whether differences between diabetic and non-diabetic groups were statistically significant.
- Outlier analysis was performed to identify unusual medical measurements and assess their clinical relevance.

---

## Feature Engineering

Domain knowledge was incorporated to create new features that improve model performance:

- **Age Group:** Categorized patients into clinically meaningful age ranges.
- **Pregnancy Group:** Grouped patients based on pregnancy count to capture changes in diabetes risk.
- **Glucose × BMI Interaction:** Combined glucose and BMI measurements to model the joint effect of elevated blood sugar and obesity.

The impact of engineered features was initially validated using logistic regression before inclusion in the final machine learning pipeline.

---

## Machine Learning Pipeline

The predictive pipeline includes:

- Automated feature engineering using `FunctionTransformer`
- Numerical feature standardization using `StandardScaler`
- Categorical encoding using `OneHotEncoder`
- Class imbalance handling using SMOTE oversampling
- Cross-validation-based hyperparameter optimization
- Model evaluation using accuracy, precision, and recall metrics

---

## Model Comparison

The following machine learning algorithms were evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Extreme Gradient Boosting (XGBoost)

Model selection was performed using cross-validation with a strong focus on maximizing **recall**, as correctly identifying diabetic patients is more critical than minimizing false positives.

---

## Best Performing Model: XGBoost

After hyperparameter optimization, **XGBoost** achieved the highest predictive performance.

### Selected Hyperparameters

- Number of estimators: **100**
- Scale positive weight: **1**
- Evaluation metric: **Log Loss**
- Random seed: **42**

### Test Set Performance

| Metric | Score |
|---|---:|
| Accuracy | **87%** |
| Precision | **88%** |
| Recall | **87%** |

The model achieved a strong balance between identifying diabetic patients and maintaining prediction reliability.

---

## Deployment

The final solution was deployed as an interactive **Streamlit dashboard**, allowing users to:

- Enter patient health information.
- Receive real-time diabetes risk predictions.
- Explore interactive visualizations from the exploratory data analysis.
- Understand key factors associated with diabetes risk.

---

## Conclusion

This project demonstrates how machine learning can transform healthcare data into actionable insights. Through rigorous data cleaning, statistical analysis, domain-driven feature engineering, class imbalance handling, and model optimization, the XGBoost model successfully achieved high predictive accuracy while maintaining strong sensitivity toward detecting diabetic cases.

The project highlights the potential of machine learning as a clinical decision-support tool that can assist healthcare professionals with early diabetes screening.

---

## Recommendations and Future Work

Future improvements could include:

- Applying Explainable AI techniques such as SHAP to interpret individual predictions.
- Testing advanced gradient boosting algorithms such as LightGBM and CatBoost.
- Incorporating additional clinical and lifestyle variables to improve predictive accuracy.
- Expanding the model using larger and more diverse patient populations.
- Deploying the application to cloud platforms for public accessibility.
- Implementing continuous model monitoring and retraining using new patient data.

---

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- SMOTE (Imbalanced-Learn)
- Matplotlib & Seaborn
- Streamlit
- Git & GitHub

---

## Author

**Sibusiso Mathebula**

MSc/Applied Data Science & Astrophysics Enthusiast | Machine Learning | Data Analytics | Scientific Computing
