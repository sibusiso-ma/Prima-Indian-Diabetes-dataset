
# Streamlit dashboard for the Pima Indian Dataset
# Sibusiso Mathebula
# 15/04/2025



import streamlit as st
import pandas as pd
from eda_functions import *
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import shap

import sklearn
print(sklearn.__version__)
        



def create_features(X):
            X = X.copy()
            X['PregnancyGroup'] = pd.cut(
                X['Pregnancies'],
                bins=[-1,0,3,6,20],
                labels=['0','1-3','4-6','7+']
            )
            X['AgeGroup'] = pd.cut(
                X['Age'],
                bins=[20,30,40,50,60,80],
                labels=['20-30','31-40','41-50','51-60','61-80']
            )
            X['Glucose_BMI'] = X['Glucose'] * X['BMI']
            return X

feature_creator = FunctionTransformer(create_features, validate=False)

st.set_page_config(
    page_title="Diabetes AI Dashboard",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA",'Outlier detection and error analysis', "Clustering", "Prediction"])

st.sidebar.markdown("## ⚙️ Controls")


st.title("Diabetes Prediction Dashboard")

df = load_data("diabetes (1).csv")
if page == "EDA":
    

    # ✅ ADD HERE
    total_cases = len(df)
    positive_cases = df["Outcome"].sum()
    negative_cases = total_cases - positive_cases
    positive_rate = positive_cases / total_cases

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", total_cases)
    col2.metric("Positive Cases", int(positive_cases))
    col3.metric("Negative Cases", int(negative_cases))
    col4.metric("Positive Rate", f"{positive_rate:.2%}")


    show_head(df)
    show_statistics(df)
    show_missing(df)

    with st.expander("See raw data"):
        st.dataframe(df)

    

    outcome_summary = df["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Count")
    outcome_summary["Outcome"] = outcome_summary["Outcome"].map({0: "Negative", 1: "Positive"})
    outcome_summary["Percentage"] = outcome_summary["Count"] / len(df)

    st.subheader("Outcome Summary Table")
    st.dataframe(outcome_summary)
    
    
    

    with st.expander("See raw data"):
        st.dataframe(df)

    st.subheader("Outcome Distribution")
    fig1 = plot_outcome_distribution(df)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2 = correlation_heatmap(df)
    st.pyplot(fig2)



    feature = st.selectbox("Select Feature for distribution plotting ", df.drop(columns=['Outcome'],axis=1).columns)

    st.subheader(f"{feature}  ")
    fig3 = distribution(df[feature])
    st.pyplot(fig3)



    x_col = st.selectbox("Select X-axis",df.drop(columns=['Outcome'],axis=1).columns)
    y_col = st.selectbox("Select Y-axis", df.drop(columns=['Outcome'],axis=1).columns)

    feature = [x_col,y_col]

    fig = scatter_plot(df, x_col,y_col)
    st.pyplot(fig)
    



# --- Correlation Section ---
    st.markdown("### 📊 Correlation Analysis")

    corr_value = df[x_col].corr(df[y_col])
    st.write(f"Correlation: **{corr_value:.3f}**")

    # Interpretation
    if abs(corr_value) > 0.7:
        strength = "Strong"
    elif abs(corr_value) > 0.4:
        strength = "Moderate"
    elif abs(corr_value) > 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"

    direction = "positive" if corr_value > 0 else "negative"

    st.write(f" {strength} {direction} relationship")

    # Statistical significance
    from scipy.stats import pearsonr

    corr, p_value = pearsonr(df[x_col], df[y_col])
    st.write(f"P-value: {p_value:.5f}")

    if p_value < 0.05:
        st.success("Statistically significant")
    else:
        st.warning("Not statistically significant")

    fig = Pregnancy_dist(df)
    st.pyplot(fig)


    feature = st.selectbox("Select Feature for boxplot  ", df.drop(columns=['Outcome'],axis=1).columns)
    fig = boxplot(df,feature)
    st.pyplot(fig)

    feature = st.selectbox("Select Feature for violinplot  ",df.drop(columns=['Outcome'],axis=1).columns)
    fig = violinplot(df,feature)
    st.pyplot(fig)



    feature = st.selectbox("Select Feature for kdeplot  ", df.drop(columns=['Outcome'],axis=1).columns)
    fig = kdeplot(df,feature)
    st.pyplot(fig)

    st.subheader("Diabetes positive rates for different age groups ")

    fig = AgeGroup(df)
    st.pyplot(fig)

    st.subheader("Diabetes positive rates for different BMI categories ")

    fig = BMI_Category(df)
    st.pyplot(fig)

    st.subheader("Diabetes positive rates for different Pregnancy groups ")

    fig = PregnancyGroup(df)
    st.pyplot(fig)

if page =='Outlier detection and error analysis':
     
    from scipy.stats import zscore

    st.title(" Outlier Detection & Error Analysis")

    # =========================
    # 1. Load Data
    # =========================
   

    st.subheader(" Dataset Overview")
    st.write(df.describe())

    # =========================
    # 2. Define Columns
    # =========================
    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # =========================
    # 3. ERROR ANALYSIS (Invalid Zeros)
    # =========================
    st.subheader("Data Quality (Invalid Zeros)")

    zero_errors = (df[medical_cols] == 0).sum()

    st.write("Count of invalid zero values (should be treated as missing):")
    st.dataframe(zero_errors)

    st.metric("Total Invalid Cells", zero_errors.sum())

    # Optional replacement toggle
    if st.checkbox("Replace zeros with NaN (recommended)"):
        df[medical_cols] = df[medical_cols].replace(0, np.nan)
        st.success("Zeros replaced with NaN")

    # =========================
    # 4. IQR OUTLIER DETECTION
    # =========================
    st.subheader(" IQR Outlier Detection")

    def iqr_outliers(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return data[(data[col] < lower) | (data[col] > upper)]

    iqr_outlier_indices = set()

    for col in medical_cols:
        out_idx = iqr_outliers(df, col).index
        iqr_outlier_indices.update(out_idx)

    iqr_outliers_df = df.loc[list(iqr_outlier_indices)]

    st.write(f"Total IQR outliers: {len(iqr_outliers_df)}")
    st.dataframe(iqr_outliers_df)

    # =========================
    # 5. Z-SCORE OUTLIER DETECTION
    # =========================
    st.subheader(" Z-Score Outlier Detection (|z| > 3)")

    z_df = df[medical_cols].dropna()
    z_scores = z_df.apply(zscore)

    z_outliers_mask = (z_scores.abs() > 3).any(axis=1)
    z_outliers_df = df.loc[z_df[z_outliers_mask].index]

    st.write(f"Total Z-score outliers: {len(z_outliers_df)}")
    st.dataframe(z_outliers_df)

    # =========================
    # 6. COMBINED ANALYSIS
    # =========================
    st.subheader(" Combined Anomaly Analysis")

    iqr_set = set(iqr_outliers_df.index)
    z_set = set(z_outliers_df.index)

    both = iqr_set & z_set
    iqr_only = iqr_set - z_set
    z_only = z_set - iqr_set

    st.write(f" High-confidence anomalies (both methods): {len(both)}")
    st.write(f"IQR-only anomalies: {len(iqr_only)}")
    st.write(f" Z-score-only anomalies: {len(z_only)}")

    combined_df = df.loc[list(both)]
    st.dataframe(combined_df)

    # =========================
    # 7. FILTER OPTION
    # =========================
    st.subheader("🧹 Filter Dataset")

    option = st.radio(
        "Select dataset view:",
        ["Original Data", "Remove IQR Outliers", "Remove All Anomalies"]
    )

    if option == "Original Data":
        st.dataframe(df)

    elif option == "Remove IQR Outliers":
        clean_df = df.drop(index=iqr_outliers_df.index)
        st.dataframe(clean_df)

    elif option == "Remove All Anomalies":
        all_outliers = iqr_set | z_set
        clean_df = df.drop(index=list(all_outliers))
        st.dataframe(clean_df)

    # =========================
    # 8. Summary Metrics
    # =========================
    st.subheader("📊 Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("IQR Outliers", len(iqr_outliers_df))
    col3.metric("Z-score Outliers", len(z_outliers_df))

# ======================
# CLUSTERING PAGE
# ======================
if page == "Clustering":
    st.title("🔍 DBSCAN Clustering")

    features = st.multiselect(
        "Select features for clustering",
        df.columns.drop('Outcome')
    )

    eps = st.slider("Select eps", 0.1, 5.0, 0.5)
    min_samples = st.slider("Select min_samples", 2, 20, 5)

    if st.button("Run DBSCAN"):
        clustered_df = run_dbscan(df, features, eps, min_samples)
        st.write(clustered_df.head())

        if len(features) == 2:
            plot_dbscan(clustered_df, features)
            
            

# ======================
# PREDICTION PAGE
# ======================
if page == "Prediction":
    st.title("🧠 Diabetes Prediction")

    import joblib
    from sklearn.pipeline import Pipeline
    import shap
    import matplotlib.pyplot as plt

    # -------------------------
    # Load your trained pipeline
    # -------------------------
    pipeline = joblib.load("diabetes_model_only.pkl")

    # Create inference pipeline without SMOTE (already fitted)
    inference_pipeline = Pipeline([
        ('features', pipeline.named_steps['features']),
        ('preprocessing', pipeline.named_steps['preprocessing']),
        ('model', pipeline.named_steps['model'])
    ])

    # -------------------------
    # Collect user input
    # -------------------------
    input_data = {}
    for col in df.drop(columns=['Outcome']).columns:
        input_data[col] = st.number_input(
            f"Enter {col}",
            value=float(df[col].mean())
        )

    input_df = pd.DataFrame([input_data])

    # -------------------------
    # Transform input using fitted pipeline
    # -------------------------
    X_transformed = inference_pipeline.named_steps['preprocessing'].transform(
        inference_pipeline.named_steps['features'].transform(input_df)
    )
    
   
    
    

    # -------------------------
    # SHAP Explainer (defined once)
    
   
    # Predict button
    # -------------------------
    if st.button("Predict"):
        prediction = inference_pipeline.named_steps['model'].predict(X_transformed)[0]
        prob = inference_pipeline.named_steps['model'].predict_proba(X_transformed)[0][1]

        st.subheader(f"Prediction: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")
        st.write(f"Probability: {prob:.2f}")

    # -------------------------
    # Explain Prediction button
    # -------------------------
    
        
        

    

# Streamlit dashboard for the Pima Indian Dataset
# Sibusiso Mathebula
# 15/04/2025



import streamlit as st
import pandas as pd
from eda_functions import *
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import shap

import sklearn
print(sklearn.__version__)
        



def create_features(X):
            X = X.copy()
            X['PregnancyGroup'] = pd.cut(
                X['Pregnancies'],
                bins=[-1,0,3,6,20],
                labels=['0','1-3','4-6','7+']
            )
            X['AgeGroup'] = pd.cut(
                X['Age'],
                bins=[20,30,40,50,60,80],
                labels=['20-30','31-40','41-50','51-60','61-80']
            )
            X['Glucose_BMI'] = X['Glucose'] * X['BMI']
            return X

feature_creator = FunctionTransformer(create_features, validate=False)

st.set_page_config(
    page_title="Diabetes AI Dashboard",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA",'Outlier detection and error analysis', "Clustering", "Prediction"])

st.sidebar.markdown("## ⚙️ Controls")


st.title("Diabetes Prediction Dashboard")

df = load_data("diabetes (1).csv")
if page == "EDA":
    

    # ✅ ADD HERE
    total_cases = len(df)
    positive_cases = df["Outcome"].sum()
    negative_cases = total_cases - positive_cases
    positive_rate = positive_cases / total_cases

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", total_cases)
    col2.metric("Positive Cases", int(positive_cases))
    col3.metric("Negative Cases", int(negative_cases))
    col4.metric("Positive Rate", f"{positive_rate:.2%}")


    show_head(df)
    show_statistics(df)
    show_missing(df)

    with st.expander("See raw data"):
        st.dataframe(df)

    

    outcome_summary = df["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Count")
    outcome_summary["Outcome"] = outcome_summary["Outcome"].map({0: "Negative", 1: "Positive"})
    outcome_summary["Percentage"] = outcome_summary["Count"] / len(df)

    st.subheader("Outcome Summary Table")
    st.dataframe(outcome_summary)
    
    
    

    with st.expander("See raw data"):
        st.dataframe(df)

    st.subheader("Outcome Distribution")
    fig1 = plot_outcome_distribution(df)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2 = correlation_heatmap(df)
    st.pyplot(fig2)



    feature = st.selectbox("Select Feature for distribution plotting ", df.drop(['Outcome'],axis=1).columns)

    st.subheader(f"{feature}  ")
    fig3 = distribution(df[feature])
    st.pyplot(fig3)



    x_col = st.selectbox("Select X-axis",df.drop(['Outcome'],axis=1).columns)
    y_col = st.selectbox("Select Y-axis", df.drop(['Outcome'],axis=1).columns)

    feature = [x_col,y_col]

    fig = scatter_plot(df, x_col,y_col)
    st.pyplot(fig)
    



# --- Correlation Section ---
    st.markdown("### 📊 Correlation Analysis")

    corr_value = df[x_col].corr(df[y_col])
    st.write(f"Correlation: **{corr_value:.3f}**")

    # Interpretation
    if abs(corr_value) > 0.7:
        strength = "Strong"
    elif abs(corr_value) > 0.4:
        strength = "Moderate"
    elif abs(corr_value) > 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"

    direction = "positive" if corr_value > 0 else "negative"

    st.write(f" {strength} {direction} relationship")

    # Statistical significance
    from scipy.stats import pearsonr

    corr, p_value = pearsonr(df[x_col], df[y_col])
    st.write(f"P-value: {p_value:.5f}")

    if p_value < 0.05:
        st.success("Statistically significant")
    else:
        st.warning("Not statistically significant")

    fig = Pregnancy_dist(df)
    st.pyplot(fig)


    feature = st.selectbox("Select Feature for boxplot  ", df.drop(['Outcome'],axis=1).columns)
    fig = boxplot(df,feature)
    st.pyplot(fig)

    feature = st.selectbox("Select Feature for violinplot  ",df.drop(['Outcome'],axis=1).columns)
    fig = violinplot(df,feature)
    st.pyplot(fig)



    feature = st.selectbox("Select Feature for kdeplot  ", df.drop(['Outcome'],axis=1).columns)
    fig = kdeplot(df,feature)
    st.pyplot(fig)

    st.subheader("Diabetes positive rates for different age groups ")

    fig = AgeGroup(df)
    st.pyplot(fig)

    st.subheader("Diabetes positive rates for different BMI categories ")

    fig = BMI_Category(df)
    st.pyplot(fig)

    st.subheader("Diabetes positive rates for different Pregnancy groups ")

    fig = PregnancyGroup(df)
    st.pyplot(fig)

if page =='Outlier detection and error analysis':
     
    from scipy.stats import zscore

    st.title(" Outlier Detection & Error Analysis")

    # =========================
    # 1. Load Data
    # =========================
   

    st.subheader(" Dataset Overview")
    st.write(df.describe())

    # =========================
    # 2. Define Columns
    # =========================
    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # =========================
    # 3. ERROR ANALYSIS (Invalid Zeros)
    # =========================
    st.subheader("Data Quality (Invalid Zeros)")

    zero_errors = (df[medical_cols] == 0).sum()

    st.write("Count of invalid zero values (should be treated as missing):")
    st.dataframe(zero_errors)

    st.metric("Total Invalid Cells", zero_errors.sum())

    # Optional replacement toggle
    if st.checkbox("Replace zeros with NaN (recommended)"):
        df[medical_cols] = df[medical_cols].replace(0, np.nan)
        st.success("Zeros replaced with NaN")

    # =========================
    # 4. IQR OUTLIER DETECTION
    # =========================
    st.subheader(" IQR Outlier Detection")

    def iqr_outliers(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return data[(data[col] < lower) | (data[col] > upper)]

    iqr_outlier_indices = set()

    for col in medical_cols:
        out_idx = iqr_outliers(df, col).index
        iqr_outlier_indices.update(out_idx)

    iqr_outliers_df = df.loc[list(iqr_outlier_indices)]

    st.write(f"Total IQR outliers: {len(iqr_outliers_df)}")
    st.dataframe(iqr_outliers_df)

    # =========================
    # 5. Z-SCORE OUTLIER DETECTION
    # =========================
    st.subheader(" Z-Score Outlier Detection (|z| > 3)")

    z_df = df[medical_cols].dropna()
    z_scores = z_df.apply(zscore)

    z_outliers_mask = (z_scores.abs() > 3).any(axis=1)
    z_outliers_df = df.loc[z_df[z_outliers_mask].index]

    st.write(f"Total Z-score outliers: {len(z_outliers_df)}")
    st.dataframe(z_outliers_df)

    # =========================
    # 6. COMBINED ANALYSIS
    # =========================
    st.subheader(" Combined Anomaly Analysis")

    iqr_set = set(iqr_outliers_df.index)
    z_set = set(z_outliers_df.index)

    both = iqr_set & z_set
    iqr_only = iqr_set - z_set
    z_only = z_set - iqr_set

    st.write(f" High-confidence anomalies (both methods): {len(both)}")
    st.write(f"IQR-only anomalies: {len(iqr_only)}")
    st.write(f" Z-score-only anomalies: {len(z_only)}")

    combined_df = df.loc[list(both)]
    st.dataframe(combined_df)

    # =========================
    # 7. FILTER OPTION
    # =========================
    st.subheader("🧹 Filter Dataset")

    option = st.radio(
        "Select dataset view:",
        ["Original Data", "Remove IQR Outliers", "Remove All Anomalies"]
    )

    if option == "Original Data":
        st.dataframe(df)

    elif option == "Remove IQR Outliers":
        clean_df = df.drop(index=iqr_outliers_df.index)
        st.dataframe(clean_df)

    elif option == "Remove All Anomalies":
        all_outliers = iqr_set | z_set
        clean_df = df.drop(index=list(all_outliers))
        st.dataframe(clean_df)

    # =========================
    # 8. Summary Metrics
    # =========================
    st.subheader("📊 Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("IQR Outliers", len(iqr_outliers_df))
    col3.metric("Z-score Outliers", len(z_outliers_df))

# ======================
# CLUSTERING PAGE
# ======================
if page == "Clustering":
    st.title("🔍 DBSCAN Clustering")

    features = st.multiselect(
        "Select features for clustering",
        df.columns.drop('Outcome')
    )

    eps = st.slider("Select eps", 0.1, 5.0, 0.5)
    min_samples = st.slider("Select min_samples", 2, 20, 5)

    if st.button("Run DBSCAN"):
        clustered_df = run_dbscan(df, features, eps, min_samples)
        st.write(clustered_df.head())

        if len(features) == 2:
            plot_dbscan(clustered_df, features)
            
            

# ======================
# PREDICTION PAGE
# ======================
if page == "Prediction":
    st.title("🧠 Diabetes Prediction")

    import joblib
    from sklearn.pipeline import Pipeline
    import shap
    import matplotlib.pyplot as plt
    import joblib
    joblib.dump(model, "diabetes_model_only.pkl")
    # -------------------------
    # Load your trained pipeline
    # -------------------------
    pipeline = joblib.load("diabetes_model_only.pkl")

    # Create inference pipeline without SMOTE (already fitted)
    inference_pipeline = Pipeline([
        ('features', pipeline.named_steps['features']),
        ('preprocessing', pipeline.named_steps['preprocessing']),
        ('model', pipeline.named_steps['model'])
    ])

    # -------------------------
    # Collect user input
    # -------------------------
    input_data = {}
    for col in df.drop(columns=['Outcome']).columns:
        input_data[col] = st.number_input(
            f"Enter {col}",
            value=float(df[col].mean())
        )

    input_df = pd.DataFrame([input_data])

    # -------------------------
    # Transform input using fitted pipeline
    # -------------------------
    X_transformed = inference_pipeline.named_steps['preprocessing'].transform(
        inference_pipeline.named_steps['features'].transform(input_df)
    )
    
   
    
    

    # -------------------------
    # SHAP Explainer (defined once)
    
   
    # Predict button
    # -------------------------
    if st.button("Predict"):
        prediction = inference_pipeline.named_steps['model'].predict(X_transformed)[0]
        prob = inference_pipeline.named_steps['model'].predict_proba(X_transformed)[0][1]

        st.subheader(f"Prediction: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")
        st.write(f"Probability: {prob:.2f}")

    # -------------------------
    # Explain Prediction button
    # -------------------------
