


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_outcome_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax)
    ax.set_title("Outcome Distribution")
    return fig


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return fig


def distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df, kde=True, ax=ax)
    ax.set_title(" Distribution plots")
    return fig 

def scatter_plot(df, x_col, y_col):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f'Scatter plot between {x_col} and {y_col}')
    return fig

def Pregnancy_dist(df):
    fig,ax = plt.subplots()
    sns.countplot(df,x='Pregnancies',hue='Outcome')
    ax.set_title('The counts for various number of pregnancies for each outcome')

    return fig

def boxplot(df,col):
    fig,ax = plt.subplots()
    sns.boxplot(data = df,x='Outcome',y=col)
    ax.set_title(f'The boxplot for {col} ')  
    return fig

def violinplot(df,col):
    
    fig,ax = plt.subplots()
    sns.violinplot(data = df,x='Outcome',y=col)
    ax.set_title(f'The violinplot for {col} ')

    return fig

def kdeplot(df,col):
    
    fig,ax = plt.subplots()
    sns.kdeplot(data = df,x=col, hue='Outcome',fill=True)
    ax.set_title(f'The KDE plot  for {col} ')

    return fig


def AgeGroup(df):
    fig,ax = plt.subplots()
    df['AgeGroup'] = pd.cut(df['Age'], bins=[20,30,40,50,60,80])
    sns.set_style("white")
    sns.barplot(x='AgeGroup', y='Outcome', data=df,errorbar=None)
    return fig

def PregnancyGroup(df):

    fig,ax = plt.subplots()
    df['PregnancyGroup'] = pd.cut(
    df['Pregnancies'],
    bins=[-1,0,3,6,20],
    labels=['0','1-3','4-6','7+'])

    sns.set_style("white")
    sns.barplot(x='PregnancyGroup', y='Outcome', data=df,errorbar=None)
    return fig

def BMI_Category(df):

    fig,ax = plt.subplots()
    df['BMI_Category'] = pd.cut(df['BMI'],
                           bins=[0,18.5,25,30,100],
                           labels=['Underweight','Normal','Overweight','Obese'])
    sns.set_style("white")
    sns.barplot(x='BMI_Category', y='Outcome', data=df,errorbar=None)
    return fig









import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ----------------------
# DATA LOADING FUNCTION
# ----------------------
def load_data(path):
    df = pd.read_csv(path)
    return df

# ----------------------
# BASIC EDA FUNCTIONS
# ----------------------
def show_head(df):
    st.write("### Dataset Preview")
    st.dataframe(df.head())
def show_missing(df):
    st.write("### Missing Values")
    st.write(df.isnull().sum())


def show_statistics(df):
    st.write("### Summary Statistics")
    st.write(df.describe())
    
def plot_histogram(df, column):
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna())
    ax.set_title(f"Histogram of {column}")
    st.pyplot(fig)

# ----------------------
# DBSCAN FUNCTION
# ----------------------
def run_dbscan(df, features, eps, min_samples):
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    df_result = df.copy()
    df_result['Cluster'] = clusters

    return df_result
def plot_dbscan(df, features):
    if len(features) != 2:
        st.warning("Select exactly 2 features for plotting")
        return

    fig, ax = plt.subplots()
    scatter = ax.scatter(df[features[0]], df[features[1]], c=df['Cluster'])
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("DBSCAN Clusters")
    st.pyplot(fig) 




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_outcome_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax)
    ax.set_title("Outcome Distribution")
    return fig


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return fig


def distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df, kde=True, ax=ax)
    ax.set_title(" Distribution plots")
    return fig 

def scatter_plot(df, x_col, y_col):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f'Scatter plot between {x_col} and {y_col}')
    return fig

def Pregnancy_dist(df):
    fig,ax = plt.subplots()
    sns.countplot(df,x='Pregnancies',hue='Outcome')
    ax.set_title('The counts for various number of pregnancies for each outcome')

    return fig

def boxplot(df,col):
    fig,ax = plt.subplots()
    sns.boxplot(data = df,x='Outcome',y=col)
    ax.set_title(f'The boxplot for {col} ')  
    return fig

def violinplot(df,col):
    
    fig,ax = plt.subplots()
    sns.violinplot(data = df,x='Outcome',y=col)
    ax.set_title(f'The violinplot for {col} ')

    return fig

def kdeplot(df,col):
    
    fig,ax = plt.subplots()
    sns.kdeplot(data = df,x=col, hue='Outcome',fill=True)
    ax.set_title(f'The KDE plot  for {col} ')

    return fig


def AgeGroup(df):
    fig,ax = plt.subplots()
    df['AgeGroup'] = pd.cut(df['Age'], bins=[20,30,40,50,60,80])
    sns.set_style("white")
    sns.barplot(x='AgeGroup', y='Outcome', data=df,errorbar=None)
    return fig

def PregnancyGroup(df):

    fig,ax = plt.subplots()
    df['PregnancyGroup'] = pd.cut(
    df['Pregnancies'],
    bins=[-1,0,3,6,20],
    labels=['0','1-3','4-6','7+'])

    sns.set_style("white")
    sns.barplot(x='PregnancyGroup', y='Outcome', data=df,errorbar=None)
    return fig

def BMI_Category(df):

    fig,ax = plt.subplots()
    df['BMI_Category'] = pd.cut(df['BMI'],
                           bins=[0,18.5,25,30,100],
                           labels=['Underweight','Normal','Overweight','Obese'])
    sns.set_style("white")
    sns.barplot(x='BMI_Category', y='Outcome', data=df,errorbar=None)
    return fig









import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ----------------------
# DATA LOADING FUNCTION
# ----------------------
def load_data(path):
    df = pd.read_csv(path)
    return df

# ----------------------
# BASIC EDA FUNCTIONS
# ----------------------
def show_head(df):
    st.write("### Dataset Preview")
    st.dataframe(df.head())
def show_missing(df):
    st.write("### Missing Values")
    st.write(df.isnull().sum())


def show_statistics(df):
    st.write("### Summary Statistics")
    st.write(df.describe())
    
def plot_histogram(df, column):
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna())
    ax.set_title(f"Histogram of {column}")
    st.pyplot(fig)

# ----------------------
# DBSCAN FUNCTION
# ----------------------
def run_dbscan(df, features, eps, min_samples):
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    df_result = df.copy()
    df_result['Cluster'] = clusters

    return df_result
def plot_dbscan(df, features):
    if len(features) != 2:
        st.warning("Select exactly 2 features for plotting")
        return

    fig, ax = plt.subplots()
    scatter = ax.scatter(df[features[0]], df[features[1]], c=df['Cluster'])
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("DBSCAN Clusters")
    st.pyplot(fig) 


