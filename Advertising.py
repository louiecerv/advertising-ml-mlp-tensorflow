#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import time

# Define the Streamlit app
def app():


    if "X_train" not in st.session_state:
        st.session_state.X_train = []

    if "X_test" not in st.session_state:
            st.session_state.X_test = []

    if "y_train" not in st.session_state:
            st.session_state.y_train = []

    if "y_test" not in st.session_state:
            st.session_state.y_test = []


    if "dataset_ready" not in st.session_state:
        st.session_state.dataset_ready = False 

    text = """Three-way comparison of ML Classifiers, MLP and Tensorflow Artificial Neural Networks on the Advertising Dataset"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('advertising.jpg', caption="Advertising and Sales")

    text = """Regression Task for Sales Prediction with Advertising Data
    \nThis scenario involves a regression task where we aim to predict sales figures based
    on advertising spend data. A common dataset used for this purpose is the advertising 
    dataset available on Kaggle [1]. This dataset contains information on advertising
    budgets allocated to TV, Radio, and Newspaper media, along with the corresponding 
    sales revenue.
    \nGoal: Build a model to predict future sales based on the advertising budget 
    allocated across different media channels.
    \nData Preprocessing:
    Load the advertising dataset.
    Handle missing values (if any) through imputation techniques.
    Explore the data to understand relationships between advertising spend and sales.
    \nFeature Selection:
    We might consider including all three advertising budgets (TV, Radio, Newspaper) 
    as features initially. Feature engineering techniques like scaling the features 
    could be used to improve model performance.
    \nModel Building and Comparison:
    Linear Regression: This is a good starting point for regression tasks. It creates a 
    linear relationship between the advertising spend (features) and sales (target variable).
    \nMLP Classifier (Multi-Layer Perceptron): While MLPs are typically used for 
    classification tasks, they can be adapted for regression by using a linear 
    activation function in the output layer. This model can capture non-linear 
    relationships between features and the target variable.
    \nTensorFlow ANN Classifier: Similar to MLPs, TensorFlow ANNs are also capable of 
    regression tasks when configured appropriately. This offers a more flexible framework 
    for building complex neural network architectures for regression.
    \nModel Evaluation:
    Train-test split the data to evaluate the performance of each model.
    Use metrics like Mean Squared Error (MSE) or R-squared to compare the models and 
    identify the one that predicts sales most accurately.
    \nInterpretation:
    Analyze the coefficients of the linear regression model (if applicable) to 
    understand the impact of each advertising channel on sales.
    For MLP and TensorFlow ANN models, feature importance techniques can be used 
    to understand which advertising channels contribute most to the sales prediction.
    \nComparison of Classifiers:
    Linear Regression: Offers a simple and interpretable model, but might not capture 
    complex non-linear relationships between features and target variable.
    MLP/TensorFlow ANN: Can capture non-linearity but are generally more complex and 
    require careful hyperparameter tuning to avoid overfitting. Additionally, 
    interpreting these models can be challenging"""
    st.write(text)
    # Load the data dataset
    df = pd.read_csv('advertising.csv', header=0)

    with st.expander('Click to browse the dataset'):
        st.write(df)

    st.subheader('Descriptive Statistics')
    st.write(df.describe(include='all').T)

    # Separate features and target variable
    X = df.drop('Sales', axis=1)  # Target variable column name
    y = df['Sales']

    # Preprocess the data (e.g., scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # store for later use
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.session_state.dataset_ready = True

    with st.expander('Click to show the graphs'):
        bins = [10, 30, 50, 70, 90]

        plot_feature(df["TV"], df["Sales"], 'TV', 'Sales', 'TV VS Sales')
        plot_feature(df["Radio"], df["Sales"], 'Radio', 'Sale', 'Radio VS Sales')
        plot_feature(df["Newspaper"], df["Sales"], 'Newspaper', 'Sale', 'Newspaper VS Sales')

def plot_feature(feature, target, labelx, labely, title):
    # Display the plots
    fig, ax = plt.subplots(figsize=(10, 6))
    # Scatter plot
    ax.scatter(feature, target)
    # Add labels and title
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    ax.set_title(title)
    # Add grid
    ax.grid(True)
    st.pyplot(fig)

#run the app
if __name__ == "__main__":
    app()
