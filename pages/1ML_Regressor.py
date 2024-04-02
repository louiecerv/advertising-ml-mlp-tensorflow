#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Define the Streamlit app
def app():
    if "dataset_ready" not in st.session_state:
        st.error("Dataset must be loaded. Click Heart Disease in the sidebar.")
        
    st.subheader("Linear Regression Analysis Using Machine Learning Regressors")
    text = """Linear regression is a supervised learning technique for continuous 
    prediction. It aims to model the relationship between independent variables 
    (features) and a dependent variable (target) using a linear equation.
    \nModel Creation and Training:
    \nInstantiate a `LinearRegression` object from scikit-learn.
    - Fit the model to the training data using `model.fit(X_train, y_train)`, where `
    X_train` is the training features and `y_train` is the training target variable.
    \nPrediction on the Testing Set
    \nUse the trained model to make predictions on the testing data using `model.predict(X_test)`.
    \nModel Evaluation:
    Calculate performance metrics: Common metrics for regression tasks include mean squared 
    error (MSE), R-squared, and adjusted R-squared. Scikit-learn provides functions like 
    `mean_squared_error` and `r2_score` to compute these metrics.
    Interpret the results: Analyze the performance metrics to assess how well the model 
    generalizes to unseen data.
    Linear regression assumes a linear relationship between features and the target variable. 
    If the relationship is non-linear, consider using other regression techniques like polynomial regression or decision tree regression.
    Regularization techniques like L1 (LASSO) or L2 (Ridge) can be helpful to prevent overfitting, especially with high-dimensional datasets."""
    st.write(text)

    #add the classifier selection to the sidebar
    clf = KNeighborsRegressor(n_neighbors=5)
    options = ['K Nearest Neighbor', 'Support Vector Machine', 'Decision Tree']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option =='Support Vector Machine':
        clf = SVR()
        st.session_state['selected_model'] = 1
    elif selected_option=='Decision Tree':        
        clf = DecisionTreeRegressor()
        st.session_state['selected_model'] = 2
    else:
        clf = KNeighborsRegressor(n_neighbors=5)
        st.session_state['selected_model'] = 0

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    if st.button("Begin Training"):

        if selected_option =='K Nearest Neighbor':
            text = """KNN achieves good accuracy on the heart disease dataset, often 
            reaching around 85-90%. However, it can be slow for large datasets 
            due to needing to compare each test image to all training images. 
            Additionally, choosing the optimal number of neighbors (k) can be 
            crucial for performance."""
            classifier = 'K-Nearest Neighbor'
        elif st.session_state['selected_model'] == 1:   # SVM
            text = """SVM can also achieve high accuracy on this dataset, 
            similar to KNN. It offers advantages like being memory-efficient, 
            but choosing the right kernel function and its parameters 
            can be challenging."""
            classifier = 'Support Vector Machine'
        elif selected_option=='Decision Tree': 
            text = """Naive Bayes is generally faster than the other two options but 
            may achieve slightly lower accuracy, typically around 80-85%. It performs 
            well when the features are independent, which might not perfectly hold true 
            for data found in the heart disease dataset."""
            classifier = "Naive Bayes"

        st.subheader('Performance of ' + classifier)
        st.write(text)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.subheader('Performance Metrics')
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print the performance metrics
        st.write("Mean Squared Error:", mse)
        st.write("R-squared:", r2)

#run the app
if __name__ == "__main__":
    app()