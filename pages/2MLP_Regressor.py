#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Define the Streamlit app
def app():
    if "dataset_ready" not in st.session_state:
        st.error("Dataset must be loaded. Click Heart Disease in the sidebar.")
    st.subheader('Neural Network Regressor Performance on the Advertising Dataset')
    text = """The advertising dataset is commonly used for a multiple linear regression task.
    Here, the goal is to predict sales figures based on the amount spent on advertising 
    through different channels like TV, radio, and newspaper. The model learns the 
    relationship between these advertising expenses (independent variables) and the 
    resulting sales (dependent variable).
    \n**MLP Regressor**
    \nWhile linear regression is a good starting point, a more complex model  can be 
    used - a Multi-Layer Perceptron (MLP) regressor. This is a type of artificial 
    neural network that can capture non-linear relationships between the 
    advertising expenses and sales.
    \nThe model takes the advertising expenses (TV, radio, newspaper) as inputs.
    These inputs are passed through multiple hidden layers with interconnected nodes. 
    Each layer performs a linear transformation followed by a non-linear activation 
    function. These functions allow the model to learn complex patterns.
    Finally, the output layer produces a single value representing the predicted 
    sales figure. Compared to linear regression, MLP regressor can model more 
    intricate relationships between advertising and sales, potentially leading to 
    more accurate predictions."""
    st.write(text)
    
   # Define MLP parameters    
    st.sidebar.subheader('Set the MLP Parameters')
    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["lbfgs", "adam", "sgd"]
    solver = st.sidebar.selectbox('Select the solver:', options)

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=5,
        max_value=100,
        value=10,  # Initial value
        step=5
    )

    alpha = st.sidebar.slider(   
        label="Set the alpha:",
        min_value=.001,
        max_value=1.0,
        value=0.1,  # In1.0itial value
    )

    max_iter = st.sidebar.slider(   
        label="Set the max iterations:",
        min_value=100,
        max_value=500,
        value=300,  
        step=10
    )

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Create MLPRegressor model
    clf = MLPRegressor(solver=solver, activation=activation, 
        alpha=0.01, hidden_layer_sizes=(hidden_layers, 10), 
        random_state=1,max_iter=max_iter)

    text = """Recommended ANN parameters: solver=lbfgs, activation=relu, n_hidden_layer=10, max_iter=500"""
    st.write(text)

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the MLP regressor can take some time please wait...")

        # Train the model 
        clf.fit(X_train, y_train)

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Regressor training completed!") 

        st.subheader('Performance of the MLP-ANN Regressor on the Advertising Dataset')
        text = """We test the performance of the MLP Regressor using the 20% of the dataset that was
        set aside for testing. The regressor performance metrics are presented below."""
        st.write(text)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        
        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Performance test completed!") 
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print performance metrics
        st.subheader('Performance Metrics')
        st.write("Mean Squared Error: {:.4f}".format(mse))  
        st.write("R2 score: {:.4f}".format(r2))

        text = """Mean Squared Error (MSE) and R-squared (R²) are two commonly used metrics 
        to evaluate the performance of an Artificial Neural Network (ANN) regressor, 
        which predicts continuous values.
        \nMean Squared Error (MSE):
        \nMeasures the average squared difference between the actual values and the 
        predicted values by your ANN regressor.
        \nA lower MSE indicates a better fit, meaning the predictions are on average 
        closer to the actual values.
        \nMSE is sensitive to outliers, as large errors get squared and contribute more 
        significantly to the overall error.
        \nSince MSE is in squared units of the target variable, it can be difficult to 
        interpret directly.
        \nR-squared (R²):
        Represents the proportion of variance in the dependent variable (what you're trying 
        to predict) that's explained by your ANN regressor. R² ranges from 0 to 1, where 0 
        indicates no explanatory power and 1 indicates a perfect fit.
        R² is easier to interpret than MSE as it's on a 0-1 scale. However, it doesn't 
        tell you the magnitude of the errors.
        It's important to consider the number of features (independent variables) in your model. 
        R² can increase simply by adding more features, even if they're not relevant. 
        For this reason, a variation called Adjusted R² is often used as a penalty for 
        model complexity.
        \nTogether:
        MSE provides an absolute measure of the prediction error, while R² gives you a 
        relative idea of how well your model explains the variance.
        Ideally, you want a low MSE and a high R², but there can be trade-offs. 
        A more complex model might achieve a lower MSE but a higher R² due to overfitting."""
        st.write(text)

#run the app
if __name__ == "__main__":
    app()
