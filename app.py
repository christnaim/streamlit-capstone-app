import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pyswarm import pso
from tqdm import tqdm
import time

# Set a random seed for reproducibility
np.random.seed(42)

# Load the trained model pipeline
pipeline_path = 'model_pipeline.pkl'  # Ensure this file is in your GitHub repository
model_pipeline = joblib.load(pipeline_path)

# Define the input features and their ranges
numeric_features = [
    'Component_A', 'Component_B', 'Component_C', 'Component_D', 
    'Component_E', 'Component_F', 'Component_G', 'Factor_A', 
    'Factor_B', 'Factor_C'
]
categorical_features = ['Factor_D']
factor_d_values = ['F1', 'F2', 'F3']

# Define the bounds for the numeric features
bounds = [
    (5, 600),  # Component_A
    (0, 400),  # Component_B
    (5, 300),  # Component_C
    (150, 300),  # Component_D
    (0, 50),  # Component_E
    (700, 1400),  # Component_F
    (500, 1000),  # Component_G
    (10, 40),  # Factor_A
    (30, 90),  # Factor_B
    (1, 260),  # Factor_C
    (0, 2)    # Factor_D (index for categorical values F1, F2, F3)
]

# Define the objective function to maximize
def objective_function(x):
    x = np.round(x)  # Round to nearest integers
    input_data = dict(zip(numeric_features, x[:-1]))
    input_data['Factor_D'] = factor_d_values[int(round(x[-1]))]
    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)
    return -prediction[0]  # Negate because pso minimizes

# Define the bounds for PSO
lb = [bound[0] for bound in bounds]
ub = [bound[1] for bound in bounds]

# Define a wrapper function to add progress bar to PSO
def pso_with_progress(func, lb, ub, swarmsize=50, maxiter=100):
    progress_bar = tqdm(total=maxiter, desc="PSO Iterations")

    def wrapped_func(x):
        result = func(x)
        progress_bar.update(1)
        return result

    xopt, fopt = pso(wrapped_func, lb, ub, swarmsize=swarmsize, maxiter=maxiter)
    progress_bar.close()
    return xopt, fopt

# Streamlit App
st.title('Optimization App using PSO')

if st.button('Run Optimization'):
    # Run PSO with progress bar and increased parameters
    start_time = time.time()
    xopt, fopt = pso_with_progress(objective_function, lb, ub, swarmsize=50, maxiter=100)
    end_time = time.time()

    # Extract optimal input values
    optimal_input_dict = dict(zip(numeric_features, np.round(xopt[:-1])))
    optimal_input_dict['Factor_D'] = factor_d_values[int(round(xopt[-1]))]

    # Display the results
    st.write("Optimal Input Values:")
    for feature, value in optimal_input_dict.items():
        if feature in numeric_features:
            st.write(f"{feature}: {value:.2f}")
        else:
            st.write(f"{feature}: {value}")

    st.write(f"\nMaximized Prediction: {-fopt:.2f}")

    # Print time taken
    st.write(f"\nTime taken for optimization: {end_time - start_time:.2f} seconds")
else:
    st.write("Click the button to run the optimization")
