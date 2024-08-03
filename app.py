import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pyswarm import pso
from tqdm import tqdm
import time
import logging
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set a random seed for reproducibility
np.random.seed(42)

# Load the trained model pipeline
pipeline_path = 'model_pipeline.pkl'  # Ensure this file is in your GitHub repository

try:
    logger.info("Loading model pipeline from %s", pipeline_path)
    model_pipeline = joblib.load(pipeline_path)
    logger.info("Model pipeline loaded successfully")
except Exception as e:
    logger.error("Error loading model pipeline: %s", e)
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

# Define the input features and their ranges
numeric_features = [
    'Component_A', 'Component_B', 'Component_C', 'Component_D', 
    'Component_E', 'Component_F', 'Component_G'
]
factor_features = ['Factor_A', 'Factor_B', 'Factor_C']
categorical_features = ['Factor_D']
all_features = numeric_features + factor_features + categorical_features
factor_d_values = ['F1', 'F2', 'F3']

feature_ranges = {
    'Component_A': (5.0, 600.0),
    'Component_B': (0.0, 400.0),
    'Component_C': (5.0, 300.0),
    'Component_D': (150.0, 300.0),
    'Component_E': (0.0, 50.0),
    'Component_F': (700.0, 1400.0),
    'Component_G': (500.0, 1000.0),
    'Factor_A': (10.0, 40.0),
    'Factor_B': (30.0, 90.0),
    'Factor_C': (1.0, 260.0)
}

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
    (0, 2)  # Factor_D (index for categorical values F1, F2, F3)
]

# Define the bounds for PSO
lb = [bound[0] for bound in bounds]
ub = [bound[1] for bound in bounds]

def objective_function(x):
    x = np.round(x)  # Round to nearest integers
    input_data = dict(zip(all_features, x[:-1]))
    input_data['Factor_D'] = factor_d_values[int(round(x[-1]))]
    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)
    return -prediction[0]  # Negate because pso minimizes

def pso_with_progress(func, lb, ub, swarmsize=50, maxiter=100):
    progress_bar = tqdm(total=maxiter, desc="PSO Iterations")

    def wrapped_func(x):
        result = func(x)
        progress_bar.update(1)
        return result

    xopt, fopt = pso(wrapped_func, lb, ub, swarmsize=swarmsize, maxiter=maxiter)
    progress_bar.close()
    return xopt, fopt

def get_user_input():
    user_input = {}
    for feature in numeric_features + factor_features:
        min_val, max_val = feature_ranges[feature]
        value = st.number_input(f"Enter the value for {feature} (min: {min_val}, max: {max_val}):", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, format="%.2f")
        user_input[feature] = value

    for feature in categorical_features:
        value = st.selectbox(f"Enter the value for {feature}:", factor_d_values)
        user_input[feature] = value

    return user_input

def main_optimization():
    st.title('Optimization App using PSO')

    if st.button('Run Optimization'):
        # Run PSO with progress bar and increased parameters
        start_time = time.time()
        xopt, fopt = pso_with_progress(objective_function, lb, ub, swarmsize=50, maxiter=100)
        end_time = time.time()

        # Extract optimal input values
        optimal_input_dict = dict(zip(all_features, np.round(xopt[:-1])))
        optimal_input_dict['Factor_D'] = factor_d_values[int(round(xopt[-1]))]

        # Display the results
        st.write("Optimal Input Values:")
        for feature, value in optimal_input_dict.items():
            if feature in numeric_features:
                st.write(f"{feature}: {value:.1f}")
            else:
                st.write(f"{feature}: {value}")

        st.write(f"\nMaximized Prediction: {-fopt:.2f}")

        # Print time taken
        st.write(f"\nTime taken for optimization: {end_time - start_time:.2f} seconds")
    else:
        st.write("Click the button to run the optimization")

def main_prediction():
    st.title('Prediction App')

    user_input = get_user_input()
    input_df = pd.DataFrame([user_input])
    if st.button('Predict'):
        prediction = model_pipeline.predict(input_df)
        rounded_prediction = round(prediction[0], 1)
        st.write(f"Prediction: {rounded_prediction:.1f}")

# Define the cost per kg for each component
component_costs = {
    'Component_A': 0.072,
    'Component_B': 0.036,
    'Component_C': 0.024,
    'Component_D': 0.012,
    'Component_E': 0.36,
    'Component_F': 0.018,
    'Component_G': 0.03
}

# Define the cost function to minimize
def cost_function(x):
    try:
        return sum(x[i] * component_costs[numeric_features[i]] for i in range(len(numeric_features)))
    except KeyError as e:
        logger.error("KeyError in cost_function: %s", e)
        st.error(f"KeyError in cost_function: {e}")
        st.stop()

# Define the strength constraint function
def strength_constraint(x, desired_strength):
    input_data = dict(zip(numeric_features + factor_features, x[:-1]))
    input_data['Factor_D'] = factor_d_values[int(round(x[-1]))]
    input_df = pd.DataFrame([input_data])
    predicted_strength = model_pipeline.predict(input_df)[0]
    return predicted_strength

# Function to run the Monte Carlo simulation
def monte_carlo_simulation(desired_strength, num_simulations=1000):
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
        (0, 2)  # Factor_D (index for categorical values F1, F2, F3)
    ]

    results = []
    feature_samples = []

    for _ in range(num_simulations):
        random_sample = [np.random.uniform(bound[0], bound[1]) for bound in bounds]
        try:
            cost = cost_function(random_sample[:len(numeric_features)])  # Only pass numeric features to cost_function
        except KeyError as e:
            logger.error("KeyError in monte_carlo_simulation: %s", e)
            continue
        strength = strength_constraint(random_sample, desired_strength)
        results.append((cost, strength))
        feature_samples.append(random_sample)

    return results, feature_samples

# Main function to run Monte Carlo simulation for varying strengths and plot results
def main_monte_carlo():
    st.title('Monte Carlo Simulation for Cost Distribution')

    desired_strength_levels = np.arange(5, 85, 5)
    num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000)

    if st.button('Run Simulation'):
        all_results = []
        min_costs = []
        mean_costs = []

        for desired_strength in desired_strength_levels:
            results, feature_samples = monte_carlo_simulation(desired_strength, num_simulations)
            costs, strengths = zip(*results)

            # Filter results to only those that meet the desired strength
            valid_results = [(cost, sample) for cost, strength, sample in zip(costs, strengths, feature_samples) if strength >= desired_strength]
            if valid_results:
                valid_costs = [round(cost, 1) for cost, sample in valid_results]
                all_results.append((desired_strength, valid_costs))
                min_costs.append(min(valid_costs))
                mean_costs.append(round(np.mean(valid_costs), 1))
            else:
                min_costs.append(None)
                mean_costs.append(None)

        # Plot the histogram
        fig = go.Figure()
        for desired_strength, valid_costs in all_results:
            fig.add_trace(go.Histogram(x=valid_costs, name=f'Strength {desired_strength} MPa', opacity=0.5))

        fig.update_layout(
            barmode='overlay',
            title='Cost Distribution for Varying Strength Levels',
            xaxis_title='Cost (Currency)',
            yaxis_title='Frequency',
            legend_title='Desired Strength Levels',
            showlegend=True
        )
        st.plotly_chart(fig)

        # Plot the line chart
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=desired_strength_levels, y=min_costs, mode='lines+markers', name='Min Cost'))
        line_fig.add_trace(go.Scatter(x=desired_strength_levels, y=mean_costs, mode='lines+markers', name='Mean Cost'))

        line_fig.update_layout(
            title='Cost vs Desired Strength Level',
            xaxis_title='Desired Strength (MPa)',
            yaxis_title='Cost (Currency)',
            legend_title='Cost Type'
        )
        st.plotly_chart(line_fig)

def main_monte_carlo_varying():
    st.title('Monte Carlo Simulation for Varying Strength Levels')

    desired_strength_levels = np.arange(5, 85, 5)
    num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000)

    if st.button('Run Simulation'):
        all_results = []
        min_costs = []
        mean_costs = []

        for desired_strength in desired_strength_levels:
            results, feature_samples = monte_carlo_simulation(desired_strength, num_simulations)
            costs, strengths = zip(*results)

            # Filter results to only those that meet the desired strength
            valid_results = [(cost, sample) for cost, strength, sample in zip(costs, strengths, feature_samples) if strength >= desired_strength]
            if valid_results:
                valid_costs = [round(cost, 1) for cost, sample in valid_results]
                all_results.append((desired_strength, valid_costs))
                min_costs.append(min(valid_costs))
                mean_costs.append(round(np.mean(valid_costs), 1))
            else:
                min_costs.append(None)
                mean_costs.append(None)

        # Plot the histogram
        fig = go.Figure()
        for desired_strength, valid_costs in all_results:
            fig.add_trace(go.Histogram(x=valid_costs, name=f'Strength {desired_strength} MPa', opacity=0.5))

        fig.update_layout(
            barmode='overlay',
            title='Cost Distribution for Varying Strength Levels',
            xaxis_title='Cost (Currency)',
            yaxis_title='Frequency',
            legend_title='Desired Strength Levels',
            showlegend=True
        )
        st.plotly_chart(fig)

        # Plot the line chart
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=desired_strength_levels, y=min_costs, mode='lines+markers', name='Min Cost'))
        line_fig.add_trace(go.Scatter(x=desired_strength_levels, y=mean_costs, mode='lines+markers', name='Mean Cost'))

        line_fig.update_layout(
            title='Cost vs Desired Strength Level',
            xaxis_title='Desired Strength (MPa)',
            yaxis_title='Cost (Currency)',
            legend_title='Cost Type'
        )
        st.plotly_chart(line_fig)

# Page navigation
page = st.sidebar.selectbox("Select a Page", ["Optimization", "Prediction", "Monte Carlo Simulation for Cost Distribution", "Monte Carlo Simulation for Varying Strength Levels"])

if page == "Optimization":
    main_optimization()
elif page == "Prediction":
    main_prediction()
elif page == "Monte Carlo Simulation for Cost Distribution":
    main_monte_carlo()
elif page == "Monte Carlo Simulation for Varying Strength Levels":
    main_monte_carlo_varying()
