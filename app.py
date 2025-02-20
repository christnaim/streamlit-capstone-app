import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pyswarm import pso
import time
import logging
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set a random seed for reproducibility in PSO
seed = 42

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
    'Component_A': (100.0, 550.0),
    'Component_B': (0.0, 400.0),
    'Component_C': (0.0, 200.0),
    'Component_D': (120.0, 250.0),
    'Component_E': (0.0, 35.0),
    'Component_F': (800.0, 1200.0),
    'Component_G': (580.0, 1000.0),
    'Factor_A': (0.0, 70.0),
    'Factor_B': (0.0, 130.0),
    'Factor_C': (1.0, 365.0)
}

# Define the bounds for the numeric features
bounds = [
    (100, 550),  # Component_A
    (0, 400),  # Component_B
    (0, 200),  # Component_C
    (120, 250),  # Component_D
    (0, 35),  # Component_E
    (800, 1200),  # Component_F
    (580, 1000),  # Component_G
    (0, 70),  # Factor_A
    (0, 130),  # Factor_B
    (1, 365),  # Factor_C
]

# Define the bounds for PSO
lb = [bound[0] for bound in bounds]
ub = [bound[1] for bound in bounds]

def objective_function(x):
    x = np.round(x)  # Round to nearest integers
    input_data = dict(zip(numeric_features + factor_features, x))
    max_prediction = float('-inf')
    for value in factor_d_values:
        input_data['Factor_D'] = value
        input_df = pd.DataFrame([input_data])
        prediction = model_pipeline.predict(input_df)
        if prediction[0] > max_prediction:
            max_prediction = prediction[0]
    return -max_prediction  # Negate because pso minimizes

def pso_with_progress(func, lb, ub, swarmsize=50, maxiter=100):
    progress_bar = st.progress(0)

    def wrapped_func(x):
        result = func(x)
        return result

    # Set a fixed random seed before each PSO run for reproducibility
    np.random.seed(seed)  # Ensure reproducibility before PSO
    xopt, fopt = pso(wrapped_func, lb, ub, swarmsize=swarmsize, maxiter=maxiter, f_ieqcons=None, minfunc=1e-8, minstep=1e-8, debug=False)
    
    for i in range(maxiter):
        progress_bar.progress((i + 1) / maxiter)

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

def main_prediction():
    st.title('Prediction')

    user_input = get_user_input()
    input_df = pd.DataFrame([user_input])
    if st.button('Predict'):
        prediction = model_pipeline.predict(input_df)
        rounded_prediction = round(prediction[0], 1)
        st.write(f"Prediction: {rounded_prediction:.1f}")

def main_optimization():
    st.title('Strength Maximization')

    if st.button('Run Optimization'):
        # Run PSO with progress bar and increased parameters
        start_time = time.time()
        xopt, fopt = pso_with_progress(objective_function, lb, ub, swarmsize=50, maxiter=100)
        end_time = time.time()

        # Extract optimal input values
        optimal_input_dict = dict(zip(numeric_features + factor_features, np.round(xopt)))
        max_prediction = float('-inf')
        best_factor_d = None
        for value in factor_d_values:
            optimal_input_dict['Factor_D'] = value
            input_df = pd.DataFrame([optimal_input_dict])
            prediction = model_pipeline.predict(input_df)
            if prediction[0] > max_prediction:
                max_prediction = prediction[0]
                best_factor_d = value

        optimal_input_dict['Factor_D'] = best_factor_d

        # Display the results
        st.write("Optimal Input Values:")
        for feature, value in optimal_input_dict.items():
            if feature in numeric_features:
                st.write(f"{feature}: {value:.1f}")
            else:
                st.write(f"{feature}: {value}")

        st.write(f"\nMaximized Prediction: {max_prediction:.2f}")

        # Print time taken
        st.write(f"\nTime taken for optimization: {end_time - start_time:.2f} seconds")
    else:
        st.write("Click the button to run the optimization")

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
    input_data = dict(zip(numeric_features + factor_features, x))
    max_prediction = float('-inf')
    for value in factor_d_values:
        input_data['Factor_D'] = value
        input_df = pd.DataFrame([input_data])
        prediction = model_pipeline.predict(input_df)
        if prediction[0] > max_prediction:
            max_prediction = prediction[0]
    return max_prediction

# Function to run the Monte Carlo simulation
def monte_carlo_simulation(desired_strength, num_simulations=1000):
    results = []
    feature_samples = []

    for _ in range(num_simulations):
        rng = np.random.default_rng()  # Do not set the seed here to ensure different results
        random_sample = [rng.uniform(bound[0], bound[1]) for bound in bounds]
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
    st.title('Cost Minimization')

    desired_strength = st.number_input("Enter the desired strength:", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000)

    if st.button('Run Simulation'):
        results, feature_samples = monte_carlo_simulation(desired_strength, num_simulations)
        costs, strengths = zip(*results)

        # Filter results to only those that meet the desired strength
        valid_results = [(cost, sample) for cost, strength, sample in zip(costs, strengths, feature_samples) if strength >= desired_strength]
        if valid_results:
            valid_costs = [round(cost, 1) for cost, sample in valid_results]
            min_cost = min(valid_costs)
            mean_cost = round(np.mean(valid_costs), 1)

            st.write(f"Desired Strength: {desired_strength:.1f} MPa")
            st.write(f"Minimum Cost: {min_cost:.1f}")
            st.write(f"Mean Cost: {mean_cost:.1f}")

            # Extracting the feature values corresponding to the minimum cost
            min_cost_sample = min(valid_results, key=lambda x: x[0])[1]
            st.write("Feature values for minimum cost:")
            for feature, value in zip(all_features, min_cost_sample):
                st.write(f"{feature}: {value:.1f}")

            # Plot the results
            fig = px.histogram(valid_costs, nbins=50, title='Cost Distribution')
            st.plotly_chart(fig)
        else:
            st.write(f"No valid results for desired strength: {desired_strength:.1f} MPa")

def main_monte_carlo_varying():
    st.title('Monte Carlo Simulation for Varying Strength Levels')

    num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000)

    if st.button('Run Simulation'):
        desired_strength_levels = np.arange(10, 105, 10)
        all_results = []
        min_costs = []
        mean_costs = []
        min_cost_samples = []

        for desired_strength in desired_strength_levels:
            results, feature_samples = monte_carlo_simulation(desired_strength, num_simulations)
            costs, strengths = zip(*results)

            # Filter results to only those that meet the desired strength
            valid_results = [(cost, sample) for cost, strength, sample in zip(costs, strengths, feature_samples) if strength >= desired_strength]
            if valid_results:
                valid_costs = [round(cost, 1) for cost, sample in valid_results]
                if min_costs and min_costs[-1] is not None:
                    min_cost = max(min(valid_costs), min_costs[-1])  # Ensure non-decreasing min costs
                else:
                    min_cost = min(valid_costs)
                min_costs.append(min_cost)
                mean_costs.append(round(np.mean(valid_costs), 1))
                min_cost_sample = min(valid_results, key=lambda x: x[0])[1]
            else:
                if min_costs:
                    min_cost = min_costs[-1]
                else:
                    min_cost = None
                min_costs.append(min_cost)
                mean_costs.append(None)
                min_cost_sample = None

            min_cost_samples.append(min_cost_sample)
            all_results.append((desired_strength, valid_costs))

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

        # Display the table of minimum costs and corresponding feature values
        st.write("Minimum Costs and Corresponding Feature Values for Each Strength Level:")
        table_data = []
        for strength, min_cost, min_cost_sample in zip(desired_strength_levels, min_costs, min_cost_samples):
            if min_cost_sample:
                feature_values = {feature: round(value, 1) for feature, value in zip(all_features, min_cost_sample)}
            else:
                feature_values = {feature: None for feature in all_features}
            table_data.append({'Strength Level (MPa)': strength, 'Minimum Cost ($)': min_cost, 'Mean Cost ($)': mean_costs[desired_strength_levels.tolist().index(strength)], **feature_values})
        
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df)

# Page navigation
page = st.sidebar.selectbox("Select a Page", [
    "Prediction",
    "Strength Maximization",
    "Cost Minimization",
    "Monte Carlo Simulation for Varying Strength Levels"
])

if page == "Strength Maximization":
    main_optimization()
elif page == "Prediction":
    main_prediction()
elif page == "Cost Minimization":
    main_monte_carlo()
elif page == "Monte Carlo Simulation for Varying Strength Levels":
    main_monte_carlo_varying()
