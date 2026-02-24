import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# App config + logging
# -----------------------------
st.set_page_config(page_title="Prediction & Optimization App", layout="wide")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed seed for reproducibility (PSO)
SEED = 42

# -----------------------------
# Optional dependency: pyswarm
# (handled safely so missing package doesn't crash the app at startup)
# -----------------------------
PYSWARM_AVAILABLE = True
PYSWARM_IMPORT_ERROR = None
try:
    from pyswarm import pso
except Exception as e:
    PYSWARM_AVAILABLE = False
    PYSWARM_IMPORT_ERROR = e
    logger.error("pyswarm import failed: %s", e)

# -----------------------------
# Features + ranges
# -----------------------------
numeric_features = [
    "Component_A",
    "Component_B",
    "Component_C",
    "Component_D",
    "Component_E",
    "Component_F",
    "Component_G",
]
factor_features = ["Factor_A", "Factor_B", "Factor_C"]
categorical_features = ["Factor_D"]
all_features = numeric_features + factor_features + categorical_features

factor_d_values = ["F1", "F2", "F3"]

feature_ranges = {
    "Component_A": (100.0, 550.0),
    "Component_B": (0.0, 400.0),
    "Component_C": (0.0, 200.0),
    "Component_D": (120.0, 250.0),
    "Component_E": (0.0, 35.0),
    "Component_F": (800.0, 1200.0),
    "Component_G": (580.0, 1000.0),
    "Factor_A": (0.0, 70.0),
    "Factor_B": (0.0, 130.0),
    "Factor_C": (1.0, 365.0),
}

# Bounds for numeric + factor (10 variables total; Factor_D is categorical and tested separately)
bounds = [
    (100, 550),   # Component_A
    (0, 400),     # Component_B
    (0, 200),     # Component_C
    (120, 250),   # Component_D
    (0, 35),      # Component_E
    (800, 1200),  # Component_F
    (580, 1000),  # Component_G
    (0, 70),      # Factor_A
    (0, 130),     # Factor_B
    (1, 365),     # Factor_C
]

lb = [b[0] for b in bounds]
ub = [b[1] for b in bounds]

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model_pipeline():
    pipeline_path = Path(__file__).parent / "model_pipeline.pkl"
    logger.info("Loading model pipeline from %s", pipeline_path)

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model file not found: {pipeline_path}")

    model = joblib.load(pipeline_path)
    logger.info("Model pipeline loaded successfully")
    return model

try:
    model_pipeline = load_model_pipeline()
except Exception as e:
    logger.error("Error loading model pipeline: %s", e)
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def build_input_df_from_vector(x_values, factor_d_value):
    """
    x_values: length 10 vector corresponding to numeric_features + factor_features
    factor_d_value: one of F1/F2/F3
    """
    input_data = dict(zip(numeric_features + factor_features, x_values))
    input_data["Factor_D"] = factor_d_value
    return pd.DataFrame([input_data], columns=all_features)

def predict_best_strength_for_vector(x_values, round_inputs=False):
    """
    Returns:
      (max_prediction, best_factor_d)
    """
    x = np.array(x_values, dtype=float)
    if round_inputs:
        x = np.round(x)

    max_prediction = float("-inf")
    best_factor_d = None

    for value in factor_d_values:
        input_df = build_input_df_from_vector(x, value)
        prediction = model_pipeline.predict(input_df)
        pred_val = float(prediction[0])

        if pred_val > max_prediction:
            max_prediction = pred_val
            best_factor_d = value

    return max_prediction, best_factor_d

# -----------------------------
# Optimization (PSO)
# -----------------------------
def objective_function(x):
    # Round to nearest integers for optimization evaluation
    max_prediction, _ = predict_best_strength_for_vector(x, round_inputs=True)
    return -float(max_prediction)  # PSO minimizes

def pso_with_progress(func, lb_vals, ub_vals, swarmsize=50, maxiter=100):
    # pyswarm has no easy iteration callback; use spinner instead of fake progress
    with st.spinner("Running optimization..."):
        np.random.seed(SEED)
        xopt, fopt = pso(
            func,
            lb_vals,
            ub_vals,
            swarmsize=swarmsize,
            maxiter=maxiter,
            f_ieqcons=None,
            minfunc=1e-8,
            minstep=1e-8,
            debug=False,
        )
    return xopt, fopt

def get_user_input():
    user_input = {}

    for feature in numeric_features + factor_features:
        min_val, max_val = feature_ranges[feature]
        value = st.number_input(
            f"Enter the value for {feature} (min: {min_val}, max: {max_val})",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            format="%.2f",
        )
        user_input[feature] = value

    for feature in categorical_features:
        value = st.selectbox(f"Enter the value for {feature}", factor_d_values)
        user_input[feature] = value

    return user_input

def main_prediction():
    st.title("Prediction")

    user_input = get_user_input()
    input_df = pd.DataFrame([user_input], columns=all_features)

    if st.button("Predict"):
        try:
            prediction = model_pipeline.predict(input_df)
            rounded_prediction = round(float(prediction[0]), 1)
            st.write(f"Prediction: {rounded_prediction:.1f}")
        except Exception as e:
            logger.error("Prediction error: %s", e)
            st.error(f"Prediction error: {e}")

def main_optimization():
    st.title("Strength Maximization")

    if not PYSWARM_AVAILABLE:
        st.error(
            "PSO optimization is unavailable because the 'pyswarm' package is not installed.\n\n"
            f"Import error: {PYSWARM_IMPORT_ERROR}\n\n"
            "Add 'pyswarm' to your requirements.txt and redeploy."
        )
        return

    if st.button("Run Optimization"):
        try:
            start_time = time.time()
            xopt, _ = pso_with_progress(objective_function, lb, ub, swarmsize=50, maxiter=100)
            end_time = time.time()

            # Round optimized values for display and evaluation
            xopt_rounded = np.round(xopt)
            max_prediction, best_factor_d = predict_best_strength_for_vector(xopt_rounded, round_inputs=False)

            optimal_input_dict = dict(zip(numeric_features + factor_features, xopt_rounded))
            optimal_input_dict["Factor_D"] = best_factor_d

            st.write("Optimal Input Values:")
            for feature, value in optimal_input_dict.items():
                if feature in numeric_features + factor_features:
                    st.write(f"{feature}: {float(value):.1f}")
                else:
                    st.write(f"{feature}: {value}")

            st.write(f"Maximized Prediction: {float(max_prediction):.2f}")
            st.write(f"Time taken for optimization: {end_time - start_time:.2f} seconds")

        except Exception as e:
            logger.error("Optimization error: %s", e)
            st.error(f"Optimization error: {e}")
    else:
        st.write("Click the button to run the optimization.")

# -----------------------------
# Cost minimization (Monte Carlo)
# -----------------------------
component_costs = {
    "Component_A": 0.072,
    "Component_B": 0.036,
    "Component_C": 0.024,
    "Component_D": 0.012,
    "Component_E": 0.36,
    "Component_F": 0.018,
    "Component_G": 0.03,
}

def cost_function(x_numeric):
    """
    x_numeric should contain only the 7 numeric component values.
    """
    try:
        return float(
            sum(float(x_numeric[i]) * component_costs[numeric_features[i]] for i in range(len(numeric_features)))
        )
    except KeyError as e:
        logger.error("KeyError in cost_function: %s", e)
        raise

def strength_constraint(x_all):
    """
    x_all should contain 10 values (numeric + factor features).
    Returns (strength, best_factor_d)
    """
    strength, best_factor_d = predict_best_strength_for_vector(x_all, round_inputs=False)
    return float(strength), best_factor_d

def monte_carlo_simulation(desired_strength, num_simulations=1000):
    """
    Returns:
      results: list of tuples (cost, strength)
      feature_samples: list of 10-value vectors
      best_factor_d_samples: list of best Factor_D values for each sample
    """
    results = []
    feature_samples = []
    best_factor_d_samples = []

    rng = np.random.default_rng()  # single RNG for the full run

    for _ in range(num_simulations):
        random_sample = [rng.uniform(bound[0], bound[1]) for bound in bounds]

        try:
            cost = cost_function(random_sample[: len(numeric_features)])
        except KeyError as e:
            logger.error("KeyError in monte_carlo_simulation: %s", e)
            continue

        strength, best_factor_d = strength_constraint(random_sample)

        results.append((cost, strength))
        feature_samples.append(random_sample)
        best_factor_d_samples.append(best_factor_d)

    return results, feature_samples, best_factor_d_samples

def main_monte_carlo():
    st.title("Cost Minimization")

    desired_strength = st.number_input(
        "Enter the desired strength:",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
    )
    num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000)

    if st.button("Run Simulation"):
        try:
            results, feature_samples, best_factor_d_samples = monte_carlo_simulation(desired_strength, num_simulations)

            if not results:
                st.warning("No simulation results were generated.")
                return

            costs, strengths = zip(*results)

            # Keep only samples meeting strength target
            valid_results = [
                (cost, sample, best_fd)
                for cost, strength, sample, best_fd in zip(costs, strengths, feature_samples, best_factor_d_samples)
                if strength >= desired_strength
            ]

            if valid_results:
                valid_costs = [round(cost, 1) for cost, _, _ in valid_results]
                min_cost = min(valid_costs)
                mean_cost = round(float(np.mean(valid_costs)), 1)

                st.write(f"Desired Strength: {desired_strength:.1f} MPa")
                st.write(f"Minimum Cost: {min_cost:.1f}")
                st.write(f"Mean Cost: {mean_cost:.1f}")

                # Feature values for min-cost valid result
                min_cost_result = min(valid_results, key=lambda x: x[0])
                min_cost_sample = min_cost_result[1]
                min_cost_factor_d = min_cost_result[2]

                st.write("Feature values for minimum cost:")
                for feature, value in zip(numeric_features + factor_features, min_cost_sample):
                    st.write(f"{feature}: {float(value):.1f}")
                st.write(f"Factor_D: {min_cost_factor_d}")

                # Plot histogram
                fig = px.histogram(valid_costs, nbins=50, title="Cost Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"No valid results for desired strength: {desired_strength:.1f} MPa")

        except Exception as e:
            logger.error("Monte Carlo error: %s", e)
            st.error(f"Monte Carlo error: {e}")

def main_monte_carlo_varying():
    st.title("Monte Carlo Simulation for Varying Strength Levels")

    num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000)

    if st.button("Run Simulation"):
        try:
            desired_strength_levels = np.arange(10, 105, 10)

            all_results = []            # list of (desired_strength, valid_costs)
            min_costs = []
            mean_costs = []
            min_cost_samples = []
            min_cost_factor_ds = []

            for desired_strength in desired_strength_levels:
                results, feature_samples, best_factor_d_samples = monte_carlo_simulation(desired_strength, num_simulations)

                if not results:
                    valid_costs = []
                    min_cost = min_costs[-1] if min_costs else None
                    min_costs.append(min_cost)
                    mean_costs.append(None)
                    min_cost_samples.append(None)
                    min_cost_factor_ds.append(None)
                    all_results.append((desired_strength, valid_costs))
                    continue

                costs, strengths = zip(*results)

                # IMPORTANT FIX: initialize valid_costs each loop
                valid_costs = []

                valid_results = [
                    (cost, sample, best_fd)
                    for cost, strength, sample, best_fd in zip(costs, strengths, feature_samples, best_factor_d_samples)
                    if strength >= desired_strength
                ]

                if valid_results:
                    valid_costs = [round(cost, 1) for cost, _, _ in valid_results]

                    # Keep min_cost non-decreasing with higher strength targets
                    if min_costs and min_costs[-1] is not None:
                        min_cost = max(min(valid_costs), min_costs[-1])
                    else:
                        min_cost = min(valid_costs)

                    mean_cost = round(float(np.mean(valid_costs)), 1)
                    min_cost_result = min(valid_results, key=lambda x: x[0])
                    min_cost_sample = min_cost_result[1]
                    min_cost_factor_d = min_cost_result[2]
                else:
                    min_cost = min_costs[-1] if min_costs else None
                    mean_cost = None
                    min_cost_sample = None
                    min_cost_factor_d = None

                min_costs.append(min_cost)
                mean_costs.append(mean_cost)
                min_cost_samples.append(min_cost_sample)
                min_cost_factor_ds.append(min_cost_factor_d)
                all_results.append((desired_strength, valid_costs))

            # Histogram overlay
            fig = go.Figure()
            for desired_strength, valid_costs in all_results:
                if len(valid_costs) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=valid_costs,
                            name=f"Strength {desired_strength} MPa",
                            opacity=0.5,
                        )
                    )

            fig.update_layout(
                barmode="overlay",
                title="Cost Distribution for Varying Strength Levels",
                xaxis_title="Cost (Currency)",
                yaxis_title="Frequency",
                legend_title="Desired Strength Levels",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Line chart: min & mean cost vs strength
            line_fig = go.Figure()
            line_fig.add_trace(
                go.Scatter(
                    x=desired_strength_levels,
                    y=min_costs,
                    mode="lines+markers",
                    name="Min Cost",
                )
            )
            line_fig.add_trace(
                go.Scatter(
                    x=desired_strength_levels,
                    y=mean_costs,
                    mode="lines+markers",
                    name="Mean Cost",
                )
            )

            line_fig.update_layout(
                title="Cost vs Desired Strength Level",
                xaxis_title="Desired Strength (MPa)",
                yaxis_title="Cost (Currency)",
                legend_title="Cost Type",
            )
            st.plotly_chart(line_fig, use_container_width=True)

            # Table of min costs + feature values
            st.write("Minimum Costs and Corresponding Feature Values for Each Strength Level:")

            table_data = []
            for strength, min_cost, mean_cost, min_cost_sample, min_factor_d in zip(
                desired_strength_levels, min_costs, mean_costs, min_cost_samples, min_cost_factor_ds
            ):
                if min_cost_sample is not None:
                    feature_values = {
                        feature: round(float(value), 1)
                        for feature, value in zip(numeric_features + factor_features, min_cost_sample)
                    }
                    feature_values["Factor_D"] = min_factor_d
                else:
                    feature_values = {feature: None for feature in numeric_features + factor_features}
                    feature_values["Factor_D"] = None

                row = {
                    "Strength Level (MPa)": int(strength),
                    "Minimum Cost ($)": min_cost,
                    "Mean Cost ($)": mean_cost,
                    **feature_values,
                }
                table_data.append(row)

            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, use_container_width=True)

        except Exception as e:
            logger.error("Varying Monte Carlo error: %s", e)
            st.error(f"Varying Monte Carlo error: {e}")

# -----------------------------
# Sidebar navigation
# -----------------------------
page = st.sidebar.selectbox(
    "Select a Page",
    [
        "Prediction",
        "Strength Maximization",
        "Cost Minimization",
        "Monte Carlo Simulation for Varying Strength Levels",
    ],
)

if page == "Strength Maximization":
    main_optimization()
elif page == "Prediction":
    main_prediction()
elif page == "Cost Minimization":
    main_monte_carlo()
elif page == "Monte Carlo Simulation for Varying Strength Levels":
    main_monte_carlo_varying()
