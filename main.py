import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import StringIO

# Function to fit a polynomial to the data points and evaluate it
def fit_and_evaluate_polynomial(x_data, y_data, degree, x_eval):
    coefficients = np.polyfit(x_data, y_data, degree)
    polynomial = np.poly1d(coefficients)
    return polynomial(x_eval), polynomial, coefficients

# Calculate the rope length (now part of total cord length)
def calculate_rope_length(goal_height, stretched_elastic_length, bottle_length, etc_length):
    return goal_height - stretched_elastic_length - bottle_length - etc_length

# Check the elasticity test
def check_elasticity_test(polynomial, unstretched_length):
    stretched_length_500g = polynomial(500)
    required_stretched_length = 125
    scaled_stretched_length = (100 / unstretched_length) * stretched_length_500g if unstretched_length > 0 else 0
    return scaled_stretched_length >= required_stretched_length, scaled_stretched_length, stretched_length_500g

# Streamlit app layout
st.title("Science Olympiad Bungee Drop Length Calculator (Regional)")

# Section to input data points
st.subheader("Input Data Points")
st.write("Enter or upload the drop mass (grams) and corresponding stretched elastic length (cm). Mass can include decimals for accuracy.")

# Default data points
default_data = {
    "Drop Mass (grams)": [100.0, 200.0, 300.0, 400.0, 500.0],
    "Stretched Elastic Length (cm)": [27.0, 36.0, 43.0, 49.0, 53.0]
}
data_df = pd.DataFrame(default_data)

# Option to upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data_df = pd.read_csv(uploaded_file)
    if not {"Drop Mass (grams)", "Stretched Elastic Length (cm)"}.issubset(data_df.columns):
        st.error("CSV must contain 'Drop Mass (grams)' and 'Stretched Elastic Length (cm)' columns.")
        data_df = pd.DataFrame(default_data)

# Allow user to edit the data points with decimal support for mass
edited_df = st.data_editor(
    data_df,
    num_rows="dynamic",
    column_config={
        "Drop Mass (grams)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.1f"),
        "Stretched Elastic Length (cm)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.1f")
    },
    key="data_editor"
)

# Option to download data as CSV
csv = edited_df.to_csv(index=False)
st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name="bungee_data.csv",
    mime="text/csv"
)

# Extract the data points
x_data = edited_df["Drop Mass (grams)"].values
y_data = edited_df["Stretched Elastic Length (cm)"].values

# Ensure enough data points
min_points = 2
if len(x_data) < min_points or len(y_data) < min_points:
    st.error(f"Please provide at least {min_points} data points to fit a polynomial.")
else:
    max_degree = min(len(x_data) - 1, 4)
    degree = st.slider("Polynomial Degree", min_value=1, max_value=max_degree, value=2, step=1)  # Changed default to 2 (quadratic)

    st.subheader("Calculation Parameters")
    drop_mass = st.number_input("Drop Mass (grams)", min_value=50.0, max_value=300.0, value=100.0, step=0.1, format="%.1f")
    goal_height = st.slider("Goal Height (cm)", min_value=200, max_value=500, value=300, step=10)
    bottle_length = st.number_input("Bottle Length (cm)", value=22.0, min_value=0.0, step=1.0)
    unstretched_length = st.number_input("Unstretched Elastic Length (cm)", value=20.0, min_value=0.1, step=1.0)
    etc_length = st.number_input("Etc Length (cm)", value=5.0, min_value=0.0, step=0.1, help="Length of additional components like clips, zip ties, etc.")

    try:
        stretched_elastic_length, polynomial, coefficients = fit_and_evaluate_polynomial(x_data, y_data, degree, drop_mass)
        stretch_ratio = stretched_elastic_length / unstretched_length if unstretched_length > 0 else 0
        rope_length = calculate_rope_length(goal_height, stretched_elastic_length, bottle_length, etc_length)
        passes_elasticity_test, scaled_stretched_length, stretched_length_500g = check_elasticity_test(polynomial, unstretched_length)

        # Check for negative stretched length at 500 g
        if stretched_length_500g < 0:
            st.warning("The fitted polynomial predicts a negative stretched length at 500 g, which is not physically realistic. Try lowering the polynomial degree or adjusting the data points to ensure positive stretched lengths.")

        st.subheader("Results")
        st.write(f"**Drop Mass:** {drop_mass:.1f} grams")
        st.write(f"**Goal Height:** {goal_height} cm")
        st.write(f"**Unstretched Elastic Length:** {unstretched_length:.2f} cm")
        st.write(f"**Stretched Elastic Length:** {stretched_elastic_length:.2f} cm")
        st.write(f"**Stretch Ratio:** {stretch_ratio:.2f}")
        st.write(f"**Etc Length (clips, zip ties, etc.):** {etc_length:.2f} cm")
        st.write(f"**Rope Length:** {rope_length:.2f} cm")
        if rope_length < 0:
            st.warning("Rope length is negative. Adjust parameters (e.g., increase Goal Height or reduce other lengths).")

        st.subheader("Elasticity Test")
        st.write("The bottom 1 meter of the cord must stretch to at least 125 cm with 500 g.")
        st.write(f"**Stretched Length at 500 g (for {unstretched_length:.2f} cm):** {stretched_length_500g:.2f} cm")
        st.write(f"**Scaled Stretched Length for 500 g (from {unstretched_length:.2f} cm to 100 cm):** {scaled_stretched_length:.2f} cm")
        if stretched_length_500g < 0:
            st.write("❌ Fails (negative stretched length is invalid)")
        else:
            st.write("✅ Passes" if passes_elasticity_test else "❌ Fails")

        st.subheader("Stretched Elastic Length vs. Drop Mass")
        x_values = np.linspace(min(x_data), max(x_data), 100)
        y_values = polynomial(x_values)
        fig = go.Figure(data=[
            go.Scatter(x=x_data, y=y_data, mode='markers', name='Data', marker=dict(color='green', size=10)),
            go.Scatter(x=x_values, y=y_values, mode='lines', name='Fit', line=dict(color='black')),
            go.Scatter(x=[drop_mass], y=[stretched_elastic_length], mode='markers', name=f'Selected ({drop_mass:.1f}g, {stretched_elastic_length:.2f}cm)', marker=dict(color='red', size=12, symbol='x'))
        ])
        fig.update_layout(xaxis_title="Drop Mass (grams)", yaxis_title="Stretched Elastic Length (cm)", width=800, height=400)
        st.plotly_chart(fig)

        st.subheader("Components")
        if rope_length >= 0:
            components = ["Rope", "Stretched Elastic", "Etc", "Bottle"]
            lengths = [rope_length, stretched_elastic_length, etc_length, bottle_length]
            colors = ["#FF9999", "#66B2FF", "#FFD700", "#99FF99"]  # Added yellow for Etc
            bar_fig = go.Figure(data=[
                go.Bar(x=[lengths[i]], y=[components[i]], orientation='h', marker=dict(color=colors[i]),
                       text=[f"{lengths[i]:.2f} cm"], textposition='auto')
                for i in range(len(components))
            ])
            bar_fig.update_layout(barmode='stack', xaxis_title="Length (cm)", width=800, height=300)
            st.plotly_chart(bar_fig)
        else:
            st.write("Cannot display components due to negative rope length.")

        st.subheader("Fitted Equation")
        terms = [f"{coef:+.6f}x^{len(coefficients)-1-i}" if i < len(coefficients)-1 else f"{coef:+.6f}" for i, coef in enumerate(coefficients)]
        st.latex(f"L_{{elastic, stretched}} = {' '.join(terms)}")

    except Exception as e:
        st.error(f"Error: {str(e)}. Check data or lower polynomial degree.")
