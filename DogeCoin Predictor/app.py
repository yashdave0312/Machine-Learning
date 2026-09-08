import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dogecoin Price Predictor",
    page_icon="🪙",
    layout="wide"
)

# ---------------------------------------------------------
# Load trained model and preprocessing objects
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("LinearRegression_model.pkl")
    scaler = joblib.load("scalar.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns


try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Required model file is missing: {e.filename}. "
        "Keep LinearRegression_model.pkl, scalar.pkl and columns.pkl "
        "in the same folder as app.py."
    )
    st.stop()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🪙 Dogecoin Price Predictor")
st.markdown(
    "### Predict the **next day's Dogecoin closing price** "
    "using the trained Linear Regression model."
)

st.info(
    "This app uses the same features and StandardScaler preprocessing "
    "used in your `main.ipynb`."
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("📌 Model Information")
st.sidebar.write("**Model:** Linear Regression")
st.sidebar.write("**Target:** Next day's Close price")
st.sidebar.write("**Scaler:** StandardScaler")

st.sidebar.markdown("---")
st.sidebar.write("Enter the market data below and click **Predict Price**.")

# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------
st.subheader("📊 Enter Dogecoin Market Data")

col1, col2, col3 = st.columns(3)

with col1:
    open_price = st.number_input(
        "Open Price",
        min_value=0.0,
        value=0.10,
        format="%.6f"
    )

    high_price = st.number_input(
        "High Price",
        min_value=0.0,
        value=0.11,
        format="%.6f"
    )

    low_price = st.number_input(
        "Low Price",
        min_value=0.0,
        value=0.09,
        format="%.6f"
    )

with col2:
    close_price = st.number_input(
        "Current Close Price",
        min_value=0.0,
        value=0.10,
        format="%.6f"
    )

    adj_close = st.number_input(
        "Adjusted Close",
        min_value=0.0,
        value=0.10,
        format="%.6f"
    )

    volume = st.number_input(
        "Trading Volume",
        min_value=0.0,
        value=100000000.0,
        step=1000000.0,
        format="%.0f"
    )

with col3:
    year = st.number_input(
        "Year",
        min_value=2017,
        max_value=2100,
        value=2026,
        step=1
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=9,
        step=1
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=8,
        step=1
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
st.markdown("---")

if st.button("🚀 Predict Next Day Price", use_container_width=True):
    try:
        input_data = pd.DataFrame(
            [[
                open_price,
                high_price,
                low_price,
                close_price,
                adj_close,
                volume,
                year,
                month,
                day
            ]],
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "Year",
                "Month",
                "Day"
            ]
        )

        # Ensure exact feature order from training
        input_data = input_data[feature_columns]

        # Apply the same scaler used during training
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Next Day Closing Price",
            value=f"${prediction:.8f}"
        )

        st.caption(
            "The prediction represents the next day's Close price "
            "according to the trained model."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Dogecoin Price Prediction • Machine Learning Project"
)
