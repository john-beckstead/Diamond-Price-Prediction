import streamlit as st
import requests
import numpy as np
import pandas as pd
import json

# --- 1. DATABRICKS ENDPOINT CONFIGURATION (REPLACE PLACEHOLDERS) ---

# Replace with your Databricks host (e.g., dbc-xxxxxxxx.cloud.databricks.com)
DATABRICKS_HOST = "https://dbc-acdc78b8-1dcc.cloud.databricks.com"
# Replace with the name of your registered model serving endpoint
SERVING_ENDPOINT_NAME = "diamond_price_prediction"
# Replace with your Databricks Personal Access Token (PAT)
# WARNING: In a production app, use Databricks Secrets or another secure method.
DATABRICKS_TOKEN = "dapi4827ad668d0f619715018692239e9092"

# Construct the full invocation URL
URL = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"


# --- 2. MODEL INPUT DEFINITIONS ---

# Categorical features and their valid values
CUT_OPTIONS = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
COLOR_OPTIONS = ['D', 'E', 'F', 'G', 'H', 'I', 'J'] # D is best, J is worst
CLARITY_OPTIONS = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'] # IF is best, I1 is worst


def get_prediction(carat, cut, color, clarity, depth, table, x, y, z):
    """
    Sends a request to the Databricks Model Serving endpoint and returns the price.
    """
    
    # 1. Create the data payload matching the model signature (9 columns)
    # The Databricks endpoint expects the data in the 'dataframe_split' format.
    data = {
        "dataframe_split": {
            "columns": ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"],
            "data": [[carat, cut, color, clarity, depth, table, x, y, z]]
        }
    }

    # 2. Define headers including authorization
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        # 3. Send the POST request
        response = requests.post(URL, headers=headers, json=data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # 4. Process the response
        predictions = response.json()['predictions']
        
        if not predictions:
            return "Error: Model returned an empty prediction list."
        
        # The model predicts the log(price + 1). We must reverse this transformation.
        log_price = predictions[0]
        actual_price = np.expm1(log_price) # Equivalent to: np.exp(log_price) - 1
        
        return actual_price

    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh}"
    except requests.exceptions.ConnectionError as errc:
        return f"Error Connecting: {errc}"
    except requests.exceptions.Timeout as errt:
        return f"Timeout Error: {errt}"
    except requests.exceptions.RequestException as err:
        return f"Unknown Request Error: {err}"
    except json.JSONDecodeError:
        return f"Error decoding JSON response: {response.text}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- 3. STREAMLIT USER INTERFACE ---

st.title("💎 Diamond Price Predictor (Databricks Model Serving)")
st.markdown("Use the controls below to define the characteristics of a diamond and get a real-time price prediction from the deployed model.")

with st.sidebar:
    st.header("Diamond Characteristics")
    st.markdown("---")
    
    # Numerical Inputs
    carat = st.slider("Carat Weight", 0.2, 5.0, 0.7, 0.01)
    depth = st.slider("Depth (%)", 50.0, 75.0, 61.8, 0.1)
    table = st.slider("Table Width (%)", 50.0, 80.0, 57.0, 0.1)
    
    # Dimensions (x, y, z) - using typical ranges
    x = st.number_input("Length (mm) [x]", min_value=0.0, max_value=12.0, value=5.5)
    y = st.number_input("Width (mm) [y]", min_value=0.0, max_value=12.0, value=5.5)
    z = st.number_input("Height (mm) [z]", min_value=0.0, max_value=10.0, value=3.4)

    st.markdown("---")

    # Categorical Inputs
    cut = st.selectbox("Cut Quality", options=CUT_OPTIONS, index=CUT_OPTIONS.index('Ideal'))
    color = st.selectbox("Color Grade", options=COLOR_OPTIONS, index=COLOR_OPTIONS.index('E'))
    clarity = st.selectbox("Clarity Grade", options=CLARITY_OPTIONS, index=CLARITY_OPTIONS.index('VS1'))


# Main area for prediction result
st.markdown(f"### Current Diamond Specifications")
st.table(pd.DataFrame({
    'Feature': ['Carat', 'Cut', 'Color', 'Clarity', 'Depth', 'Table', 'Length (x)', 'Width (y)', 'Height (z)'],
    'Value': [carat, cut, color, clarity, depth, table, x, y, z]
}))

if st.button("Get Prediction", type="primary"):
    with st.spinner('Querying Databricks Model Serving Endpoint...'):
        
        # Call the prediction function
        result = get_prediction(carat, cut, color, clarity, depth, table, x, y, z)

        if isinstance(result, str):
            # Display error message
            st.error(result)
        else:
            # Display successful prediction
            st.success("Prediction Successful!")
            st.metric(
                label="Predicted Price (USD)", 
                value=f"${result:,.2f}"
            )
            st.balloons()