
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction")
st.write(
    "Enter the feature measurements below to obtain "
    "a prediction from the trained neural network."
)

st.warning(
    "Educational project only. This application is not "
    "intended for medical diagnosis or treatment decisions."
)


# ==========================================
# 2. FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "cancer_model.pth"
IMPUTER_PATH = MODEL_DIR / "imputer.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"


# ==========================================
# 3. CHECK REQUIRED FILES
# ==========================================

required_files = [
    MODEL_PATH,
    IMPUTER_PATH,
    SCALER_PATH,
    FEATURES_PATH
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error(
        "Required files are missing. Run model.py again "
        "to train the model and save its preprocessing objects."
    )

    st.code("\n".join(missing_files))
    st.stop()


# ==========================================
# 4. DEFINE THE SAME NEURAL NETWORK
# ==========================================

class BreastCancerNN(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 5. LOAD SAVED MODEL AND PREPROCESSORS
# ==========================================

@st.cache_resource
def load_model():

    features = joblib.load(FEATURES_PATH)

    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    model = BreastCancerNN(
        input_features=len(features)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, imputer, scaler, features


try:

    model, imputer, scaler, features = load_model()

except Exception as e:

    st.error("Could not load the saved model.")
    st.exception(e)
    st.stop()


# ==========================================
# 6. USER INPUT FORM
# ==========================================

st.header("Enter Feature Measurements")

st.info(
    "Enter measurements in the same units and feature order "
    "used by the original dataset. Default values are dataset "
    "medians for demonstration, not patient measurements."
)

# Read feature ranges from the original CSV for convenient inputs.
# The CSV is used only to determine reasonable UI defaults.
DATA_PATH = BASE_DIR / "breast_cancer.csv"

if DATA_PATH.exists():

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(axis=1, how="all")

    reference = df.reindex(columns=features)

    for feature in features:
        reference[feature] = pd.to_numeric(
            reference[feature],
            errors="coerce"
        )

else:
    reference = None


with st.form("prediction_form"):

    input_values = {}

    columns = st.columns(3)

    for i, feature in enumerate(features):

        default_value = 0.0

        if reference is not None:

            valid_values = reference[feature].dropna()

            if not valid_values.empty:
                default_value = float(valid_values.median())

        with columns[i % 3]:

            input_values[feature] = st.number_input(
                label=feature,
                value=default_value,
                format="%.6f",
                key=f"input_{feature}"
            )

    submitted = st.form_submit_button(
        "Predict Diagnosis",
        use_container_width=True
    )


# ==========================================
# 7. MAKE PREDICTION
# ==========================================

if submitted:

    try:

        # Preserve the exact feature order used in training.
        input_df = pd.DataFrame(
            [input_values],
            columns=features
        )

        input_array = input_df.to_numpy(dtype=np.float64)

        if not np.isfinite(input_array).all():

            st.error(
                "Please enter valid finite numeric values."
            )

            st.stop()

        # Apply the SAME preprocessing fitted during training.
        input_imputed = imputer.transform(input_df)

        input_scaled = scaler.transform(input_imputed)

        input_tensor = torch.tensor(
            input_scaled,
            dtype=torch.float32
        )

        # Model inference
        with torch.no_grad():

            logits = model(input_tensor)

            probability = torch.sigmoid(logits).item()

            prediction = int(probability >= 0.5)

        # Display results
        st.divider()

        st.header("Prediction Result")

        if prediction == 1:

            st.error(
                "Predicted Class: Malignant"
            )

        else:

            st.success(
                "Predicted Class: Benign"
            )

        st.metric(
            label="Estimated probability of malignant class",
            value=f"{probability:.2%}"
        )

        st.progress(float(probability))

        st.caption(
            "The probability is the model's estimate, not a "
            "confirmed diagnosis or a measure of clinical certainty."
        )

    except Exception as e:

        st.error("Prediction failed. Check the input and saved files.")
        st.exception(e)


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Built with Python, PyTorch, scikit-learn and Streamlit."
)

