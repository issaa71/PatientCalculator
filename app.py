"""
Web version of the Pain Score Calculator using Streamlit

Run with: streamlit run pain_calculator_web_fixed.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Instead of importing from pain_calculator, include the necessary functions directly here
# Constants
T3_IMPORTANT_FEATURES = [
    'LOS', 'BMI_Current', 'WOMACP_5', 'WeightCurrent', 'ICOAPC_3',
    'ICOAPC_1', 'AgePreOp', 'WOMACP_3', 'WalkPain', 'MobilityAidWalker',
    'Pre-Op Pain', 'HeightCurrent', 'ResultsRelief'
]

T5_IMPORTANT_FEATURES = [
    'AgePreOp', 'BMI_Current', 'WeightCurrent', 'HeightCurrent', 'LOS',
    'WOMACP_5', 'ResultsRelief', 'ICOAPC_3', 'Pre-Op Pain', 'WalkPain',
    'Approach', 'HeadSize'
]

MODELS_DIR = 'trained_models'

def get_feature_descriptions():
    """Return descriptions for the features used in the models"""
    feature_descriptions = {
        'LOS': 'Length of stay (days)',
        'BMI_Current': 'Body Mass Index',
        'WOMACP_5': 'WOMAC Pain Score Question 5 (0-4)',
        'WeightCurrent': 'Current weight (kg)',
        'ICOAPC_3': 'ICOA Pain Score Question 3 (0-4)',
        'ICOAPC_1': 'ICOA Pain Score Question 1 (0-4)',
        'AgePreOp': 'Age at pre-operation (years)',
        'WOMACP_3': 'WOMAC Pain Score Question 3 (0-4)',
        'WalkPain': 'Pain while walking (0-10)',
        'MobilityAidWalker': 'Uses walker as mobility aid (0=No, 1=Yes)',
        'Pre-Op Pain': 'Pre-operation pain score (0-10)',
        'HeightCurrent': 'Current height (cm)',
        'ResultsRelief': 'Expected relief result (1-5)',
        'Approach': 'Surgical approach (e.g., "Posterior", "Anterior")',
        'HeadSize': 'Size of the femoral head implant (mm)'
    }
    return feature_descriptions

def load_models(timepoint):
    """Load saved models and preprocessors"""
    model_path = os.path.join(MODELS_DIR, f'{timepoint.lower()}_model.pkl')
    preprocessor_path = os.path.join(MODELS_DIR, f'{timepoint.lower()}_preprocessor.pkl')
    features_path = os.path.join(MODELS_DIR, f'{timepoint.lower()}_features.pkl')
    
    if not (os.path.exists(model_path) and 
            os.path.exists(preprocessor_path) and 
            os.path.exists(features_path)):
        raise FileNotFoundError(
            f"Pre-trained models not found in '{MODELS_DIR}' directory. "
            f"Please run 'train_models.py' first to create the models."
        )
    
    # Load model, preprocessor, and feature info
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    return model, preprocessor, features

def predict_pain(patient_data, timepoint='T3'):
    """
    Predict pain score for a patient using the specified timepoint model.
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient features
    timepoint : str, optional (default='T3')
        Timepoint for which to predict pain ('T3' or 'T5')
    
    Returns:
    --------
    float
        Predicted pain score
    """
    # Ensure timepoint is uppercase
    timepoint = timepoint.upper()
    if timepoint not in ['T3', 'T5']:
        raise ValueError("Timepoint must be 'T3' or 'T5'")
    
    try:
        # Load model, preprocessor, and feature info
        model, preprocessor, features = load_models(timepoint)
    except FileNotFoundError as e:
        st.error(f"Error: {str(e)}")
        return None
    
    # Determine required features
    required_features = T3_IMPORTANT_FEATURES if timepoint == 'T3' else T5_IMPORTANT_FEATURES
    
    # Check that all required features are provided
    missing_features = [f for f in required_features if f not in patient_data]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Create DataFrame from patient data
    patient_df = pd.DataFrame([patient_data])
    
    # Convert HeadSize to string if present
    if 'HeadSize' in patient_df.columns:
        patient_df['HeadSize'] = patient_df['HeadSize'].astype(str)
    
    # Preprocess the patient data
    patient_processed = preprocessor.transform(patient_df)
    
    # Make prediction
    prediction = model.predict(patient_processed)[0]
    
    # Clip prediction to valid range [0, 8]
    prediction = np.clip(prediction, 0, 8)
    
    return prediction

# Set up page
st.set_page_config(
    page_title="Hip Replacement Pain Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Hip Replacement Pain Predictor")
    st.markdown("""
    This tool predicts post-operative pain scores for hip replacement patients at two timepoints:
    * **T3**: 6 weeks post-operation
    * **T5**: 6 months post-operation
    
    **Note**: This calculator is based on statistical models and should be used only as a reference. 
    Actual patient outcomes may vary. Always consult with healthcare professionals for medical advice.
    """)
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        st.warning(f"Created '{MODELS_DIR}' directory. Models must be trained before using the calculator.")
    
    # Check if models exist
    model_files_exist = check_models_exist()
    
    if not model_files_exist:
        st.error("""
        Pre-trained models not found! 
        
        When deploying online, you must include your trained models in the 'trained_models' directory.
        If this is your first time setting up, run 'train_models.py' locally and then upload the resulting
        'trained_models' directory to your deployment.
        """)
        
        # In the online deployment, show a demo mode
        st.info("Continuing in DEMO MODE with simulated predictions (not using actual models)")
    
    # Sidebar
    st.sidebar.title("Pain Score Prediction")
    timepoint = st.sidebar.radio("Select timepoint to predict:", ["T3 (6 weeks)", "T5 (6 months)"])
    
    # Remove parentheses from timepoint string
    timepoint_code = timepoint.split(" ")[0]
    
    st.header(f"Patient Information for {timepoint}")
    
    # Get required features based on timepoint
    required_features = T3_IMPORTANT_FEATURES if timepoint_code == "T3" else T5_IMPORTANT_FEATURES
    feature_descriptions = get_feature_descriptions()
    
    # Use columns to organize the layout
    col1, col2 = st.columns(2)
    
    # Initialize patient data dictionary
    patient_data = {}
    
    # Create input fields for required features
    for i, feature in enumerate(required_features):
        description = feature_descriptions.get(feature, "")
        
        # Decide which column to put the feature in (alternate between columns)
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            if feature == 'MobilityAidWalker':
                patient_data[feature] = int(st.selectbox(
                    f"{feature} ({description})",
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes"
                ))
            elif feature == 'Approach':
                patient_data[feature] = st.selectbox(
                    f"{feature} ({description})",
                    options=["Posterior", "Anterior", "Lateral", "Other"]
                )
            elif feature in ['WOMACP_5', 'WOMACP_3', 'ICOAPC_3', 'ICOAPC_1']:
                # WOMAC and ICOA scores are on 0-4 scale
                patient_data[feature] = st.slider(
                    f"{feature} ({description})",
                    min_value=0,
                    max_value=4,
                    step=1
                )
            elif feature == 'ResultsRelief':
                # ResultsRelief is on 1-5 scale
                patient_data[feature] = st.slider(
                    f"{feature} ({description})",
                    min_value=1,
                    max_value=5,
                    step=1
                )
            elif feature == 'WalkPain' or feature == 'Pre-Op Pain':
                # Pain scores are on 0-10 scale
                patient_data[feature] = st.slider(
                    f"{feature} ({description})",
                    min_value=0,
                    max_value=10,
                    step=1
                )
            elif feature == 'HeadSize':
                # HeadSize is typically 28, 32, or 36 mm
                patient_data[feature] = st.selectbox(
                    f"{feature} ({description})",
                    options=["28", "32", "36", "40", "Other"]
                )
            else:
                # Default numeric input for other fields
                patient_data[feature] = st.number_input(
                    f"{feature} ({description})",
                    value=0.0,
                    step=0.1
                )
    
    # Predict button
    if st.button("Predict Pain Score"):
        # If in demo mode or models don't exist, use a simulated prediction
        if not model_files_exist:
            # Simple simulation logic based on input values
            if timepoint_code == "T3":
                # In demo mode, create a simple formula for demonstration
                bmi_factor = 0.01 * patient_data.get('BMI_Current', 25)
                age_factor = 0.01 * patient_data.get('AgePreOp', 65)
                preop_factor = 0.2 * patient_data.get('Pre-Op Pain', 5)
                walk_factor = 0.1 * patient_data.get('WalkPain', 5)
                
                prediction = 2.0 + bmi_factor + age_factor + preop_factor - walk_factor
                prediction = max(0, min(8, prediction))  # Ensure between 0-8
            else:
                # Different formula for T5
                bmi_factor = 0.008 * patient_data.get('BMI_Current', 25)
                age_factor = 0.005 * patient_data.get('AgePreOp', 65)
                preop_factor = 0.15 * patient_data.get('Pre-Op Pain', 5)
                approach_factor = 0.5 if patient_data.get('Approach') == 'Posterior' else 0
                
                prediction = 1.5 + bmi_factor + age_factor + preop_factor - approach_factor
                prediction = max(0, min(8, prediction))  # Ensure between 0-8
                
            st.info("DEMO MODE: Using simulated prediction (not from trained model)")
        else:
            try:
                # Use actual model for prediction
                prediction = predict_pain(patient_data, timepoint_code)
                if prediction is None:
                    st.error("Error loading models. Please check if models are properly trained.")
                    return
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                return
        
        # Display results
        st.header("Prediction Results")
        
        # Create a gauge-chart-like display
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Create a color gradient for the gauge
        cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green reversed (red is high pain)
        
        # Background bar (grey)
        ax.barh(0, 8, color='lightgrey', alpha=0.3)
        
        # Colored bar based on prediction
        ax.barh(0, prediction, color=cmap(prediction/8))
        
        # Customize appearance
        ax.set_xlim(0, 8)
        ax.set_yticks([])
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_xticklabels(['0\nNo Pain', '2', '4', '6', '8\nExtreme Pain'])
        
        # Add marker for the prediction
        ax.plot(prediction, 0, 'ko', markersize=12)
        ax.text(prediction, 0, f'{prediction:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Remove y-axis
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Display the plot
        st.pyplot(fig)
        
        # Interpret the prediction
        if prediction <= 2:
            interpretation = "minimal"
            color = "green"
        elif prediction <= 4:
            interpretation = "mild"
            color = "blue"
        elif prediction <= 6:
            interpretation = "moderate"
            color = "orange"
        else:
            interpretation = "severe"
            color = "red"
        
        st.markdown(f"<h3 style='color:{color}'>Predicted pain level: <b>{interpretation}</b></h3>", unsafe_allow_html=True)
        
        # Additional interpretation
        st.markdown(f"""
        This prediction suggests a **{interpretation}** pain level ({prediction:.1f}/8) at {timepoint}.
        
        **Remember**:
        - This is a statistical prediction and individual results may vary
        - The model has 40-49% accuracy within ±1 point of actual pain
        - Always consult with healthcare professionals for medical advice
        """)

def check_models_exist():
    """Check if pre-trained models exist"""
    model_files = [
        os.path.join(MODELS_DIR, 't3_model.pkl'),
        os.path.join(MODELS_DIR, 't3_preprocessor.pkl'),
        os.path.join(MODELS_DIR, 't5_model.pkl'),
        os.path.join(MODELS_DIR, 't5_preprocessor.pkl')
    ]
    
    return all(os.path.exists(f) for f in model_files)

if __name__ == "__main__":
    main()
