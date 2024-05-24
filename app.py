import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
import propy.CTD as pc
import propy.PseudoAAC as pp
import propy.AAComposition as pa

# Load models and label encoder
svm_model = joblib.load('models/svm_model.pkl')
logistic_regression_model = joblib.load('models/logistic_regression_model.pkl')
decision_tree_model = joblib.load('models/decision_tree_model.pkl')
catboost_model = joblib.load('models/catboost_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Load PCA models
pca_models = {pca_value: joblib.load(f'models/pca_{pca_value}.pkl') for pca_value in [5, 10, 20]}
decision_tree_pca_models = {pca_value: joblib.load(f'models/decision_tree_pca_{pca_value}.pkl') for pca_value in [5, 10, 20]}
svm_pca_models = {pca_value: joblib.load(f'models/svm_pca_{pca_value}.pkl') for pca_value in [5, 10, 20]}
logistic_regression_pca_models = {pca_value: joblib.load(f'models/logistic_regression_pca_{pca_value}.pkl') for pca_value in [5, 10, 20]}

# List of standard amino acids
standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

# Define the structure of the zero-filled feature dictionary based on actual feature names
def get_zero_feature():
    zero_feature = {}
    zero_feature.update(pp.GetAPseudoAAC("ACDEFGHIKLMNPQRSTVWY", lamda=4, weight=0.05))
    zero_feature.update(pp.GetAAComposition("ACDEFGHIKLMNPQRSTVWY"))
    zero_feature.update(pc.CalculateC("ACDEFGHIKLMNPQRSTVWY"))
    for key in zero_feature:
        zero_feature[key] = 0
    return zero_feature

zero_feature = get_zero_feature()

# Define the descriptor_calc function
def descriptor_calc(sequence):
    feature = {}
    if not set(sequence).issubset(standard_amino_acids):
        feature = zero_feature.copy()  # Ensure a unique copy for each sequence
        return feature
    if len(sequence) > 4:
        feature.update(pp.GetAPseudoAAC(sequence, lamda=4, weight=0.05))
        feature.update(pp.GetAAComposition(sequence))
        feature.update(pc.CalculateC(sequence))
    return feature

# Function to make predictions
def predict_ensemble(models, pca_models, features):
    features_array = np.array(list(features.values())).reshape(1, -1)  # Convert dictionary to NumPy array

    original_predictions = [model.predict(features_array)[0] for model in models]
    pca_predictions = []

    for pca_value, pca in pca_models.items():
        transformed_features = pca.transform(features_array)
        pca_predictions.extend([
            decision_tree_pca_models[pca_value].predict(transformed_features)[0],
            svm_pca_models[pca_value].predict(transformed_features)[0],
            logistic_regression_pca_models[pca_value].predict(transformed_features)[0]
        ])

    all_predictions = original_predictions + pca_predictions
    ensemble_prediction = np.round(np.mean(all_predictions)).astype(int)
    return original_predictions, pca_predictions, ensemble_prediction

# Streamlit app
st.title("AMP Prediction App")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    sequence = st.sidebar.text_area('Sequence', height=100, placeholder='Enter sequence (length > 4)')
    return sequence

sequence = user_input_features()

feature_info = pd.read_excel('Feature_Information.xlsx')

if sequence:
    st.subheader('User Input Sequence')
    st.write(sequence)
    
    # Extract features from the sequence
    features = descriptor_calc(sequence)
    features_df = pd.DataFrame([features])

    st.subheader('Extracted Features')
    st.write(features_df)
    # Add a button
    if 'show_df' not in st.session_state:
        st.session_state.show_df = False
    
    # Add a button to toggle the DataFrame display
    if st.button('Details'):
        st.session_state.show_df = not st.session_state.show_df
    
    # Display the DataFrame based on the session state
    if st.session_state.show_df:
        st.write(feature_info)
        
    models = [svm_model, logistic_regression_model, decision_tree_model, catboost_model]

    # Make ensemble prediction
    original_predictions, pca_predictions, ensemble_prediction = predict_ensemble(models, pca_models, features)

    # Decode the ensemble prediction
    decoded_original_predictions = label_encoder.inverse_transform(original_predictions)
    decoded_pca_predictions = label_encoder.inverse_transform(pca_predictions)
    decoded_ensemble_prediction = label_encoder.inverse_transform([ensemble_prediction])[0]
     
    st.subheader('Individual Model Predictions')
    model_names = ['SVM', 'Logistic Regression', 'Decision Tree', 'CatBoost']
    original_predictions_data = [(name, prediction) for name, prediction in zip(model_names, decoded_original_predictions)]
    st.table(original_predictions_data)
    
    pca_model_names = []
    for pca_value in pca_models.keys():
        pca_model_names.extend([
            f'Decision Tree PCA {pca_value}', 
            f'SVM PCA {pca_value}', 
            f'Logistic Regression PCA {pca_value}'
        ])
    # Create a dictionary to store PCA predictions for each model
    pca_predictions_dict = {}
    for model_name, pca_value in zip(pca_model_names, decoded_pca_predictions):
        model, pca_component = model_name.split(' PCA ')
        if model not in pca_predictions_dict:
            pca_predictions_dict[model] = {}
        pca_predictions_dict[model][pca_component] = pca_value

    # Convert the dictionary to a pandas DataFrame
    pca_predictions_df = pd.DataFrame(pca_predictions_dict).transpose()

    # Display the DataFrame as a table
    st.subheader('PCA Model Predictions')
    st.write(pca_predictions_df)

    st.subheader(f'Ensemble Prediction : {decoded_ensemble_prediction}')
    
    st.write(f'AMP stands for Anti Microbial Peptides')