# BT3040-TPA

## AMP Prediction App

This Streamlit application predicts Anti Microbial Peptides (AMPs) using multiple machine learning models and Principal Component Analysis (PCA) models. The app allows users to input a peptide sequence and returns predictions from various models, along with an ensemble prediction.

## Features

- **User Input Sequence**: Enter a peptide sequence of length greater than 4.
- **Extracted Features**: Displays the features extracted from the input sequence.
- **Individual Model Predictions**: Shows predictions from individual models (SVM, Logistic Regression, Decision Tree, CatBoost).
- **PCA Model Predictions**: Displays predictions from PCA-transformed models.
- **Ensemble Prediction**: Provides a combined prediction based on all models.
- **Feature Information**: Toggle button to display detailed feature information.

### Requirements

- Python 3.x
- Required Python packages are listed in `requirements.txt`.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/amp-prediction-app.git
    cd amp-prediction-app
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have the necessary model files (`svm_model.pkl`, `logistic_regression_model.pkl`, `decision_tree_model.pkl`, `catboost_model.pkl`, `label_encoder.pkl`) and PCA model files (`pca_5.pkl`, `pca_10.pkl`, `pca_20.pkl`, `decision_tree_pca_5.pkl`, `decision_tree_pca_10.pkl`, `decision_tree_pca_20.pkl`, `svm_pca_5.pkl`, `svm_pca_10.pkl`, `svm_pca_20.pkl`, `logistic_regression_pca_5.pkl`, `logistic_regression_pca_10.pkl`, `logistic_regression_pca_20.pkl`) in the same directory as the script.

5. Place the `Feature_Information.xlsx` file in the same directory as the script.

## Usage

Run the Streamlit app using the following command:
```bash
streamlit run app.py
