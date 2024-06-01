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

4. Place the `Feature_Information.xlsx` file in the same directory as the script.

## Usage

Run the Streamlit app using the following command:
```bash
streamlit run app.py
