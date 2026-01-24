# Developer Guide - Tailyo Technologies Service Recommendation System

This guide provides technical details for developers maintaining or extending the ML-Powered Service Recommendation System.

## 🏗️ Architecture Overview

The system follows a standard modular ML pipeline architecture:

1.  **Data Layer**: Raw CSV data (`data/raw`) is processed and cleaned (`data/cleaned`).
2.  **Preprocessing Layer**: Text normalization, missing value imputation, and feature engineering.
3.  **Model Layer**: 
    -   **Encoders**: OneHotEncoder for business types/locations, OrdinalEncoder for pricing.
    -   **Vectorizers**: TF-IDF for service descriptions.
    -   **Inference Engine**: NearestNeighbor (KNN) algorithm with Cosine Similarity metric.
4.  **Application Layer**: Streamlit frontend for user interaction.

## 💻 Tech Stack

-   **Language**: Python 3.8+
-   **Core Libraries**: `pandas`, `numpy`, `scikit-learn`, `scipy`
-   **Web Framework**: `streamlit`
-   **Testing**: `pytest`

## 🗂️ Key Components

### 1. Preprocessing (`src/preprocessing/`)
-   **`data_cleaner.py`**: Handles standardization of raw input data.
-   **`feature_engineering.py`**: Transforms cleaned data into numerical feature matrices (`features.npy`) and saves encoders (`encoders.pkl`).

### 2. Models (`src/models/`)
-   **`recommendation_engine.py`**: The core class. Loads artifacts, filters data based on hard constraints (e.g., City), and computes similarity scores using the feature matrix.
-   **`user_encoder.py`**: Converts user form input into a 1xN query vector matching the training data schema.
-   **`explanation_generator.py`**: Rule-based logic to generate human-readable "Why This Match?" bullets.

## 🔄 workflows

### Adding New Data
1.  Add new raw data to `data/raw/`.
2.  Run the cleaning pipeline:
    ```bash
    python src/preprocessing/data_cleaner.py
    ```
3.  Re-generate features and encoders:
    ```bash
    python src/preprocessing/feature_engineering.py
    ```
4.  Restart the Streamlit app to load the new artifacts.

### Retraining/Modifying Models
1.  Modify `src/models/recommendation_engine.py` to change the algorithm (e.g., change `metric='cosine'` to `metric='euclidean'`).
2.  Run tests to ensure no regression:
    ```bash
    pytest tests/
    ```

## 🧪 Testing

The project uses `pytest` for unit and integration testing.
-   **`tests/test_preprocessing.py`**: Validates data cleaning and feature shapes.
-   **`tests/test_system_integration.py`**: End-to-end tests for the recommendation engine.

Run all tests:
```bash
pytest
```
