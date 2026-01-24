import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

"""
Feature Engineering Module
-------------------------
Transforms cleaned data into numerical feature matrices suitable for machine learning models.
Handles:
- Ordinal encoding for price categories.
- One-hot encoding for categorical variables.
- TF-IDF vectorization for text descriptions.
- Feature scaling and weighting.
"""

# Paths
CLEANED_DATA_PATH = r"E:\Internship\ml-service-recommendation\data\cleaned\service_recommendation_data_cleaned.csv"
PROCESSED_DATA_DIR = r"E:\Internship\ml-service-recommendation\data\processed"
MODELS_DIR = r"E:\Internship\ml-service-recommendation\src\models"

def load_data(path):
    """Load cleaned data from CSV."""
    return pd.read_csv(path)

def process_features(df):
    """
    Generate feature matrix from the dataframe.

    Logic:
    1. Business Logic Features: Manual mapping of Price (0.25-1.0) and Language support.
    2. Categorical Features: One-Hot Encoding for Business Type and Location.
    3. Text Features: TF-IDF for descriptions (boosted by 10x to prioritize semantic matching).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        tuple: 
            - final_feature_matrix (numpy.ndarray): The complete feature set.
            - service_ids (numpy.ndarray): Corresponding Service IDs.
            - encoders (tuple): (OneHotEncoder object, TfidfVectorizer object).
            - all_feature_names (numpy.ndarray): Names of all generated features.
    """
    print("Starting feature engineering...")
    
    # 1. Price Category Encoding (Ordinal - Normalized 0.25 to 1.0)
    price_map = {'low': 0.25, 'medium': 0.50, 'high': 0.75, 'premium': 1.0}
    df['price_score'] = df['Price_Category'].map(price_map).fillna(0.5) # Default to Medium
    
    # 2. Language Support (Custom Multi-Hot Encoding)
    # create independent flags
    df['lang_english'] = df['Language_Support'].apply(lambda x: 1 if x in ['english', 'both'] else 0)
    df['lang_hindi'] = df['Language_Support'].apply(lambda x: 1 if x in ['hindi', 'both'] else 0)
    df['lang_regional'] = df['Language_Support'].apply(lambda x: 1 if x == 'regional' else 0)
    
    # 3. Location (One-Hot + Remote Flag)
    df['is_remote'] = df['Location_Area'].apply(lambda x: 1 if x == 'remote' else 0)
    
    # 4. TF-IDF for Description
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))
    feature_names_tfidf = tfidf.get_feature_names_out()
    
    # 5. One-Hot Encoding for Business Type and Location
    # We use sklearn's OneHotEncoder for these to easily handle user input later
    # Note: We don't include Language here because we manually handled it above
    categorical_features = ['Target_Business_Type', 'Location_Area']
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_matrix = ohe.fit_transform(df[categorical_features])
    feature_names_ohe = ohe.get_feature_names_out(categorical_features)
    
    # 6. Combine all features
    # Numerical/Manual features: price_score, lang_english, lang_hindi, lang_regional, is_remote
    manual_features = df[['price_score', 'lang_english', 'lang_hindi', 'lang_regional', 'is_remote']].values
    feature_names_manual = ['price_score', 'lang_english', 'lang_hindi', 'lang_regional', 'is_remote']
    
    # Concatenate: [Manual (5), OHE (N), TF-IDF (500)]
    # BOOST TEXT RELAVANCE: Multiply TF-IDF by 10.0 (Aggressive boost to fix ranking)
    final_feature_matrix = np.hstack([manual_features, ohe_matrix, tfidf_matrix.toarray() * 10.0])
    
    all_feature_names = np.concatenate([feature_names_manual, feature_names_ohe, feature_names_tfidf])
    
    print(f"Feature Matrix Shape: {final_feature_matrix.shape}")
    
    return final_feature_matrix, df['Service_ID'].values, (ohe, tfidf), all_feature_names

def save_artifacts(matrix, service_ids, encoders, feature_names):
    """
    Save generated feature artifacts to disk.

    Artifacts:
    - features.npy: The numerical feature matrix.
    - service_ids.npy: The ordered service IDs.
    - encoders.pkl: The fitted encoders for transforming new user input.
    - feature_names.pkl: Names of the features for debugging/explanation.

    Args:
        matrix (numpy.ndarray): Feature matrix.
        service_ids (numpy.ndarray): Service IDs.
        encoders (tuple): Fitted encoders.
        feature_names (list): Feature names.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save Matrix
    np.save(os.path.join(PROCESSED_DATA_DIR, 'features.npy'), matrix)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'service_ids.npy'), service_ids)
    
    # Save Encoders (Tuple of ohe, tfidf)
    with open(os.path.join(MODELS_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
        
    # Save Feature Names
    with open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
        
    print("Artifacts saved successfully.")

def main():
    try:
        df = load_data(CLEANED_DATA_PATH)
        matrix, service_ids, encoders, feature_names = process_features(df)
        save_artifacts(matrix, service_ids, encoders, feature_names)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
