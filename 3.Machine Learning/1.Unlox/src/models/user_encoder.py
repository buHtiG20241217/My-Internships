import pandas as pd
import numpy as np
import pickle
import os

# Paths (adjust as needed if running from different root)
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

class UserEncoder:
    def __init__(self):
        self.encoders = self._load_encoders()
        self.ohe = self.encoders[0] # OneHotEncoder
        self.tfidf = self.encoders[1] # TfidfVectorizer
        
    def _load_encoders(self):
        path = os.path.join(MODELS_DIR, 'encoders.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Encoders not found at {path}. Run feature_engineering.py first.")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def encode_user_input(self, user_input):
        """
        Transforms user input dict into a 1xN feature vector.
        Expected keys: 'Price_Category', 'Language_Support', 'Location_Area', 'Target_Business_Type', 'Description'
        """
        
        # 1. Manual Features (5)
        # Price (Normalized 0.25-1.0)
        price_map = {'low': 0.25, 'medium': 0.50, 'high': 0.75, 'premium': 1.0}
        price_val = user_input.get('Price_Category', 'medium').lower()
        price_score = price_map.get(price_val, 0.5)
        
        # Language
        langs = [l.strip().lower() for l in user_input.get('Language_Support', [])] # Expecting list
        # Handle simple string input just in case
        if isinstance(user_input.get('Language_Support'), str):
             langs = [user_input.get('Language_Support').lower()]
             
        lang_english = 1 if 'english' in langs or 'both' in langs else 0
        lang_hindi = 1 if 'hindi' in langs or 'both' in langs else 0
        lang_regional = 1 if 'regional' in langs else 0
        
        # Location Remote Flag
        loc = user_input.get('Location_Area', '').lower()
        is_remote = 1 if loc == 'remote' else 0
        
        manual_features = np.array([[price_score, lang_english, lang_hindi, lang_regional, is_remote]])
        
        # 2. One-Hot Encoding (Business Type, Location)
        # We need to construct a df-like frame or just list of lists for OHE
        # Order must match training: ['Target_Business_Type', 'Location_Area']
        cat_input_df = pd.DataFrame(
            [[user_input.get('Target_Business_Type', 'other').lower(), loc]], 
            columns=['Target_Business_Type', 'Location_Area']
        )
        ohe_features = self.ohe.transform(cat_input_df)
        
        # 3. TF-IDF
        desc = user_input.get('Description', '')
        # Apply same boost factor (10.0)
        tfidf_features = self.tfidf.transform([desc]).toarray() * 10.0
        
        # 4. Combine
        final_vector = np.hstack([manual_features, ohe_features, tfidf_features])
        
        return final_vector
