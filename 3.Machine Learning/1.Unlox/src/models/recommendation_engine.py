import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from src.models.user_encoder import UserEncoder
from src.models.explanation_generator import ExplanationGenerator

# Paths
# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up: src/models -> src -> project_root
PROJECT_ROOT = os.path.dirname(os.path.dirname(current_dir))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'service_recommendation_data_cleaned.csv')

class RecommendationEngine:
    def __init__(self, ranking_method='cosine'):
        """
        Initialize recommendation engine.
        
        Args:
            ranking_method: 'cosine' for Cosine Similarity or 'knn' for K-Nearest Neighbors
        """
        self.ranking_method = ranking_method
        self.encoder = UserEncoder()
        self.feature_matrix = np.load(os.path.join(PROCESSED_DATA_DIR, 'features.npy'))
        self.service_ids = np.load(os.path.join(PROCESSED_DATA_DIR, 'service_ids.npy'))
        self.df = pd.read_csv(CLEANED_DATA_PATH)
        self.explainer = ExplanationGenerator()
        
    def get_recommendations(self, user_input, top_k=5, strict_filters=True):
        """
        Main function to get recommendations.
        user_input: Dict with user preferences.
        top_k: Number of recommendations to return.
        strict_filters: If True, uses hard filtering (only exact matches).
        """
        
        # 1. HARD FILTERS - Pre-filter candidates to match ALL criteria
        filtered_df = self.df.copy()
        
        # Filter by Business Type (STRICT)
        if 'Target_Business_Type' in user_input and user_input['Target_Business_Type']:
            user_business = user_input['Target_Business_Type'].lower()
            filtered_df = filtered_df[filtered_df['Target_Business_Type'] == user_business]
        
        # Filter by Price (STRICT - at or below budget)
        if 'Price_Category' in user_input and user_input['Price_Category']:
            price_order = {'low': 1, 'medium': 2, 'high': 3, 'premium': 4}
            user_budget = user_input['Price_Category'].lower()
            user_budget_val = price_order.get(user_budget, 2)
            
            # Only include services at or below budget
            filtered_df = filtered_df[
                filtered_df['Price_Category'].map(price_order).fillna(2) <= user_budget_val
            ]
        
        # Filter by Location (STRICT)
        if 'Location_Area' in user_input and user_input['Location_Area']:
            user_location = user_input['Location_Area'].lower()
            filtered_df = filtered_df[filtered_df['Location_Area'] == user_location]
        
        # Check if we have any candidates left
        if len(filtered_df) == 0:
            return []  # No matches found
        
        # Get indices of filtered candidates
        candidate_indices = filtered_df.index.tolist()
        
        # 2. Encode User Input
        user_vector = self.encoder.encode_user_input(user_input)
        
        # 3. Compute Similarity only on filtered candidates
        candidate_matrix = self.feature_matrix[candidate_indices]
        similarities = cosine_similarity(user_vector, candidate_matrix).flatten()
        
        # 4. Rank by similarity using selected method
        if self.ranking_method == 'knn':
            # Use KNN ranking
            temp_knn = NearestNeighbors(
                n_neighbors=min(top_k, len(candidate_indices)),
                metric='euclidean',
                algorithm='auto'
            )
            temp_knn.fit(candidate_matrix)
            
            # Find K nearest neighbors
            distances, local_indices = temp_knn.kneighbors(user_vector)
            
            # Convert to similarity scores and global indices
            scored_candidates = []
            for dist, local_idx in zip(distances[0], local_indices[0]):
                global_idx = candidate_indices[local_idx]
                # Convert distance to similarity: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + dist)
                scored_candidates.append((global_idx, similarity))
        else:
            # Use Cosine Similarity ranking (default)
            scored_candidates = [(candidate_indices[i], similarities[i]) for i in range(len(similarities))]
            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            # Take top K
            scored_candidates = scored_candidates[:top_k]
        
        # 5. Format results
        results = []
        for global_idx, score in scored_candidates:
            original_row = self.df.iloc[global_idx]
            
            # Skip if score is too low (optional threshold)
            if score < 0.1: 
                continue
            
            explanations = self.explainer.generate_explanation(user_input, original_row)
            
            results.append({
                'Service_ID': original_row['Service_ID'],
                'Service_Name': original_row['Service_Name'],
                'Match_Score': round(score * 100, 2),
                'Description': original_row['Description'],
                'Price_Category': original_row['Price_Category'],
                'Target_Business_Type': original_row['Target_Business_Type'],
                'Location': original_row['Location_Area'],
                'Explanations': explanations
            })
            
        return results

if __name__ == "__main__":
    # Simple Test
    engine = RecommendationEngine()
    test_input = {
        'Target_Business_Type': 'Restaurant',
        'Price_Category': 'Low',
        'Language_Support': ['Hindi'],
        'Location_Area': 'Remote',
        'Description': 'accounting and tax help'
    }
    
    print("Testing Recommendation Engine...")
    recs = engine.get_recommendations(test_input, top_k=3)
    for r in recs:
        print(f"[{r['Match_Score']}%] {r['Service_Name']} ({r['Price_Category']})")
        print(f"   -> Reasons: {r['Explanations']}")
