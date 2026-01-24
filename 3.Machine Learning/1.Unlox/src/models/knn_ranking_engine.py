"""
KNN-based Ranking Engine
Implements K-Nearest Neighbors algorithm for service recommendations.
Alternative to Cosine Similarity-based ranking.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os

# Paths
PROCESSED_DATA_DIR = r"E:\Internship\ml-service-recommendation\data\processed"
CLEANED_DATA_PATH = r"E:\Internship\ml-service-recommendation\data\cleaned\service_recommendation_data_cleaned.csv"

class KNNRankingEngine:
    """
    K-Nearest Neighbors based ranking engine for service recommendations.
    Uses Euclidean distance to find the K most similar services.
    """
    
    def __init__(self, n_neighbors=5, metric='euclidean'):
        """
        Initialize KNN ranking engine.
        
        Args:
            n_neighbors: Number of neighbors to retrieve (default: 5)
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn_model = None
        self.feature_matrix = None
        self.df = None
        
        # Load data
        self._load_data()
        
        # Train KNN model
        self._train_knn()
    
    def _load_data(self):
        """Load feature matrix and service data."""
        self.feature_matrix = np.load(os.path.join(PROCESSED_DATA_DIR, 'features.npy'))
        self.df = pd.read_csv(CLEANED_DATA_PATH)
        
        print(f"Loaded {len(self.df)} services with {self.feature_matrix.shape[1]} features")
    
    def _train_knn(self):
        """Train KNN model on the feature matrix."""
        self.knn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm='auto'  # Let sklearn choose the best algorithm
        )
        
        self.knn_model.fit(self.feature_matrix)
        print(f"KNN model trained with {self.n_neighbors} neighbors using {self.metric} metric")
    
    def get_knn_recommendations(self, user_vector, filtered_indices, top_k=5):
        """
        Get top K recommendations using KNN.
        
        Args:
            user_vector: User feature vector (1D array)
            filtered_indices: List of indices after hard filtering
            top_k: Number of recommendations to return
            
        Returns:
            List of tuples (index, distance) sorted by distance (ascending)
        """
        if len(filtered_indices) == 0:
            return []
        
        # Get feature matrix for filtered candidates
        candidate_matrix = self.feature_matrix[filtered_indices]
        
        # Reshape user vector for KNN
        user_vector_reshaped = user_vector.reshape(1, -1)
        
        # Create a temporary KNN model for the filtered candidates
        temp_knn = NearestNeighbors(
            n_neighbors=min(top_k, len(filtered_indices)),
            metric=self.metric,
            algorithm='auto'
        )
        temp_knn.fit(candidate_matrix)
        
        # Find K nearest neighbors
        # distances: shape (1, k), indices: shape (1, k)
        distances, local_indices = temp_knn.kneighbors(user_vector_reshaped)
        
        # Convert local indices to global indices and calculate similarity scores
        results = []
        for dist, local_idx in zip(distances[0], local_indices[0]):
            global_idx = filtered_indices[local_idx]
            
            # Convert distance to similarity score (0-1 scale, higher is better)
            # For Euclidean distance, we use: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + dist)
            
            results.append((global_idx, similarity))
        
        return results
    
    def compare_with_cosine(self, user_vector, filtered_indices, top_k=5):
        """
        Compare KNN results with Cosine Similarity results.
        
        Args:
            user_vector: User feature vector
            filtered_indices: List of indices after hard filtering
            top_k: Number of recommendations
            
        Returns:
            Dictionary with both KNN and Cosine results
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # KNN results
        knn_results = self.get_knn_recommendations(user_vector, filtered_indices, top_k)
        
        # Cosine Similarity results
        candidate_matrix = self.feature_matrix[filtered_indices]
        user_vector_reshaped = user_vector.reshape(1, -1)
        cosine_scores = cosine_similarity(user_vector_reshaped, candidate_matrix).flatten()
        
        cosine_results = []
        for local_idx, score in enumerate(cosine_scores):
            global_idx = filtered_indices[local_idx]
            cosine_results.append((global_idx, score))
        
        # Sort cosine results by score descending
        cosine_results.sort(key=lambda x: x[1], reverse=True)
        cosine_results = cosine_results[:top_k]
        
        return {
            'knn': knn_results,
            'cosine': cosine_results
        }

if __name__ == "__main__":
    # Test KNN ranking engine
    from src.models.user_encoder import UserEncoder
    
    print("\n" + "=" * 80)
    print("TESTING KNN RANKING ENGINE")
    print("=" * 80)
    
    # Initialize engines
    knn_engine = KNNRankingEngine(n_neighbors=10, metric='euclidean')
    encoder = UserEncoder()
    
    # Test case
    test_input = {
        'Target_Business_Type': 'E-commerce',
        'Price_Category': 'Medium',
        'Description': knn_engine.df[knn_engine.df['Service_Name'] == 'Social Media Setup']['Description'].iloc[0],
        'Location_Area': 'remote',
        'Language_Support': ['English']
    }
    
    print("\nTest Input:")
    print(f"  Business: {test_input['Target_Business_Type']}")
    print(f"  Budget: {test_input['Price_Category']}")
    print(f"  Location: {test_input['Location_Area']}")
    print(f"  Service: Social Media Setup")
    
    # Encode user input
    user_vector = encoder.encode_user_input(test_input)
    
    # Apply hard filters
    filtered_df = knn_engine.df.copy()
    filtered_df = filtered_df[filtered_df['Target_Business_Type'] == 'e-commerce']
    filtered_df = filtered_df[filtered_df['Location_Area'] == 'remote']
    
    price_order = {'low': 1, 'medium': 2, 'high': 3, 'premium': 4}
    filtered_df = filtered_df[
        filtered_df['Price_Category'].map(price_order).fillna(2) <= 2  # Medium or below
    ]
    
    filtered_indices = filtered_df.index.tolist()
    
    print(f"\nFiltered candidates: {len(filtered_indices)}")
    
    # Compare KNN vs Cosine
    comparison = knn_engine.compare_with_cosine(user_vector, filtered_indices, top_k=5)
    
    print("\n" + "=" * 80)
    print("KNN RESULTS (Euclidean Distance)")
    print("=" * 80)
    for i, (idx, score) in enumerate(comparison['knn'], 1):
        service = knn_engine.df.iloc[idx]
        print(f"{i}. {service['Service_Name']} - {service['Price_Category'].upper()}")
        print(f"   Similarity: {score*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY RESULTS")
    print("=" * 80)
    for i, (idx, score) in enumerate(comparison['cosine'], 1):
        service = knn_engine.df.iloc[idx]
        print(f"{i}. {service['Service_Name']} - {service['Price_Category'].upper()}")
        print(f"   Similarity: {score*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Check overlap in top 5
    knn_top_services = [knn_engine.df.iloc[idx]['Service_Name'] for idx, _ in comparison['knn']]
    cosine_top_services = [knn_engine.df.iloc[idx]['Service_Name'] for idx, _ in comparison['cosine']]
    
    overlap = set(knn_top_services) & set(cosine_top_services)
    print(f"Services in both top 5: {len(overlap)}/5")
    print(f"Overlap: {overlap}")
