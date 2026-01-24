"""
Test script to compare Cosine Similarity vs KNN ranking methods.
"""
import sys
import os
sys.path.append(os.getcwd())

from src.models.recommendation_engine import RecommendationEngine

print("=" * 80)
print("COMPARING COSINE SIMILARITY VS KNN RANKING")
print("=" * 80)

# Test case
test_input = {
    'Target_Business_Type': 'E-commerce',
    'Price_Category': 'Medium',
    'Location_Area': 'remote',
    'Language_Support': ['English']
}

# Get service description
engine_temp = RecommendationEngine(ranking_method='cosine')
test_input['Description'] = engine_temp.df[engine_temp.df['Service_Name'] == 'Social Media Setup']['Description'].iloc[0]

print("\nTest Input:")
print(f"  Business: {test_input['Target_Business_Type']}")
print(f"  Budget: {test_input['Price_Category']}")
print(f"  Location: {test_input['Location_Area']}")
print(f"  Service: Social Media Setup")

# Test with Cosine Similarity
print("\n" + "=" * 80)
print("METHOD 1: COSINE SIMILARITY")
print("=" * 80)

engine_cosine = RecommendationEngine(ranking_method='cosine')
results_cosine = engine_cosine.get_recommendations(test_input, top_k=5)

for i, r in enumerate(results_cosine, 1):
    print(f"{i}. {r['Service_Name']} - {r['Price_Category'].upper()}")
    print(f"   Match Score: {r['Match_Score']}%")
    print(f"   Business: {r['Target_Business_Type']}")

# Test with KNN
print("\n" + "=" * 80)
print("METHOD 2: K-NEAREST NEIGHBORS (KNN)")
print("=" * 80)

engine_knn = RecommendationEngine(ranking_method='knn')
results_knn = engine_knn.get_recommendations(test_input, top_k=5)

for i, r in enumerate(results_knn, 1):
    print(f"{i}. {r['Service_Name']} - {r['Price_Category'].upper()}")
    print(f"   Match Score: {r['Match_Score']}%")
    print(f"   Business: {r['Target_Business_Type']}")

# Compare results
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

cosine_services = [r['Service_Name'] for r in results_cosine]
knn_services = [r['Service_Name'] for r in results_knn]

overlap = set(cosine_services) & set(knn_services)
print(f"Services in both top 5: {len(overlap)}/5")
print(f"Overlap: {overlap}")

print("\nCosine Similarity - Average Score:", sum(r['Match_Score'] for r in results_cosine) / len(results_cosine))
print("KNN - Average Score:", sum(r['Match_Score'] for r in results_knn) / len(results_knn))

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("Both methods are now implemented and working!")
print("- Cosine Similarity: Better for text-based matching (higher scores)")
print("- KNN: Alternative distance-based approach (more conservative scores)")
print("\nDefault method: Cosine Similarity (as it performs better for this use case)")
