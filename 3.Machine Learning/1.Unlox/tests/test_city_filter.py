import sys
import os
sys.path.append(os.getcwd())

from src.models.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()

test_input = {
    'Target_Business_Type': 'E-commerce',
    'Price_Category': 'Low',
    'Description': 'Social media marketing',
    'Location_Area': 'delhi',  # Specific city
    'Language_Support': ['English']
}

print("Testing with E-commerce + Low budget + Delhi (Specific City)...")
results = engine.get_recommendations(test_input, top_k=5)

for r in results:
    print(f"{r['Service_Name']} - {r['Location'].upper()} - Score: {r['Match_Score']}%")
    
print("\n" + "="*60)
print("Expected: All results should be DELHI only (no REMOTE)")
