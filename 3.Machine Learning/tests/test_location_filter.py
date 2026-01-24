import sys
import os
sys.path.append(os.getcwd())

from src.models.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()

test_input = {
    'Target_Business_Type': 'E-commerce',
    'Price_Category': 'Low',
    'Description': 'Social media marketing',
    'Location_Area': 'remote',
    'Language_Support': ['English']
}

print("Testing with E-commerce + Low budget + Remote location...")
results = engine.get_recommendations(test_input, top_k=5)

for r in results:
    print(f"{r['Service_Name']} - {r['Target_Business_Type'].upper()} - {r['Location'].upper()} - {r['Price_Category'].upper()} - Score: {r['Match_Score']}%")
