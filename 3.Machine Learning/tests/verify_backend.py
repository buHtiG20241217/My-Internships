import sys
import os
import pandas as pd

# Add project root
sys.path.append(os.getcwd())

try:
    from src.models.recommendation_engine import RecommendationEngine

    print("Loading Backend Engine...")
    engine = RecommendationEngine()
    
    # Test Query
    query = {
        'Target_Business_Type': 'E-commerce', 
        'Price_Category': 'High', 
        'Description': 'Social media marketing', 
        'Location_Area': 'Remote', 
        'Language_Support': ['English']
    }
    
    print(f"\nTesting Query: {query['Description']}")
    results = engine.get_recommendations(query, top_k=3)
    
    if not results:
        print("FAILURE: No results returned.")
        sys.exit(1)
        
    print("\nTop Results:")
    for r in results:
        print(f"- {r['Service_Name']} (Score: {r['Match_Score']}%)")
        
    # Validation
    top_name = results[0]['Service_Name']
    if top_name == "Social Media Setup":
        print("\nSUCCESS: Backend logic is verified (Text Boost Active).")
    else:
        print(f"\nWARNING: Unexpected top result '{top_name}'. Check weights.")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    sys.exit(1)
