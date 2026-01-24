"""
Test to find which specific user input combinations return zero recommendations.
"""
import sys
import os
sys.path.append(os.getcwd())

from src.models.recommendation_engine import RecommendationEngine
import pandas as pd

engine = RecommendationEngine()

# Get all unique values
business_types = ['E-commerce', 'Restaurant', 'Tech Startup', 'Retail', 'Freelancer', 'Clinic']
budgets = ['Low', 'Medium', 'High', 'Premium']
locations = ['remote', 'delhi', 'mumbai', 'bengaluru', 'chennai']
services = engine.df['Service_Name'].unique()

print("=" * 80)
print("TESTING FOR ZERO-RESULT SCENARIOS")
print("=" * 80)

zero_result_cases = []

# Test a sample of combinations
for business in business_types:
    for budget in budgets:
        for location in locations:
            for service in services[:3]:  # Test first 3 services to save time
                # Get service description
                service_desc = engine.df[engine.df['Service_Name'] == service]['Description'].iloc[0]
                
                user_input = {
                    'Target_Business_Type': business,
                    'Price_Category': budget,
                    'Description': service_desc,
                    'Location_Area': location,
                    'Language_Support': ['English']
                }
                
                results = engine.get_recommendations(user_input, top_k=5)
                
                if len(results) == 0:
                    zero_result_cases.append({
                        'business': business,
                        'budget': budget,
                        'location': location,
                        'service': service
                    })

print(f"\nFound {len(zero_result_cases)} combinations that return ZERO results:\n")

if zero_result_cases:
    for i, case in enumerate(zero_result_cases[:10], 1):  # Show first 10
        print(f"{i}. {case['business']} + {case['budget']} + {case['location']} + {case['service']}")
    
    if len(zero_result_cases) > 10:
        print(f"\n... and {len(zero_result_cases) - 10} more combinations")
else:
    print("✅ No zero-result cases found in the tested combinations!")

# Test specific edge cases
print("\n" + "=" * 80)
print("TESTING SPECIFIC EDGE CASES")
print("=" * 80)

edge_cases = [
    {
        'name': 'Low budget + Premium service in small city',
        'input': {
            'Target_Business_Type': 'E-commerce',
            'Price_Category': 'Low',
            'Description': engine.df[engine.df['Service_Name'] == 'Financial Audit']['Description'].iloc[0],
            'Location_Area': 'bengaluru',
            'Language_Support': ['English']
        }
    },
    {
        'name': 'Rare business type + specific city',
        'input': {
            'Target_Business_Type': 'Clinic',
            'Price_Category': 'Medium',
            'Description': engine.df[engine.df['Service_Name'] == 'SEO Optimization']['Description'].iloc[0],
            'Location_Area': 'chennai',
            'Language_Support': ['English']
        }
    },
]

for case in edge_cases:
    results = engine.get_recommendations(case['input'], top_k=5)
    status = "✅ FOUND" if len(results) > 0 else "❌ ZERO"
    print(f"\n{case['name']}: {status} ({len(results)} results)")
    if len(results) > 0:
        print(f"  Top result: {results[0]['Service_Name']} - {results[0]['Match_Score']}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"Total zero-result combinations found: {len(zero_result_cases)}")
print("\nThis happens when:")
print("1. The selected service doesn't exist for that business type")
print("2. The service exists but not at or below the selected budget")
print("3. The service exists but not in the selected location")
print("4. Combination of all three filters eliminates all options")
