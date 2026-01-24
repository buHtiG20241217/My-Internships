"""
Comprehensive test suite for recommendation engine.
Tests all possible user scenarios to validate filtering and ranking behavior.
"""
import sys
import os
sys.path.append(os.getcwd())

from src.models.recommendation_engine import RecommendationEngine
import pandas as pd

def test_business_type_filtering():
    """Test that results match the user's selected business type."""
    engine = RecommendationEngine()
    
    test_cases = [
        {'business': 'E-commerce', 'service': 'Advanced Tax Filing'},
        {'business': 'Restaurant', 'service': 'Payroll Processing'},
        {'business': 'Tech Startup', 'service': 'Social Media Setup'},
        {'business': 'Retail', 'service': 'Basic Accounting'},
        {'business': 'Freelancer', 'service': 'Contract Review'},
    ]
    
    print("=" * 80)
    print("TEST 1: Business Type Filtering")
    print("=" * 80)
    
    for case in test_cases:
        user_input = {
            'Target_Business_Type': case['business'],
            'Price_Category': 'Medium',
            'Description': engine.df[engine.df['Service_Name'] == case['service']]['Description'].iloc[0],
            'Location_Area': 'remote',
            'Language_Support': ['English']
        }
        
        results = engine.get_recommendations(user_input, top_k=5)
        
        print(f"\n{case['business']} + {case['service']}:")
        
        # Count how many results match the business type
        matching = sum(1 for r in results if r['Target_Business_Type'].lower() == case['business'].lower())
        non_matching = len(results) - matching
        
        print(f"  ✓ Matching business type: {matching}/{len(results)}")
        print(f"  ✗ Non-matching business type: {non_matching}/{len(results)}")
        
        # Show top 3 results
        for i, r in enumerate(results[:3]):
            match_indicator = "✓" if r['Target_Business_Type'].lower() == case['business'].lower() else "✗"
            print(f"    {i+1}. {match_indicator} {r['Service_Name']} - {r['Target_Business_Type']} - {r['Match_Score']}%")
        
        if non_matching > 0:
            print(f"  ⚠️  WARNING: Found {non_matching} results with wrong business type!")

def test_price_filtering():
    """Test that expensive services are penalized correctly."""
    engine = RecommendationEngine()
    
    budgets = ['Low', 'Medium', 'High', 'Premium']
    
    print("\n" + "=" * 80)
    print("TEST 2: Price Filtering")
    print("=" * 80)
    
    for budget in budgets:
        user_input = {
            'Target_Business_Type': 'E-commerce',
            'Price_Category': budget,
            'Description': engine.df[engine.df['Service_Name'] == 'Social Media Setup']['Description'].iloc[0],
            'Location_Area': 'remote',
            'Language_Support': ['English']
        }
        
        results = engine.get_recommendations(user_input, top_k=5)
        
        print(f"\nBudget: {budget}")
        
        price_order = {'low': 1, 'medium': 2, 'high': 3, 'premium': 4}
        budget_val = price_order[budget.lower()]
        
        within_budget = sum(1 for r in results if price_order.get(r['Price_Category'], 2) <= budget_val)
        above_budget = len(results) - within_budget
        
        print(f"  ✓ Within/below budget: {within_budget}/{len(results)}")
        print(f"  ✗ Above budget: {above_budget}/{len(results)}")
        
        # Show prices
        for i, r in enumerate(results[:3]):
            price_val = price_order.get(r['Price_Category'], 2)
            match_indicator = "✓" if price_val <= budget_val else "✗"
            print(f"    {i+1}. {match_indicator} {r['Price_Category'].upper()} - Score: {r['Match_Score']}%")

def test_location_filtering():
    """Test that location filtering works correctly."""
    engine = RecommendationEngine()
    
    locations = ['remote', 'delhi', 'mumbai', 'bengaluru']
    
    print("\n" + "=" * 80)
    print("TEST 3: Location Filtering")
    print("=" * 80)
    
    for location in locations:
        user_input = {
            'Target_Business_Type': 'E-commerce',
            'Price_Category': 'Medium',
            'Description': engine.df[engine.df['Service_Name'] == 'Digital Marketing Strategy']['Description'].iloc[0],
            'Location_Area': location,
            'Language_Support': ['English']
        }
        
        results = engine.get_recommendations(user_input, top_k=5)
        
        print(f"\nLocation: {location.upper()}")
        
        matching_location = sum(1 for r in results if r['Location'].lower() == location)
        non_matching = len(results) - matching_location
        
        print(f"  ✓ Matching location: {matching_location}/{len(results)}")
        print(f"  ✗ Non-matching location: {non_matching}/{len(results)}")
        
        # Show locations
        for i, r in enumerate(results[:3]):
            match_indicator = "✓" if r['Location'].lower() == location else "✗"
            print(f"    {i+1}. {match_indicator} {r['Location'].upper()} - Score: {r['Match_Score']}%")
        
        if non_matching > 0:
            print(f"  ⚠️  WARNING: Found {non_matching} results with wrong location!")

def test_combined_filtering():
    """Test that all filters work together correctly."""
    engine = RecommendationEngine()
    
    print("\n" + "=" * 80)
    print("TEST 4: Combined Filtering (Business + Price + Location)")
    print("=" * 80)
    
    test_case = {
        'Target_Business_Type': 'E-commerce',
        'Price_Category': 'Low',
        'Description': engine.df[engine.df['Service_Name'] == 'Payroll Processing']['Description'].iloc[0],
        'Location_Area': 'delhi',
        'Language_Support': ['English']
    }
    
    results = engine.get_recommendations(test_case, top_k=5)
    
    print(f"\nTest: E-commerce + Low Budget + Delhi + Payroll Processing")
    print(f"Total results: {len(results)}\n")
    
    price_order = {'low': 1, 'medium': 2, 'high': 3, 'premium': 4}
    
    perfect_matches = 0
    for i, r in enumerate(results):
        business_match = r['Target_Business_Type'].lower() == 'e-commerce'
        price_match = price_order.get(r['Price_Category'], 2) <= 1  # Low or below
        location_match = r['Location'].lower() == 'delhi'
        
        all_match = business_match and price_match and location_match
        if all_match:
            perfect_matches += 1
        
        print(f"{i+1}. {r['Service_Name']}")
        print(f"   Business: {'✓' if business_match else '✗'} {r['Target_Business_Type']}")
        print(f"   Price: {'✓' if price_match else '✗'} {r['Price_Category'].upper()}")
        print(f"   Location: {'✓' if location_match else '✗'} {r['Location'].upper()}")
        print(f"   Score: {r['Match_Score']}%")
        print()
    
    print(f"Perfect matches (all 3 criteria): {perfect_matches}/{len(results)}")
    if perfect_matches < len(results):
        print(f"⚠️  WARNING: {len(results) - perfect_matches} results don't match all criteria!")

def test_service_relevance():
    """Test that the selected service appears in top results."""
    engine = RecommendationEngine()
    
    print("\n" + "=" * 80)
    print("TEST 5: Service Relevance")
    print("=" * 80)
    
    services = ['Payroll Processing', 'Social Media Setup', 'Advanced Tax Filing', 'Basic Accounting']
    
    for service in services:
        user_input = {
            'Target_Business_Type': 'E-commerce',
            'Price_Category': 'Medium',
            'Description': engine.df[engine.df['Service_Name'] == service]['Description'].iloc[0],
            'Location_Area': 'remote',
            'Language_Support': ['English']
        }
        
        results = engine.get_recommendations(user_input, top_k=5)
        
        print(f"\nSearching for: {service}")
        
        # Check if the service appears in results
        found_positions = [i+1 for i, r in enumerate(results) if r['Service_Name'] == service]
        
        if found_positions:
            print(f"  ✓ Found at positions: {found_positions}")
            print(f"  Top result: {results[0]['Service_Name']} ({results[0]['Match_Score']}%)")
        else:
            print(f"  ✗ NOT FOUND in top 5 results!")
            print(f"  Top result: {results[0]['Service_Name']} ({results[0]['Match_Score']}%)")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RECOMMENDATION ENGINE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    test_business_type_filtering()
    test_price_filtering()
    test_location_filtering()
    test_combined_filtering()
    test_service_relevance()
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
