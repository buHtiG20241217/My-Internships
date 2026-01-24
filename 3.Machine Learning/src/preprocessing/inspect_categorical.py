import pandas as pd

try:
    df = pd.read_csv(r"E:\Internship\ml-service-recommendation\data\cleaned\service_recommendation_data_cleaned.csv")

    print("\n=== Value Counts ===")
    for col in ['Price_Category', 'Language_Support', 'Location_Area', 'Target_Business_Type']:
        print(f"\n{col}:")
        print(df[col].value_counts())

    print("\n=== Language 'Regional' Check ===")
    regional_df = df[df['Language_Support'] == 'regional']
    print("\nLocations for 'Regional' language:")
    print(regional_df['Location_Area'].value_counts())

    print("\n=== Price 'Premium' vs 'High' Check ===")
    print("\nSample 'Premium' services:")
    print(df[df['Price_Category'] == 'premium'][['Service_Name', 'Price_Category']].head())

except Exception as e:
    print(f"Error: {e}")
