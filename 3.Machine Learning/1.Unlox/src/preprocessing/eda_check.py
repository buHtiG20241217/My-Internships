import pandas as pd
import os

DATA_PATH = r"E:\Internship\ml-service-recommendation\data\raw\service_recommendation_data.csv"

def analyze_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        print("Data Loaded Successfully!")
        print("-" * 30)
        
        print(f"Shape: {df.shape}")
        print("-" * 30)
        
        print("Columns:")
        print(df.columns.tolist())
        print("-" * 30)
        
        print("Missing Values:")
        print(df.isnull().sum())
        print("-" * 30)
        
        print("Data Types:")
        print(df.dtypes)
        print("-" * 30)
        
        print("First 3 Rows:")
        print(df.head(3))
        print("-" * 30)

        # Check for unique values in potential categorical columns
        print("Unique Values in Categorical Columns (approx):")
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 20:
                print(f"{col}: {df[col].unique()}")

    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    analyze_data()
