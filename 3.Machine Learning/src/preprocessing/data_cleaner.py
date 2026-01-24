import pandas as pd
import os

"""
Data Cleaner Module
------------------
Handles the loading, cleaning, and standardization of raw service recommendation data.
"""

RAW_DATA_PATH = r"E:\Internship\ml-service-recommendation\data\raw\service_recommendation_data.csv"
CLEANED_DATA_PATH = r"E:\Internship\ml-service-recommendation\data\cleaned\service_recommendation_data_cleaned.csv"

def load_data(path):
    """
    Load raw data from a CSV file.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found at {path}")
    return pd.read_csv(path)

def clean_text(text):
    """
    Standardize text entries by stripping whitespace and converting to lowercase.

    Args:
        text (str or any): The input text to clean.

    Returns:
        str: Cleaned text if input is a string.
        any: Original input if it is not a string.
    """
    if isinstance(text, str):
        return text.strip().lower()
    return text

def clean_dataset(df):
    """
    Apply cleaning transformations to the dataset.
    
    Operations:
    1. Standardizes categorical columns (lowercase, stripped).
    2. Strips whitespace from descriptions (preserving case).
    3. Removes duplicate rows.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for processing.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df_clean = df.copy()

    # Standardize text columns
    text_cols = ['Target_Business_Type', 'Price_Category', 'Language_Support', 'Location_Area', 'Match_Quality']
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_text)
            
    # For Description, we might want to keep case for display but create a processed version later.
    # For now, let's just strip whitespace.
    if 'Description' in df_clean.columns:
        df_clean['Description'] = df_clean['Description'].str.strip()

    # Ensure no duplicates
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < before_dedup:
        print(f"Removed {before_dedup - len(df_clean)} duplicate rows.")

    return df_clean

def save_data(df, path):
    """
    Save the cleaned dataframe to a CSV file.
    Creates parent directories if they don't exist.

    Args:
        df (pd.DataFrame): Dataframe to save.
        path (str): Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to {path}")

def main():
    try:
        print("Loading raw data...")
        df = load_data(RAW_DATA_PATH)
        
        print("Cleaning data...")
        df_clean = clean_dataset(df)
        
        print("Saving cleaned data...")
        save_data(df_clean, CLEANED_DATA_PATH)
        
        print("Data cleaning completed successfully!")
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")

if __name__ == "__main__":
    main()
