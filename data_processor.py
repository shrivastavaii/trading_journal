# data_processor.py
import pandas as pd
from clean_data import clean_and_process_data  # We'll modify clean_data.py to expose this function

def clean_and_save_to_mysql(file_path):
    """
    Process uploaded CSV file and save to MySQL
    Returns cleaned DataFrame for display
    """
    try:
        # Read the uploaded file
        df = pd.read_csv(file_path)
        
        # Process the data using the cleaning logic from clean_data.py
        df_cleaned = clean_and_process_data(df)
        
        return df_cleaned
        
    except Exception as e:
        raise Exception(f"Data processing error: {str(e)}")