import pandas as pd
import glob
import os
from typing import Dict

def load_raw_data_dic(folder_path: str) -> Dict[str, pd.DataFrame]:
    '''
    Load CSV files from a specified folder into a dictionary of DataFrames.
    '''
    
    # Use glob to get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    # Initialize an empty dictionary to store DataFrames
    dataframes_dic = {}
    
    # Loop through the list of CSV files and read each one into a DataFrame
    for file in csv_files:
        # Extract the file name without the folder path and extension
        file_name = os.path.basename(file).replace('.csv', '')
    
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
    
        # Store the DataFrame in the dictionary
        dataframes_dic[file_name] = df
    
    # Optionally, display the keys of the dictionary to see the loaded DataFrames
    print(dataframes_dic.keys())

    return dataframes_dic
