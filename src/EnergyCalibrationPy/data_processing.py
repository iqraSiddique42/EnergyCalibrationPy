import pandas as pd

def load_data(file_path, column_names=None, skip_rows=4):
    """
    Loads the histogram data from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
        column_names (list or None): Names of the columns. Default is ['area', 'counts'].
        skip_rows (int): Number of rows to skip in the file header. Default is 4.
    
    Returns:
        pd.DataFrame: Processed data with columns 'x' and 'counts'.
    """
    column_names = column_names or ['area', 'counts']  # Default column names
    try:
        df = pd.read_csv(file_path, skiprows=skip_rows)
        df.columns = column_names
        return df
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {e}")

