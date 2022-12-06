import pandas as pd


def read_excel(input_file_path="./input_data/Cork+Stoppers.xlsx", sheet_name="Data"):
    """
    This function reads in an Excel file and returns a dataframe.

    :param sheet_name: sheet name in Excel file
    :param input_file_path: The path to the input file, defaults to ./input_data/Cork+Stoppers.xlsx (optional)
    """
    data = pd.read_excel(input_file_path, sheet_name, index_col=0)
    print("-------------------------------input data----------------------------------------")
    print(data)
    return data
