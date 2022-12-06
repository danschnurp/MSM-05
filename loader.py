import pandas as pd


def read_excel(input_file_path="./Cork+Stoppers.xlsx"):
    data = pd.read_excel(input_file_path, "Data", index_col=0)
    data = data.dropna()
    print("-------------------------------input data----------------------------------------")
    print(data)
    return data
