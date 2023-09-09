import pandas as pd

def preprocess_df(path_in, path_out):
    """
    Function for processing dataset. Deleting column 'Unnamed: 0' and drop duplicates by 'description'
        Args:
            path_in: read dataset from path
            path_out: write cleaned dataset to path
    """
    df = pd.read_csv(path_in)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.drop_duplicates(subset='description', keep="last", inplace=True)
    df.reset_index(inplace=True)
    df.to_csv(path_out, index=False)
