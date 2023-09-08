import pandas as pd
from utils import ExecutionTime


def extract_description(path_to_df: str, column: str):
    print(f'Reading from {path_to_df}...')
    t = ExecutionTime()
    t.start()
    description = pd.read_csv(path_to_df, usecols=[column])
    t.end()
    print(f'Reading completed in {t.get_exec_time():.2f} sec')
    return description[column].tolist()
