import pandas as pd
from utils import ExecutionTime, write_list


def extract_description(path_to_df: str, column: str, is_saved=True) -> list[str]:
    """Extracts 'description' column from .csv and returns as a list
    Args:
        path_to_df: Path to .csv file where we get the description from
        column: Column from which we take the description
    Return:
        A list of wine descriptions with one or some sentences. For example:
        ['This wine super cool. Drink it every day.',
        'That has fruity flavour.']
    """
    print(f'Reading description from {path_to_df}...')
    t = ExecutionTime()
    t.start()
    description = pd.read_csv(path_to_df, usecols=[column])
    t.end()
    print(f'Reading of description completed in {t.get_exec_time():.2f} sec')
    if is_saved:
        path = './data/description'
        write_list(path, description[column].tolist())
        print(f'Description saved to {path}')
    return description[column].tolist()
