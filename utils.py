import time
import pickle


class ExecutionTime:

    """
    Class with functions for measuring execution time.
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.exec_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        self.exec_time = self.end_time - self.start_time

    def get_exec_time(self):
        return self.exec_time


def write_list(save_to_file: str, var):
    """
    Util for saving file from variable to file.
        Args:
            save_to_file: Path to saving file
            var: variable for saving
        Example:
            write_list('./data/description', list_of_description)
    """
    with open(save_to_file, 'wb') as fp:
        pickle.dump(var, fp)


def read_list(load_file: str):
    """
        Util for loading file to variable.
            Args:
                load_file: Path to load file

            Example:
                list_of_description = read_list('./data/description')
    """
    with open(load_file, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
