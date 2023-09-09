import time


class ExecutionTime:
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


import pickle
def write_list(save_to_file: str, var):
    with open(save_to_file, 'wb') as fp:
        pickle.dump(var, fp)

# Read list to memory
def read_list(load_file: str):
    with open(load_file, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list