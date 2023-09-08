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
