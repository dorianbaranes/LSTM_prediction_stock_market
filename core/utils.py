import time

class Timer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"Task duration: {self.duration} seconds")
