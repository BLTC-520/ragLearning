import time
import csv
from pathlib import Path

def measure_inference_time(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, round(end - start, 4)

def log_metrics(model_name: str, query: str, answer: str, time_taken: float, filepath: str = "results.csv"):
    file = Path(filepath)
    new_file = not file.exists()
    
    with open(file, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Model", "Query", "Answer", "TimeTaken"])
        writer.writerow([model_name, query, answer, time_taken])