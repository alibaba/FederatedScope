import os
import time
import datetime

def check_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File path {file_path} doesn't exist.")
    else:
        return True


def list_and_filter_dir(path):
    files = [_ for _ in os.listdir(path) if not _.startswith(".")]
    return files


def get_time_interval(start_time, end_time, format="%Y-%m-%d %H:%M:%S"):
    year = str(datetime.datetime.today().year)
    if isinstance(start_time, str):
        timestamp, millisecond = start_time.split(".")
        start_time = time.mktime(time.strptime(f"{year}-"+timestamp, format)) + int(millisecond) / 1000.
    if isinstance(end_time, str):
        timestamp, millisecond = end_time.split(".")
        end_time = time.mktime(time.strptime(f"{year}-"+timestamp, format)) + int(millisecond) / 1000.

    return end_time - start_time

