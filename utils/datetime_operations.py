import numpy as np
from datetime import datetime

def datetime64_to_datetime(datetime64):
    timestamp = ((datetime64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)

def get_day_of_year(datetime):
    return datetime.timetuple().tm_yday