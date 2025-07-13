import csv
import threading

import numpy as np

from matrix_math import one_hot_encode


def load_data(sources: dict[str, str], data: dict[str, list[tuple[np.ndarray, np.ndarray]]]):
    threads = [
        threading.Thread(target=load_csv_to_numpy, args=(split, path, data))
        for split, path in sources.items()
    ]
    for thread in threads: thread.start()
    for thread in threads: thread.join()


def load_csv_to_numpy(split: str, path: str, data: dict[str, list[tuple[np.ndarray, np.ndarray]]]):
    print(f'Loading "{split}" CSV from "{path}"...')
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        raw_data = np.array(list(reader))
    print(f'Finished loading "{split}" CSV from "{path}"!')

    print(f'Preprocessing "{split}" data...')
    data[split] = [
        (one_hot_encode(int(raw_vector[0])), raw_vector[1:].reshape(-1, 1).astype(np.float64))
        for raw_vector in raw_data
    ]
    print(f'Finished preprocessing "{split}" data!')
