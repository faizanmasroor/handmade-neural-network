import csv
import threading

import numpy as np


def load_data(sources: dict[str, str], destinations: dict[str, np.ndarray]):
    threads = [threading.Thread(target=load_csv_to_numpy, args=(label, path, destinations)) for label, path in sources.items()]
    for thread in threads: thread.start()
    for thread in threads: thread.join()


def load_csv_to_numpy(label: str, path: str, destinations: dict[str, np.ndarray]):
    print(f'Loading "{label}" CSV from "{path}"...')
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        data = np.array(list(reader))
        destinations[label] = data
    print(f'Finished loading "{label}" CSV from "{path}"!')
