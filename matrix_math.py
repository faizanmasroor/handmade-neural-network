import numpy as np


def relu(x: np.ndarray) -> np.ndarray: return np.maximum(0, x)


def deriv_relu(x: np.ndarray) -> np.ndarray: return np.where(x > 0, 1, 0)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))  # Reduces chance of encountering OverflowError; does not affect return value
    return e_x / np.sum(e_x)


def categorical_cross_entropy_loss(true_label: np.ndarray, predicted_label: np.ndarray) -> float:
    return -np.sum(true_label * np.log(predicted_label + 1e-9))  # Epsilon (1e-9) is added to avoid computing log(0)


def one_hot_encode(val: int, num_categories: int = 10) -> np.ndarray:
    if val >= num_categories: raise ValueError(f'The value {val} is out of the range {num_categories}')

    result = np.zeros((num_categories, 1))
    result[val] = 1

    return result


def one_hot_decode(vector: np.ndarray) -> int: return int(np.argmax(vector))
