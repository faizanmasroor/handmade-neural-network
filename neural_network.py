import math

import numpy as np


def relu(x): return np.maximum(0, x)


def sigmoid(x): return 1 / (1 + np.exp(-x))


def init_matrices(layer_sizes: list[int]) -> list[list[None | np.ndarray]]:
    """
    Returns activation and biases matrices with all elements initialized to 0 and weights matrices initialized using
    Kaiming He initialization. weights[0] and biases[0] are None to align with conventional neural network notation.
    E.g, in forward propagation, the following statement may be used: "a[1] = activation_function(W[1] * a[0] + b[1])"
    """
    activations = [np.zeros((layer_size, 1)) for layer_size in layer_sizes]
    weights = [None] + [
        np.random.normal(loc=0, scale=math.sqrt(2 / input_size), size=(output_size, input_size))  # He initialization
        # For layer_sizes=[5, 3, 2], zip() returns iterable [(3, 5), (2, 3)]
        for output_size, input_size in zip(layer_sizes[1:], layer_sizes[:-1])
    ]
    biases = [None] + [np.zeros((layer_size, 1)) for layer_size in layer_sizes[1:]]

    return [activations, weights, biases]


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int]):
        self.a, self.W, self.b = init_matrices(layer_sizes)

    def __str__(self):
        return '\nActivations:\n%s\n\nWeights:\n%s\n\nBiases:\n%s\n' % (
            '\n\n'.join([np.array2string(matrix) for matrix in self.a]),
            '\n\n'.join([np.array2string(matrix) for matrix in self.W[1:]]),
            '\n\n'.join([np.array2string(matrix) for matrix in self.b[1:]]))

    def forward_propagate(self, input_array: np.ndarray) -> np.ndarray:
        self.a[0] = input_array
        for i in range(1, len(self.a)):
            self.a[i] = relu(self.W[i] @ self.a[i - 1] + self.b[i])

        return self.a[-1]


if __name__ == '__main__':
    nn = NeuralNetwork([5, 3, 2])
    print(nn)
    nn.forward_propagate(np.array([1, 2, 3, 4, 5]).reshape(-1, 1))
    print(nn)
