import math

import numpy as np


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


if __name__ == '__main__':
    nn = NeuralNetwork([784, 100, 100, 10])
    print(f'Activations: {nn.a}'
          f'\nWeights: {nn.W}'
          f'\nBias: {nn.b}'
          )
