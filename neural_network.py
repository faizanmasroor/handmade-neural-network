import math

import numpy as np

np.set_printoptions(suppress=True)


def relu(x): return np.maximum(0, x)


def sigmoid(x): return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Removes the max value of the input vector, has no effect on final result
    return e_x / np.sum(e_x)


def categorical_cross_entropy_loss(true: np.ndarray, pred: np.ndarray) -> float:
    return -np.sum(true * np.log(pred + 1e-9))  # Epsilon (1e-9) is added to avoid np.log(0)


def one_hot_encode(val: int, num_categories: int) -> np.ndarray:
    if val >= num_categories: raise ValueError("The value {} is out of the range {}".format(val, num_categories))

    result = np.zeros((num_categories, 1))
    result[val] = 1

    return result


def one_hot_decode(vector: np.ndarray) -> int: return int(np.argmax(vector))


class NeuralNetwork:
    def __init__(self, activation_function: callable, layer_sizes: list[int]):
        self.activation_function = activation_function
        self._init_matrices(layer_sizes)

    def _init_matrices(self, layer_sizes: list[int]):
        """
        Returns activation and biases matrices with all elements initialized to 0 and weights matrices initialized using
        Kaiming He initialization. weights[0] and biases[0] are None to align with conventional neural network notation.
        E.g, in forward propagation, the following statement may be used:
        "a[1] = activation_function(W[1] * a[0] + b[1])"
        """
        self.a = [np.zeros((layer_size, 1)) for layer_size in layer_sizes]
        self.z = [None] + [np.zeros((layer_size, 1)) for layer_size in layer_sizes[1:]]
        self.W = NeuralNetwork._he_init_weights(
            layer_sizes) if self.activation_function == relu else NeuralNetwork._glorot_init_weights(layer_sizes)
        self.b = [None] + [np.zeros((layer_size, 1)) for layer_size in layer_sizes[1:]]

    @staticmethod
    def _he_init_weights(layer_sizes: list[int]) -> list[None | np.ndarray]:
        return [None] + [
            np.random.normal(loc=0, scale=math.sqrt(2 / input_size), size=(output_size, input_size))
            # For layer_sizes=[5, 3, 2], zip() returns iterable [(3, 5), (2, 3)]
            for output_size, input_size in zip(layer_sizes[1:], layer_sizes[:-1])
        ]

    @staticmethod
    def _glorot_init_weights(layer_sizes: list[int], gaussian=False) -> list[None | np.ndarray]:
        return [None] + [
            np.random.normal(loc=0, scale=math.sqrt(2 / (in_len + out_len)), size=(out_len, in_len)) if gaussian
            else np.random.uniform(low=-math.sqrt(6 / (in_len + out_len)), high=math.sqrt(6 / (in_len + out_len)),
                                   size=(out_len, in_len))
            # For layer_sizes=[5, 3, 2], zip() returns iterable [(3, 5), (2, 3)]
            for out_len, in_len in zip(layer_sizes[1:], layer_sizes[:-1])
        ]

    def __str__(self):
        return '\nActivations (a):\n%s\n\nRaw Activations (z):\n%s\n\nWeights (W):\n%s\n\nBiases (b):\n%s\n' % (
            '\n\n'.join([np.array2string(matrix) for matrix in self.a]),
            '\n\n'.join([np.array2string(matrix) for matrix in self.z[1:]]),
            '\n\n'.join([np.array2string(matrix) for matrix in self.W[1:]]),
            '\n\n'.join([np.array2string(matrix) for matrix in self.b[1:]])
        )

    def forward_propagate(self, input_array: np.ndarray) -> np.ndarray:
        self.a[0] = input_array
        for i in range(1, len(self.a)):
            self.z[i] = self.W[i] @ self.a[i - 1] + self.b[i]  # Don't inline this! It's needed for backpropagation
            self.a[i] = self.activation_function(self.z[i])
        self.a[-1] = softmax(self.z[-1])  # Overrides final activation function (sigmoid/tanh/relu) and applies softmax

        return self.a[-1]


if __name__ == '__main__':
    label = one_hot_encode(np.random.randint(0, 9), 10)

    nn = NeuralNetwork(sigmoid, [784, 100, 100, 10])
    input_vector = np.random.randint(0, 255, (784, 1))
    output_vector = nn.forward_propagate(input_vector)

    print(
        f'Label ({one_hot_decode(label)}):\n{label}\n\nOutput:\n{output_vector}\n\nLoss: {categorical_cross_entropy_loss(label, output_vector)}')
