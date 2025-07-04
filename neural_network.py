import math

import numpy as np

from data import load_data

np.set_printoptions(threshold=12, edgeitems=6, linewidth=200, suppress=True)


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


class NeuralNetwork:
    def __init__(self, activation_function: callable, layer_sizes: list[int]):
        self.activation_function = activation_function
        self.__init_matrices(layer_sizes)

    def __str__(self):
        a, z, W, b = self.a, self.z[1:], self.W[1:], self.b[1:]
        headers = ("Activation Matrices", "Raw Activation Matrices", "Weight Matrices", "Bias Matrices")

        display = (
                      f'{headers[0]} ({len(a)}):\n%s\n\n'
                      f'{headers[1]} ({len(z)}):\n%s\n\n'
                      f'{headers[2]} ({len(W)}):\n%s\n\n'
                      f'{headers[3]} ({len(b)}):\n%s\n\n'
                  ) % (
                      '\n\n'.join(
                          [f'Layer Index: {i}, Shape: {mat.shape}\n{np.array2string(mat)}' for i, mat in
                           enumerate(a, start=0)]
                      ),
                      '\n\n'.join(
                          [f'Layer Index: {i}, Shape: {mat.shape}\n{np.array2string(mat)}' for i, mat in
                           enumerate(z, start=1)]
                      ),
                      '\n\n'.join(
                          [f'Layer Index: {i}, Shape: {mat.shape}\n{np.array2string(mat)}' for i, mat in
                           enumerate(W, start=1)]
                      ),
                      '\n\n'.join(
                          [f'Layer Index: {i}, Shape: {mat.shape}\n{np.array2string(mat)}' for i, mat in
                           enumerate(b, start=1)]
                      ),
                  )

        return '\n'.join(
            line if any(line.startswith(header) for header in headers) else '\t' + line for line in display.splitlines()
        )

    def forward_propagate(self, input_array: np.ndarray) -> np.ndarray:
        self.a[0] = input_array
        for i in range(1, len(self.a)):
            self.z[i] = self.W[i] @ self.a[i - 1] + self.b[i]  # Don't inline this! It's needed for backpropagation
            self.a[i] = self.activation_function(self.z[i])
        self.a[-1] = softmax(self.z[-1])  # Overrides final activation function (sigmoid/tanh/relu) and applies softmax

        return self.a[-1]

    def backward_propagate(self, label: np.ndarray) -> dict[str, list[np.ndarray]]:
        d = [None] + [np.zeros(mat.shape) for mat in self.a[1:]]
        dW = [None] + [np.zeros(mat.shape) for mat in self.W[1:]]
        db = [None] + [np.zeros(mat.shape) for mat in self.b[1:]]

        for i in range(len(self.a) - 1, 0, -1):
            d[i] = self.a[i] - label if i == len(self.a) - 1 else (self.W[i + 1].T @ d[i + 1]) * deriv_relu(self.z[i])
            dW[i] = d[i] @ self.a[i - 1].T
            db[i] = d[i]

        return {'dW': dW, 'db': db}

    def __init_matrices(self, layer_sizes: list[int]):
        """
        Initializes all matrices to 0, except for the weights matrices, which are initialized using Kaiming He
        initialization. W[0] and b[0] are None to align with conventional neural network notation:
        e.g, in forward propagation, the following statement may be used: "a[1] = relu(W[1] @ a[0] + b[1])"
        """
        self.a = [np.zeros((layer_size, 1)) for layer_size in layer_sizes]
        self.z = [None] + [np.zeros((layer_size, 1)) for layer_size in layer_sizes[1:]]
        self.W = NeuralNetwork.__he_init_weights(layer_sizes)
        self.b = [None] + [np.zeros((layer_size, 1)) for layer_size in layer_sizes[1:]]

    @staticmethod
    def __he_init_weights(layer_sizes: list[int]) -> list[None | np.ndarray]:
        return [None] + [
            np.random.normal(loc=0, scale=math.sqrt(2 / input_size), size=(output_size, input_size))
            # For layer_sizes=[5, 3, 2], zip() returns iterable [(3, 5), (2, 3)]
            for output_size, input_size in zip(layer_sizes[1:], layer_sizes[:-1])
        ]


if __name__ == '__main__':
    data = {}
    load_data(
        {
            'train': 'image_data/mnist_train.csv',
            'test': 'image_data/mnist_test.csv'
        }, data
    )

    nn = NeuralNetwork(relu, [784, 128, 64, 10])

    input_vector = np.random.randint(256, size=(784, 1))
    prediction = nn.forward_propagate(input_vector)
    label = one_hot_encode(np.random.randint(10))
    loss = categorical_cross_entropy_loss(label, prediction)
    gradients = nn.backward_propagate(label)

    print(
        f'Prediction (Decoded => {one_hot_decode(prediction)})\n{prediction}\n\n'
        f'Label (Decoded => {one_hot_decode(label)})\n{label}\n\n'
        f'Loss: {loss}'
    )

    print(
        f'Weights Gradients:\n{gradients['dW']}\n\n'
        f'Bias Gradients:\n{gradients['db']}\n\n'
    )
