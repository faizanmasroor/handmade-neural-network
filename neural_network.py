import math

import numpy as np

from data import load_data

NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

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
        input_array = input_array.astype(np.float64)

        self.a[0] = input_array
        for i in range(1, len(self.a)):
            self.z[i] = self.W[i] @ self.a[i - 1] + self.b[i]  # Don't inline this! It's needed for backpropagation
            self.a[i] = self.activation_function(self.z[i])
        self.a[-1] = softmax(self.z[-1])  # Overrides final activation function (sigmoid/tanh/relu) and applies softmax

        return self.a[-1]

    def backward_propagate(self, label: np.ndarray) -> dict[str, list[None | np.ndarray]]:
        label = label.astype(np.float64)

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
            'test': 'image_data/mnist_test.csv',
            'train': 'image_data/mnist_train.csv'
        }, data
    )

    print('Preprocessing "test" data...')
    data['test'] = [
        (one_hot_encode(int(raw_vector[0])), raw_vector[1:].reshape(-1, 1).astype(np.float64))
        for raw_vector in data['test']
    ]
    print('Finished preprocessing "test" data!')

    print('Preprocessing "train" data...')
    data['train'] = [
        (one_hot_encode(int(raw_vector[0])), raw_vector[1:].reshape(-1, 1).astype(np.float64))
        for raw_vector in data['train']
    ]
    print('Finished preprocessing "train" data!')

    nn = NeuralNetwork(relu, [784, 128, 64, 10])

    for epoch_idx in range(NUM_EPOCHS):
        # Size should not exceed the constant "BATCH_SIZE"
        batch_gradients: list[dict[str, list[None | np.ndarray]]] = []
        batch_losses: list[float] = []
        batch_num = 0
        for sample_idx, (label, pixels) in enumerate(data['train'], 1):
            gradient_descent_performed = False

            # IMPORTANT! Make forward propagation and backward propagation are called in order
            prediction = nn.forward_propagate(pixels)
            gradients = nn.backward_propagate(label)

            batch_gradients += [gradients]
            batch_losses += [categorical_cross_entropy_loss(label, prediction)]

            # If true, perform batch gradient descent
            if len(batch_gradients) >= BATCH_SIZE or sample_idx == len(data['train']):
                batch_num += 1
                gradient_descent_performed = True
                num_layers = len(batch_gradients[0]['dW'])

                # The scaled (by learning rate) batch-averaged gradient for the weights and biases, at each layer;
                # e.g., batch_estimate['dW'][2] stores the average dW2 np.ndarray of the last batch's backpropagations
                batch_estimate: dict[str, list[None | np.ndarray]] = {
                    'dW': [None] + [
                        LEARNING_RATE * np.mean(
                            [batch_gradient['dW'][layer_idx] for batch_gradient in batch_gradients],
                            axis=0
                        )
                        for layer_idx in range(1, num_layers)
                    ],
                    'db': [None] + [
                        LEARNING_RATE * np.mean(
                            [batch_gradient['db'][layer_idx] for batch_gradient in batch_gradients],
                            axis=0
                        )
                        for layer_idx in range(1, num_layers)
                    ]
                }

                for layer_idx in range(1, num_layers):
                    nn.W[layer_idx] -= batch_estimate['dW'][layer_idx]
                    nn.b[layer_idx] -= batch_estimate['db'][layer_idx]

                print(
                    f'Epoch #{epoch_idx}, Sample #{sample_idx}{f', Batch #{batch_num}, Batch Average Loss: {sum(batch_losses) / len(batch_losses)}' if gradient_descent_performed else ''}'
                )

                batch_gradients = []
                batch_losses = []

    num_correct = 0
    for label, pixels in data['test']:
        prediction = nn.forward_propagate(pixels)
        if one_hot_decode(prediction) == one_hot_decode(label): num_correct += 1

    print(f'Accuracy: {100 * num_correct / len(data['test'])}% ({num_correct}/{len(data['test'])})')
