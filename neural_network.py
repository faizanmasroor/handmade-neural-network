import math
import pickle as pkl

import cv2
import numpy as np
from PIL import Image

from data import load_data
from matrix_math import relu, deriv_relu, softmax, categorical_cross_entropy_loss, one_hot_decode

DATA_PICKLE = 'pickles/data.pkl'
NN_PICKLE = 'pickles/nn.pkl'

NUM_EPOCHS = 80
BATCH_SIZE = 128
LR_INITIAL = 0.05  # Represents initial learning rate
LR_DECAY_RATE = 0.95  # Learning rate decays by 5% each epoch; set to 1 to deactivate learning rate decay
DROPOUT_RATE = 0.4

CANVAS_SIZE = 280
DOWNSCALE_SIZE = 28
PAINTBRUSH_THICKNESS = 16

np.set_printoptions(threshold=12, edgeitems=6, linewidth=200, suppress=True)


class NeuralNetwork:
    def __init__(self, activation_function: callable, layer_sizes: list[int]):
        self.activation_function = activation_function
        self._init_matrices(layer_sizes)

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

    def train(self, num_epochs: int, train_data: list[tuple[np.ndarray, np.ndarray]]):
        for epoch_idx in range(num_epochs):
            # Size should not exceed the constant "BATCH_SIZE"
            batch_gradients: list[dict[str, list[None | np.ndarray]]] = []
            batch_losses: list[float] = []
            batch_num = 0
            np.random.shuffle(train_data)
            for sample_idx, (label, pixels) in enumerate(train_data, 1):
                # IMPORTANT! Make forward propagation and backward propagation are called in order
                prediction = self.forward_propagate(pixels, DROPOUT_RATE, True)
                gradients = self.backward_propagate(label)

                batch_gradients += [gradients]
                batch_losses += [categorical_cross_entropy_loss(label, prediction)]

                # If true, perform batch gradient descent
                if len(batch_gradients) >= BATCH_SIZE or sample_idx == len(train_data):
                    batch_num += 1
                    num_layers = len(batch_gradients[0]['dW'])

                    """
                    The scaled (by learning rate) batch-averaged gradient for the weights and biases, at each layer;
                    e.g., batch_estimate['dW'][2] stores the average dW2 np.ndarray of the last batch's backpropagations
                    """
                    batch_estimate: dict[str, list[None | np.ndarray]] = {
                        'dW': [None] + [
                            (LR_INITIAL * (LR_DECAY_RATE ** epoch_idx)) * np.mean(
                                [batch_gradient['dW'][layer_idx] for batch_gradient in batch_gradients],
                                axis=0
                            )
                            for layer_idx in range(1, num_layers)
                        ],
                        'db': [None] + [
                            (LR_INITIAL * (LR_DECAY_RATE ** epoch_idx)) * np.mean(
                                [batch_gradient['db'][layer_idx] for batch_gradient in batch_gradients],
                                axis=0
                            )
                            for layer_idx in range(1, num_layers)
                        ]
                    }

                    for layer_idx in range(1, num_layers):
                        self.W[layer_idx] -= batch_estimate['dW'][layer_idx]
                        self.b[layer_idx] -= batch_estimate['db'][layer_idx]

                    print(
                        f'Epoch #{epoch_idx + 1}, '
                        f'Sample #{sample_idx}, '
                        f'Batch #{batch_num}, '
                        f'Batch Average Loss: {sum(batch_losses) / len(batch_losses)}'
                    )

                    batch_gradients = []
                    batch_losses = []

    def test(self, test_data: list[tuple[np.ndarray, np.ndarray]]):
        num_correct = 0
        for label, pixels in test_data:
            prediction = self.forward_propagate(pixels)
            if one_hot_decode(prediction) == one_hot_decode(label): num_correct += 1

        print(f'Accuracy: {100 * num_correct / len(test_data)}% ({num_correct}/{len(test_data)})')

    def forward_propagate(self, input_array: np.ndarray, dropout_rate: float = 0, training: bool = False) -> np.ndarray:
        input_array = input_array.astype(np.float64)

        self.a[0] = input_array
        for i in range(1, len(self.a)):
            self.z[i] = self.W[i] @ self.a[i - 1] + self.b[i]  # Don't inline this! It's needed for backpropagation
            self.a[i] = self.activation_function(self.z[i])

            # Dropout (only on hidden layers during training)
            if training and dropout_rate > 0 and i != len(self.a) - 1:
                dropout_mask = (np.random.rand(*self.a[i].shape) > dropout_rate).astype(np.float64)
                self.a[i] *= dropout_mask
                # Inverted dropout; results in not needing to scale activations by dropout rate during inference
                self.a[i] /= 1 - dropout_rate
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

    def _init_matrices(self, layer_sizes: list[int]):
        """
        Initializes all matrices to 0, except for the weights matrices, which are initialized using Kaiming He
        initialization. W[0] and b[0] are None to align with conventional neural network notation:
        e.g, in forward propagation, the following statement may be used: "a[1] = relu(W[1] @ a[0] + b[1])"
        """
        self.a = [np.zeros((layer_size, 1)) for layer_size in layer_sizes]
        self.z = [None] + [np.zeros((layer_size, 1)) for layer_size in layer_sizes[1:]]
        self.W = NeuralNetwork._he_init_weights(layer_sizes)
        self.b = [None] + [np.zeros((layer_size, 1)) for layer_size in layer_sizes[1:]]

    @staticmethod
    def _he_init_weights(layer_sizes: list[int]) -> list[None | np.ndarray]:
        return [None] + [
            np.random.normal(loc=0, scale=math.sqrt(2 / input_size), size=(output_size, input_size))
            # For layer_sizes=[5, 3, 2], zip() returns iterable [(3, 5), (2, 3)]
            for output_size, input_size in zip(layer_sizes[1:], layer_sizes[:-1])
        ]


if __name__ == '__main__':
    try:
        with open(DATA_PICKLE, 'rb') as f:
            data = pkl.load(f)
    except FileNotFoundError as e:
        print(f'Pickle file not found: {e.filename}; Loading data from CSV...')
        data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        load_data({
            'test': 'data/mnist_test.csv',
            'train': 'data/mnist_train.csv'
        }, data)

        print(f'Saving training and testing data as {DATA_PICKLE}...')
        with open(DATA_PICKLE, 'wb') as f:
            pkl.dump(data, f)

    try:
        with open(NN_PICKLE, 'rb') as f:
            nn = pkl.load(f)
        print(f'Successfully loaded NeuralNetwork from {NN_PICKLE}!')
        nn.test(data['test'])
    except FileNotFoundError as e:
        print(f'Pickle file not found: {e.filename}; Initializing and training neural network...')
        nn = NeuralNetwork(relu, [784, 512, 256, 128, 10])
        nn.train(NUM_EPOCHS, data['train'])

        print(f'Saving neural network as {NN_PICKLE}...')
        with open(NN_PICKLE, 'wb') as f:
            pkl.dump(nn, f)

    # Stream canvas window and predict image input
    while True:
        canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
        state: dict[str, bool | None | tuple[int, int]] = {'drawing': False, 'last_point': None}

        # Do not remove unused parameters, they are needed for cv2 to successfully use this draw() function
        def draw(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                state['drawing'] = True
                state['last_point'] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
                if state['last_point'] is not None:
                    cv2.line(canvas, state['last_point'], (x, y), (0,), thickness=PAINTBRUSH_THICKNESS)
                    state['last_point'] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                state['drawing'] = False
                state['last_point'] = None

        cv2.namedWindow('Draw a digit (ESC to quit, Enter to predict)')
        cv2.setMouseCallback('Draw a digit (ESC to quit, Enter to predict)', draw)

        while True:
            cv2.imshow('Draw a digit (ESC to quit, Enter to predict)', canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                exit()
            elif key == 13:  # Enter
                break

        # Downscale to 28x28
        img_resized = cv2.resize(canvas, (DOWNSCALE_SIZE, DOWNSCALE_SIZE), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(img_resized)
        arr = np.asarray(image, dtype=np.float64)
        arr = arr.flatten().reshape(-1, 1)
        arr = (255 - arr) / 255  # Invert the grayscale to match the MNIST dataset and scales pixel values to [0, 1]

        print('Predicted digit:', one_hot_decode(nn.forward_propagate(arr)))
        canvas[:, :] = 255
