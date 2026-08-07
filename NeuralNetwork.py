import numpy as np


# =========================
# DENSE LAYER
# =========================

class Dense:

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)

        self.input = None
        self.dweights = None
        self.dbiases = None

    def forward(self, x):
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, gradient):

        # batch version
        self.dweights = self.input.T @ gradient

        self.dbiases = np.sum(
            gradient,
            axis=0
        )

        dinput = gradient @ self.weights.T

        return dinput



# =========================
# SIGMOID
# =========================

class Sigmoid:

    def forward(self, x):

        self.output = 1 / (1 + np.exp(-x))

        return self.output


    def backward(self, gradient):

        derivative = self.output * (1 - self.output)

        return gradient * derivative



# =========================
# RELU
# =========================

class ReLU:

    def forward(self, x):

        self.input = x

        return np.maximum(0, x)


    def backward(self, gradient):

        derivative = self.input > 0

        return gradient * derivative



# =========================
# LEAKY RELU
# =========================

class LeakyReLU:

    def __init__(self, alpha=0.01):

        self.alpha = alpha


    def forward(self, x):

        self.input = x

        return np.maximum(self.alpha * x, x)


    def backward(self, gradient):

        derivative = np.where(
            self.input > 0,
            1.0,
            self.alpha
        )

        return gradient * derivative



# =========================
# NEURAL NETWORK
# =========================

class NeuralNetwork:

    def __init__(self):

        self.layers = []


    def add(self, layer):

        self.layers.append(layer)


    def forward(self, x):

        for layer in self.layers:

            x = layer.forward(x)

        return x


    def backward(self, gradient):

        for layer in reversed(self.layers):

            gradient = layer.backward(gradient)

        return gradient



# =========================
# MSE LOSS
# =========================

class MSELoss:

    def forward(self, pred, target):

        return np.mean((pred - target) ** 2)


    def backward(self, pred, target):

        return 2 * (pred - target) / pred.size


# =========================
# CrossEntropy LOSS
# =========================

class CrossEntropyLoss:

    def forward(self, logits, targets):

        # numerical stability
        shifted = logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )

        exp_values = np.exp(shifted)

        self.probabilities = (
            exp_values
            / np.sum(
                exp_values,
                axis=1,
                keepdims=True
            )
        )

        self.targets = targets

        n = logits.shape[0]

        correct_probs = self.probabilities[
            np.arange(n),
            targets
        ]

        loss = -np.mean(
            np.log(correct_probs + 1e-8)
        )

        return loss


    def backward(self):

        n = self.probabilities.shape[0]

        gradient = self.probabilities.copy()

        gradient[
            np.arange(n),
            self.targets
        ] -= 1

        gradient /= n

        return gradient



