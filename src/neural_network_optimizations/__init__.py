"""Neural Network with Optimization (Gradient Descent, Momentum, Adam)."""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sklearn.datasets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Supported optimizer types."""

    GRADIENT_DESCENT = "gd"
    MOMENTUM = "momentum"
    ADAM = "adam"


@dataclass
class ModelConfig:
    """Configuration for neural network model."""

    learning_rate: float = 0.0007
    mini_batch_size: int = 64
    beta: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    num_epochs: int = 10000
    print_cost: bool = True


class ActivationFunctions:
    """Neural network activation functions."""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Compute sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Compute ReLU activation."""
        return np.maximum(0, x)


class NeuralNetwork:
    """Implementation of a neural network with various optimization algorithms."""

    def __init__(
        self,
        layer_dims: list[int],
        optimizer_type: OptimizerType,
        config: ModelConfig,
    ) -> None:
        """Init Neural Network."""
        self.layer_dims = layer_dims
        self.optimizer_type = optimizer_type
        self.config = config
        self.parameters = self._initialize_parameters()
        self.cache = {}

    def _initialize_parameters(self) -> dict[str, np.ndarray]:
        """Initialize network parameters using He initialization."""
        np.random.default_rng(3)
        parameters = {}

        for layer in range(1, len(self.layer_dims)):
            parameters[f"W{layer}"] = np.random.randn(
                self.layer_dims[layer],
                self.layer_dims[layer - 1],
            ) * np.sqrt(2 / self.layer_dims[layer - 1])
            parameters[f"b{layer}"] = np.zeros((self.layer_dims[layer], 1))

        return parameters

    def _forward_propagation(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Implement forward propagation.

        Args:
            X: Input data of shape (input_size, m_examples)

        Returns:
            A tuple of (final_activation, cache)

        """
        cache = {}
        A = X
        L = len(self.layer_dims)

        # Forward propagate through layers
        for l in range(1, L):
            A_prev = A
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]

            Z = np.dot(W, A_prev) + b

            # Use ReLU for hidden layers and sigmoid for output
            if l == L - 1:
                A = ActivationFunctions.sigmoid(Z)
            else:
                A = ActivationFunctions.relu(Z)

            cache[f"A{l - 1}"] = A_prev
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        return A, cache

    def _compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """Compute binary cross-entropy cost.

        Args:
            AL: Output of final layer, shape (1, m_examples)
            Y: True labels, shape (1, m_examples)

        """
        m = Y.shape[1]
        return (
            -np.sum(
                Y * np.log(AL + self.config.epsilon)
                + (1 - Y) * np.log(1 - AL + self.config.epsilon),
            )
            / m
        )

    def _backward_propagation(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cache: dict,
    ) -> dict[str, np.ndarray]:
        """Implement backward propagation.

        Returns:
            Dictionary containing gradients

        """
        grads = {}
        m = X.shape[1]
        L = len(self.layer_dims)

        # Output layer
        AL = cache[f"A{L - 1}"]
        dAL = -(
            np.divide(Y, AL + self.config.epsilon)
            - np.divide(1 - Y, 1 - AL + self.config.epsilon)
        )

        # Current layer gradients
        dZ = dAL * AL * (1 - AL)  # Sigmoid derivative
        grads[f"dW{L - 1}"] = np.dot(dZ, cache[f"A{L - 2}"].T) / m
        grads[f"db{L - 1}"] = np.sum(dZ, axis=1, keepdims=True) / m

        # Hidden layers
        for l in reversed(range(1, L - 1)):
            dA = np.dot(self.parameters[f"W{l + 1}"].T, dZ)
            dZ = dA * (cache[f"A{l}"] > 0)  # ReLU derivative
            grads[f"dW{l}"] = np.dot(dZ, cache[f"A{l - 1}"].T) / m
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m

        return grads

    def _random_mini_batches(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        seed: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create random mini-batches from training data."""
        rng = np.random.default_rng(seed)
        m = X.shape[1]
        mini_batches = []

        # Shuffle
        permutation = list(rng.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        # Create mini-batches
        num_complete_batches = m // self.config.mini_batch_size

        for k in range(num_complete_batches):
            start_idx = k * self.config.mini_batch_size
            end_idx = (k + 1) * self.config.mini_batch_size
            mini_batch_X = shuffled_X[:, start_idx:end_idx]
            mini_batch_Y = shuffled_Y[:, start_idx:end_idx]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        # Handle final incomplete batch if needed
        if m % self.config.mini_batch_size != 0:
            mini_batch_X = shuffled_X[
                :,
                num_complete_batches * self.config.mini_batch_size :,
            ]
            mini_batch_Y = shuffled_Y[
                :,
                num_complete_batches * self.config.mini_batch_size :,
            ]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        return mini_batches

    def train(self, X: np.ndarray, Y: np.ndarray) -> list[float]:
        """Train the neural network.

        Args:
            X: Training data of shape (n_features, m_examples)
            Y: Labels of shape (1, m_examples)

        Returns:
            list of costs during training

        """
        costs = []
        t = 0  # Adam iteration counter
        m = X.shape[1]

        # Initialize optimizer-specific variables
        if self.optimizer_type == OptimizerType.MOMENTUM:
            v = self._initialize_velocity()
        elif self.optimizer_type == OptimizerType.ADAM:
            v, s = self._initialize_adam()

        # Training loop
        for i in range(self.config.num_epochs):
            epoch_cost = 0
            mini_batches = self._random_mini_batches(X, Y, seed=i)

            for mini_batch in mini_batches:
                mini_X, mini_Y = mini_batch

                # Forward propagation
                AL, cache = self._forward_propagation(mini_X)

                # Compute cost
                mini_cost = self._compute_cost(AL, mini_Y)
                epoch_cost += mini_cost

                # Backward propagation
                grads = self._backward_propagation(mini_X, mini_Y, cache)

                # Update parameters based on optimizer
                if self.optimizer_type == OptimizerType.GRADIENT_DESCENT:
                    self._update_parameters_gd(grads)
                elif self.optimizer_type == OptimizerType.MOMENTUM:
                    self._update_parameters_momentum(grads, v)
                elif self.optimizer_type == OptimizerType.ADAM:
                    t += 1
                    self._update_parameters_adam(grads, v, s, t)

            # Average epoch cost
            epoch_cost /= m

            # Print cost and save to history
            if self.config.print_cost and i % 1000 == 0:
                cost_msg = f"Cost after epoch {i}: {epoch_cost:.6f}"
                logger.info(cost_msg)
            if i % 100 == 0:
                costs.append(epoch_cost)

        return costs

    def predict(self, X: np.ndarray) -> int:
        """Make predictions using trained model.

        Args:
            X: Input data of shape (n_features, m_examples)

        Returns:
            Predictions (0/1) of shape (1, m_examples)

        """
        AL, _ = self._forward_propagation(X)
        threshold = 0.5
        return (threshold < AL).astype(int)

    def _initialize_velocity(self) -> dict[str, np.ndarray]:
        """Initialize velocity for momentum optimization."""
        v = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            v[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
            v[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])

        return v

    def _initialize_adam(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Initialize Adam optimizer variables."""
        v = {}  # First moment
        s = {}  # Second moment
        L = len(self.layer_dims)

        for l in range(1, L):
            v[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
            v[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])
            s[f"dW{l}"] = np.zeros_like(self.parameters[f"W{l}"])
            s[f"db{l}"] = np.zeros_like(self.parameters[f"b{l}"])

        return v, s

    def _update_parameters_gd(self, grads: dict[str, np.ndarray]) -> None:
        """Update parameters using gradient descent."""
        L = len(self.layer_dims)

        for l in range(1, L):
            self.parameters[f"W{l}"] -= self.config.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.config.learning_rate * grads[f"db{l}"]

    def _update_parameters_momentum(
        self,
        grads: dict[str, np.ndarray],
        v: dict[str, np.ndarray],
    ) -> None:
        """Update parameters using momentum optimization."""
        L = len(self.layer_dims)

        for l in range(1, L):
            # Update velocities
            v[f"dW{l}"] = (
                self.config.beta * v[f"dW{l}"]
                + (1 - self.config.beta) * grads[f"dW{l}"]
            )
            v[f"db{l}"] = (
                self.config.beta * v[f"db{l}"]
                + (1 - self.config.beta) * grads[f"db{l}"]
            )

            # Update parameters
            self.parameters[f"W{l}"] -= self.config.learning_rate * v[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.config.learning_rate * v[f"db{l}"]

    def _update_parameters_adam(
        self,
        grads: dict[str, np.ndarray],
        v: dict[str, np.ndarray],
        s: dict[str, np.ndarray],
        t: int,
    ) -> None:
        """Update parameters using Adam optimization."""
        L = len(self.layer_dims)
        v_corrected = {}
        s_corrected = {}

        for l in range(1, L):
            # Update first moment
            v[f"dW{l}"] = (
                self.config.beta1 * v[f"dW{l}"]
                + (1 - self.config.beta1) * grads[f"dW{l}"]
            )
            v[f"db{l}"] = (
                self.config.beta1 * v[f"db{l}"]
                + (1 - self.config.beta1) * grads[f"db{l}"]
            )

            # Update second moment
            s[f"dW{l}"] = self.config.beta2 * s[f"dW{l}"] + (
                1 - self.config.beta2
            ) * np.square(grads[f"dW{l}"])
            s[f"db{l}"] = self.config.beta2 * s[f"db{l}"] + (
                1 - self.config.beta2
            ) * np.square(grads[f"db{l}"])

            # Correct bias
            v_corrected[f"dW{l}"] = v[f"dW{l}"] / (1 - self.config.beta1**t)
            v_corrected[f"db{l}"] = v[f"db{l}"] / (1 - self.config.beta1**t)
            s_corrected[f"dW{l}"] = s[f"dW{l}"] / (1 - self.config.beta2**t)
            s_corrected[f"db{l}"] = s[f"db{l}"] / (1 - self.config.beta2**t)

            # Update parameters
            self.parameters[f"W{l}"] -= (
                self.config.learning_rate
                * v_corrected[f"dW{l}"]
                / (np.sqrt(s_corrected[f"dW{l}"]) + self.config.epsilon)
            )

            self.parameters[f"b{l}"] -= (
                self.config.learning_rate
                * v_corrected[f"db{l}"]
                / (np.sqrt(s_corrected[f"db{l}"]) + self.config.epsilon)
            )


class ModelMetrics:
    """Utility class for computing model metrics."""

    @staticmethod
    def compute_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
        """Compute prediction accuracy."""
        return np.mean(predictions == Y)

    @staticmethod
    def compute_f1_score(predictions: np.ndarray, Y: np.ndarray) -> float:
        """Compute F1 score."""
        true_positives = np.sum((predictions == 1) & (Y == 1))
        false_positives = np.sum((predictions == 1) & (Y == 0))
        false_negatives = np.sum((predictions == 0) & (Y == 1))

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        return 2 * (precision * recall) / (precision + recall + 1e-10)


def create_neural_network(
    layer_dims: list[int],
    optimizer: str = "adam",
    **kwargs: dict[str, any],
) -> NeuralNetwork:
    """Create a neural network with specified configuration.

    Args:
        layer_dims: list of layer dimensions
        optimizer: Optimization algorithm ('gd', 'momentum', or 'adam')
        **kwargs: Additional configuration parameters

    Returns:
        Configured NeuralNetwork instance

    """
    optimizer_type = OptimizerType(optimizer)

    config = ModelConfig(**kwargs)

    return NeuralNetwork(layer_dims, optimizer_type, config)


if __name__ == "__main__":
    # Generate sample data
    X, Y = sklearn.datasets.make_moons(n_samples=300, noise=0.2)
    X = X.T
    Y = Y.reshape(1, -1)

    # Create and configure network
    layer_dims = [2, 5, 2, 1]  # 2 input features, 2 hidden layers, 1 output

    # Create network using factory function
    model = create_neural_network(
        layer_dims=layer_dims,
        optimizer=OptimizerType.ADAM,
        learning_rate=0.001,
        mini_batch_size=32,
        num_epochs=10000,
    )

    # Train model
    model.train(X, Y)

    # Make predictions
    predictions = model.predict(X)

    # Compute metrics
    accuracy = ModelMetrics.compute_accuracy(predictions, Y)
    f1_score = ModelMetrics.compute_f1_score(predictions, Y)

    accuracy_msg = f"Final accuracy: {accuracy:.4f}"
    f1_msg = f"F1 score: {f1_score:.4f}"
    logger.info("Training completed successfully")
    logger.info(accuracy_msg)
    logger.info(f1_msg)
