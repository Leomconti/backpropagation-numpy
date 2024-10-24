from dataclasses import dataclass
from random import random, seed
from typing import List

import numpy as np

seed(1)


@dataclass
class Neuron:
    weights: List[float]
    bias: float = 0.0
    output: float = 0.0
    delta: float = 0.0

    def __str__(self):
        return f"Neuron: Weights: {[f'{w:.4f}' for w in self.weights]}, Bias: {self.bias:.4f}, Output: {self.output:.4f}, Delta: {self.delta:.4f}"


@dataclass
class Layer:
    neurons: List[Neuron]

    def __str__(self):
        return f"Layer: [{', '.join([str(neuron) for neuron in self.neurons])}]"


@dataclass
class Network:
    layers: List[Layer]

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def get_layer(self, index: int) -> Layer:
        return self.layers[index]

    def __str__(self):
        return f"Network: [{', '.join([str(layer) for layer in self.layers])}]"


@dataclass
class TrainingDataRow:
    inputs: List[float]
    expected: List[float]


@dataclass
class TrainingDataset:
    rows: List[TrainingDataRow]

    @property
    def num_inputs(self) -> int:
        return len(self.rows[0].inputs)

    @property
    def num_outputs(self) -> int:
        return len(self.rows[0].expected)


def init_network(n_inputs: int, n_hidden: int, n_outputs: int) -> Network:
    hidden = Layer(
        neurons=[
            Neuron(
                weights=[random() - 0.5 for _ in range(n_inputs)],
                bias=random() - 0.5,
            )
            for _ in range(n_hidden)
        ]
    )
    output = Layer(
        neurons=[
            Neuron(
                weights=[random() - 0.5 for _ in range(n_hidden)],
                bias=random() - 0.5,
            )
            for _ in range(n_outputs)
        ]
    )
    return Network(layers=[hidden, output])


# 2. Forward Propagation
# 2.1 Neuron Activation
def activate(neuron: Neuron, inputs: List[float]) -> float:
    """
    Neuron activation is the weighted sum of the inputs
     - activation = sum(weight_i * input_i) + bias
    Could use np.dot to make it more efficient, as its the dot product of the weights and inputs + bias at the end
    """
    return np.dot(inputs, neuron.weights) + neuron.bias


# 2.2 Neuron transfer
def transfer(activation: float) -> float:
    """
    Transfer the activation to see what the neuron should output
    Transfer functions are used to normalize the output of a neuron
    to a value between 0 and 1
    sigmoid is a common transfer function, also tanh, rectifier..
    - output = 1 / (1 + e^(-activation))
    """
    return 1.0 / (1.0 + np.exp(-activation))


def forward_propagate(network: Network, inputs: List[float]) -> List[float]:
    """
    All of the outputs from one layer become inputs to the neurons on the next layer
    """
    current_inputs = inputs
    for layer in network.layers:
        new_inputs = []
        for neuron in layer.neurons:
            activation = activate(neuron, current_inputs)
            neuron.output = transfer(activation)
            new_inputs.append(neuron.output)
        current_inputs = new_inputs
    return current_inputs


# 3. Backpropagation
# 3.1 Calculate Derivative of Transfer Function
# Derivative of sigmoid is output * (1 - output)
def transfer_derivative(output: float) -> float:
    return output * (1.0 - output)


# 3.2 Calculate Error Backpropagation
def backpropagate_error(network: Network, expected: List[float]):
    """
    Calculate the error for each output neuron and propagate it backwards through the network.

    For output neurons:
    - error = (output - expected) * transfer_derivative(output)

    For hidden neurons:
    - error = (weight_k * error_j) * transfer_derivative(output)

    This gives the error signal (input) to propagate backwards through the layers.
    """
    for i in reversed(range(len(network.layers))):
        layer = network.layers[i]
        errors = []

        if i == len(network.layers) - 1:
            # Output layer
            for j, neuron in enumerate(layer.neurons):
                errors.append(expected[j] - neuron.output)
        else:
            # Hidden layer
            for j in range(len(layer.neurons)):
                error = 0.0
                for neuron in network.layers[i + 1].neurons:
                    error += neuron.weights[j] * neuron.delta
                errors.append(error)

        for j, neuron in enumerate(layer.neurons):
            neuron.delta = errors[j] * transfer_derivative(neuron.output)


# 4. Train Network
def update_weights(network: Network, inputs: List[float], learning_rate: float):
    """
    Update the weights and biases of the network based on the calculated deltas.
    """
    for i, layer in enumerate(network.layers):
        if i != 0:
            inputs = [neuron.output for neuron in network.layers[i - 1].neurons]
        for neuron in layer.neurons:
            for j, input_val in enumerate(inputs):
                neuron.weights[j] += learning_rate * neuron.delta * input_val
            neuron.bias += learning_rate * neuron.delta


def train_network(
    network: Network, dataset: TrainingDataset, learning_rate: float, n_epochs: int, target_error: float = 0.01
) -> Network:
    """
    Train the network on a list of input-output pairs. That will be given in the training data model.
    """
    for epoch in range(n_epochs):
        sum_error = 0
        for row in dataset.rows:
            outputs = forward_propagate(network, row.inputs)
            error = sum((expected - output) ** 2 for expected, output in zip(row.expected, outputs))
            sum_error += error
            backpropagate_error(network, row.expected)
            update_weights(network, row.inputs, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch={epoch}, Error={sum_error:.7f}")

        if sum_error < target_error:
            print(f"Early stopping at epoch {epoch}, Error={sum_error:.7f}")
            break

    return network


def predict(network: Network, inputs: List[float]) -> List[float]:
    """
    Predict the output of the network for a given set of inputs.
    """
    return forward_propagate(network, inputs)


# Test functions for different logical operations
def test_network(network: Network, dataset: TrainingDataset, operation: str):
    print(f"\nTesting {operation} operation:")
    correct = 0
    total = len(dataset.rows)

    for row in dataset.rows:
        outputs = predict(network, row.inputs)
        prediction = 1 if outputs[0] >= 0.5 else 0
        expected = row.expected[0]
        print(f"Input: {row.inputs}, Expected: {expected}, Got: {prediction} (Raw: {outputs[0]:.4f})")
        if prediction == expected:
            correct += 1

    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy}%")


def main():
    # XOR dataset
    xor_dataset = TrainingDataset(
        rows=[
            TrainingDataRow(inputs=[0, 0], expected=[0]),
            TrainingDataRow(inputs=[0, 1], expected=[1]),
            TrainingDataRow(inputs=[1, 0], expected=[1]),
            TrainingDataRow(inputs=[1, 1], expected=[0]),
        ]
    )

    # AND dataset
    and_dataset = TrainingDataset(
        rows=[
            TrainingDataRow(inputs=[0, 0], expected=[0]),
            TrainingDataRow(inputs=[0, 1], expected=[0]),
            TrainingDataRow(inputs=[1, 0], expected=[0]),
            TrainingDataRow(inputs=[1, 1], expected=[1]),
        ]
    )

    # OR dataset
    or_dataset = TrainingDataset(
        rows=[
            TrainingDataRow(inputs=[0, 0], expected=[0]),
            TrainingDataRow(inputs=[0, 1], expected=[1]),
            TrainingDataRow(inputs=[1, 0], expected=[1]),
            TrainingDataRow(inputs=[1, 1], expected=[1]),
        ]
    )

    # Training parameters
    n_hidden = 4  # Increased number of hidden neurons
    learning_rate = 0.3
    n_epochs = 5000

    # Train and test XOR
    print("\nTraining XOR network...")
    xor_network = init_network(2, n_hidden, 1)
    xor_network = train_network(xor_network, xor_dataset, learning_rate, n_epochs)
    test_network(xor_network, xor_dataset, "XOR")

    # Train and test AND
    print("\nTraining AND network...")
    and_network = init_network(2, n_hidden, 1)
    and_network = train_network(and_network, and_dataset, learning_rate, n_epochs)
    test_network(and_network, and_dataset, "AND")

    # Train and test OR
    print("\nTraining OR network...")
    or_network = init_network(2, n_hidden, 1)
    or_network = train_network(or_network, or_dataset, learning_rate, n_epochs)
    test_network(or_network, or_dataset, "OR")


if __name__ == "__main__":
    main()
