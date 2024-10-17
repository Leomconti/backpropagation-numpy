from cgi import print_directory
from typing import List

import numpy as np
from pydantic import BaseModel


class Neuron(BaseModel):
    weights: List[float]
    bias: float = 0.0
    output: float = 0.0
    delta: float = 0.0

    def __str__(self):
        return f"Neuron: Weights: {[f'{w:.4f}' for w in self.weights]}, Bias: {self.bias:.4f}, Output: {self.output:.4f}, Delta: {self.delta:.4f}"


class Layer(BaseModel):
    neurons: List[Neuron]

    def __str__(self):
        return f"Layer: [{', '.join([str(neuron) for neuron in self.neurons])}]"


class Network(BaseModel):
    layers: List[Layer] = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def get_layer(self, index: int) -> Layer:
        return self.layers[index]

    def __str__(self):
        return f"Network: [{', '.join([str(layer) for layer in self.layers])}]"


# 1. Define the network
def init_network(n_inputs: int, n_hidden: int, n_outputs: int) -> Network:
    network = Network()

    hidden = Layer(
        neurons=[
            Neuron(
                weights=[np.random.random() for _ in range(n_inputs)],
                bias=np.random.random(),
            )
            for _ in range(n_hidden)
        ]
    )
    network.add_layer(hidden)
    output = Layer(
        neurons=[
            Neuron(
                weights=[np.random.random() for _ in range(n_hidden)],
                bias=np.random.random(),
            )
            for _ in range(n_outputs)
        ]
    )
    network.add_layer(output)

    return network


# 2. Forward propagation


# 2.1 Neuron Activation
def activate(neuron: Neuron, inputs: List[float]) -> float:
    """
    Neuron activation is the weighted sum of the inputs
     - activation = sum(weight_i * input_i) + bias
    Could use np.dot to make it more efficient, as its the dot product of the weights and inputs + bias at the end
    """
    # activation = neuron.bias
    # for i, _input in enumerate(inputs):
    #     activation += _input * neuron.weights[i]
    # return activation
    return np.dot(inputs, neuron.weights) + neuron.bias


# 2.2 Neuron Transfer
def transfer(activation: float) -> float:
    """
    Transfer the activation to see what the neuron should output
    Transfer functions are used to normalize the output of a neuron
    to a value between 0 and 1
    sigmoid is a common transfer function, also tanh, rectifier..
    - output = 1 / (1 + e^(-activation))
    """
    return 1.0 / (1.0 + np.exp(-activation))


# 2.3 Forward Propagation
def forward_propagate(network: Network, inputs: List[float]) -> List[float]:
    """
    All of the outputs from one layer become inputs to the neurons on the next layer
    """
    for i, layer in enumerate(network.layers):
        # print(f"Passing layer {i}")
        new_inputs = []
        for j, neuron in enumerate(layer.neurons):
            # print(f"Passing neuron {j}")
            activation = activate(neuron, inputs)
            neuron.output = transfer(activation)
            new_inputs.append(neuron.output)
        inputs = new_inputs  # Set new inputs to the outputs of the current layer
    #     print("Inputs calculated... next layer")
    # print("Finished forward propagation")
    return inputs


# 3. Backpropagation


# 3.1 Calculate Derivative of Transfer Function
# Derivative of sigmoid is output * (1 - output)
def transfer_derivative(output: float) -> float:
    return output * (1 - output)


# 3.2 Calculate Error Backpropagation
def backpropagate_error(network: Network, expected: List[float]) -> Network:
    """
    Calculate the error for each output neuron and propagate it backwards through the network.

    For output neurons:
    - error = (output - expected) * transfer_derivative(output)

    For hidden neurons:
    - error = (weight_k * error_j) * transfer_derivative(output)

    This gives the error signal (input) to propagate backwards through the layers.
    """
    # print("Starting backpropagation...")
    # For layer in the network, reverse order
    for i in reversed(range(len(network.layers))):
        layer = network.get_layer(i)
        # print(f"Processing layer {i}")
        errors = []
        if i != len(network.layers) - 1:
            # print("Processing hidden layer")
            # For neuron in the layer
            for j, neuron in enumerate(layer.neurons):
                error = 0.0
                # For neuron in the next layer
                for neuron in network.get_layer(i + 1).neurons:
                    error += neuron.weights[j] * neuron.delta
                errors.append(error)
                # print(f"Hidden neuron {j} error: {error}")
        else:
            # print("Processing output layer")
            for j, neuron in enumerate(layer.neurons):
                error = neuron.output - expected[j]
                errors.append(error)
                # print(f"Output neuron {j} error: {error}")

        # For neuron in the layer
        for j, neuron in enumerate(layer.neurons):
            neuron.delta = errors[j] * transfer_derivative(neuron.output)
            # print(f"Neuron {j} delta: {neuron.delta}")

    # print("Backpropagation completed")
    return network


# 4. Train Network
# weight = weight + (learning_rate * neuron.delta * input)
def update_weights(network: Network, inputs: List[float], learning_rate: float) -> Network:
    """
    Update the weights and biases of the network based on the calculated deltas.
    """
    # print("Starting weight updates...")
    for i, layer in enumerate(network.layers):
        # print(f"Updating weights for layer {i}")
        for n, neuron in enumerate(layer.neurons):
            # print(f"  Updating neuron {n}")
            for j, _input in enumerate(inputs):
                old_weight = neuron.weights[j]
                neuron.weights[j] -= learning_rate * neuron.delta * _input
                # print(f"    Weight {j} updated: {old_weight:.4f} -> {neuron.weights[j]:.4f}")
            old_bias = neuron.bias
            neuron.bias -= learning_rate * neuron.delta
            # print(f"    Bias updated: {old_bias:.4f} -> {neuron.bias:.4f}")
    # print("Weight updates completed")
    return network


class TrainingDataRow(BaseModel):
    inputs: List[float]
    expected: List[float]


class TrainingDataset(BaseModel):
    rows: List[TrainingDataRow]

    @property
    def num_inputs(self) -> int:
        return len(self.rows[0].inputs)

    @property
    def num_outputs(self) -> int:
        return len(self.rows[0].expected)


def calculate_error(outputs: List[float], expected: List[float]) -> float:
    """
    Calculate the error between the outputs and the expected outputs.
    """
    return np.sum(np.square(np.array(expected) - np.array(outputs)))


def train_network(network: Network, training_data: TrainingDataset, learning_rate: float, n_epochs: int) -> Network:
    """
    Train the network on a list of input-output pairs. That will be given in the training data model.
    """
    for epoch in range(n_epochs):
        total_error = 0.0
        for data in training_data.rows:
            # Check the error
            outputs = forward_propagate(network, data.inputs)
            total_error += calculate_error(outputs, data.expected)
            backpropagate_error(network, data.expected)
            update_weights(network, data.inputs, learning_rate)
        print(f">epoch={epoch}, lrate={learning_rate:.3f}, error={total_error:.3f}")
    return network


def predict(network: Network, inputs: List[float]) -> float:
    """
    Predict the output of the network for a given set of inputs.
    """
    outputs = forward_propagate(network, inputs)
    return max(outputs)


def predict_class(network: Network, inputs: List[float]) -> int:
    """
    Predict the class of the output of the network for a given set of inputs.
    """
    outputs = forward_propagate(network, inputs)
    return outputs.index(max(outputs))


if __name__ == "__main__":
    print("Initializing network...")
    # dataset for XOR
    dataset = TrainingDataset(
        rows=[
            TrainingDataRow(inputs=[0, 0], expected=[0]),
            TrainingDataRow(inputs=[0, 1], expected=[1]),
            TrainingDataRow(inputs=[1, 0], expected=[1]),
            TrainingDataRow(inputs=[1, 1], expected=[0]),
        ]
    )

    network = init_network(dataset.num_inputs, 4, dataset.num_outputs)
    network = train_network(network, dataset, 0.2, 200)

    for data in dataset.rows:
        prediction = predict_class(network, data.inputs)
        print(f"Prediction: {prediction}, Expected: {data.expected}")
