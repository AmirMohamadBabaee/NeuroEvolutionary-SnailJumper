import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        layer_sizes_pairs = list(zip(layer_sizes[1:], layer_sizes[:-1]))
        self.parameters = {}
        self.layer_num = len(layer_sizes_pairs)
        for i in range(self.layer_num):
            pair = layer_sizes_pairs[i]
            self.parameters[f'W_{i+1}'] = np.random.normal(0, 1, pair)
            self.parameters[f'b_{i+1}'] = np.zeros((pair[0], 1))

    def activation(self, x, type='sigmoid'):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        if type == 'sigmoid':
            return 1/(1 + np.exp(-x))

        elif type == 'relu':
            return (abs(x) + x)/2

        elif type == 'tanh':
            return np.tanh(x)

    def forward(self, x, activation_type='sigmoid'):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        self.cache = {}
        self.cache['a_0'] = x
        for i in range(self.layer_num):
            self.cache[f'a_{i+1}'] = self.activation(self.parameters[f'W_{i+1}'] @ self.cache[f'a_{i}'] + self.parameters[f'b_{i+1}'], type=activation_type)
        return self.cache[f'a_{self.layer_num}']
