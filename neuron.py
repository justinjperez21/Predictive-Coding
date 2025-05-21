import numpy as np

class PC_Neuron:
    def __init__(self, num_weights, activations=(lambda x: x, lambda x: 1), update_rate=0.1):
        """
        Initialize a Neuron instance.
        :param num_outputs: Number of outputs this neuron will have. Usually the number of neurons in the next layer.
        :param activations: A tuple of two functions. The first function is the activation function, and the second function is the derivative of the activation function.
        """
        assert isinstance(num_weights, int) and num_weights >= 0, "Number of weights must be a non-negative integer."
        assert isinstance(activations, tuple) and len(activations) == 2, "Activations must be a tuple of two functions."
        assert callable(activations[0]), "Activation function must be callable."
        assert callable(activations[1]), "Activation function must be callable."

        self.activations = activations
        self.activity = np.random.uniform(-1, 1)
        self.prediction = 0
        self.update_error()
        self.weights = np.random.uniform(-1, 1, num_weights)
        self.update_rate = update_rate

    def update_error(self):
        """
        Update the neuron's error based on its activity and prediction.
        """
        self.prediction_error = self.activity - self.prediction

    def update_prediction(self, inputs):
        """
        Update the neuron's prediction based on the inputs and weights.
        :param inputs: Outputs from the previous layer of neurons.
        """
        self.prediction = np.sum(inputs)
        self.update_error()

    def update_activity(self, next_layer_errors):
        """
        Update the neuron's activity based on the errors from the next layer and its prediction error.
        :param next_layer_errors: 1d array of errors from the next layer of neurons.
        """
        self.activity += self.update_rate * (-self.prediction_error + np.sum(next_layer_errors * self.weights * self.activations[1](self.activity)))
        self.update_error()

    def update_weights(self, next_layer_errors):
        """
        Update the neuron's weights based on the errors from the next layer and its activity.
        :param next_layer_errors: Errors from the next layer of neurons.
        """
        self.weights += next_layer_errors * self.activations[0](self.activity)

    def get_output(self):
        """
        Get the output of the neuron based on its activation function and weights.
        """
        return self.weights * self.activations[0](self.activity)