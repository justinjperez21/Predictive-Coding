import numpy as np


class PC_Neuron:
    def __init__(self, num_weights: int, activations: tuple = (lambda x: x, lambda x: 1), update_rate: float = 0.1):
        """
        Initialize a Neuron instance.

        :param num_weights: Number of neurons in the next layer this neuron projects to.
        :param activations: A tuple of two functions. The first function is the activation function, 
                            and the second function is the derivative of the activation function.
        """
        assert isinstance(num_weights, int) and num_weights >= 0, \
            "Number of weights must be a non-negative integer."
        assert isinstance(activations, tuple) and len(activations) == 2, \
            "Activations must be a tuple of two functions."
        assert callable(activations[0]), "Activation function must be callable."
        assert callable(activations[1]), "Derivative of activation function must be callable."

        self.activations = activations
        self.activity = np.random.uniform(-1, 1)
        self.prediction = 0
        self.weights = np.random.uniform(-1, 1, num_weights)
        self.update_rate = update_rate
        self.update_error()  # Initialize prediction_error

    def update_error(self) -> None:
        """
        Update the neuron's error based on its activity and prediction.
        """
        self.prediction_error = self.activity - self.prediction

    def update_prediction(self, inputs: np.ndarray) -> None:
        """
        Update the neuron's prediction based on the inputs.
        Note: In the PC model, 'prediction' is typically the sum of inputs from the layer below.
        
        :param inputs: 1D NumPy array of inputs from the previous layer of neurons.
        """
        self.prediction = np.sum(inputs)
        self.update_error()

    def update_activity(self, next_layer_errors: np.ndarray) -> None:
        """
        Update the neuron's activity. 
        This is based on the errors from the next layer and its own prediction error.
        The update rule aims to reduce both the neuron's own prediction error and
        the error it contributes to the next layer.

        :param next_layer_errors: 1D NumPy array of error signals from the next layer of neurons.
                                   Each element `j` is the error signal from neuron `j` in the next layer
                                   that this neuron projects to.
        """
        # The derivative of the activation function self.activations[1] is applied to the current activity.
        # This term modulates how much the neuron's activity changes in response to the error signals.
        error_signal_from_next_layer = np.sum(next_layer_errors * self.weights)
        weighted_error_signal = error_signal_from_next_layer * self.activations[1](self.activity)
        
        self.activity += self.update_rate * (-self.prediction_error + weighted_error_signal)
        self.update_error()
        
    def reset_activity(self) -> None:
        """
        Reset the neuron's activity to a random value between -1 and 1.
        """
        self.activity = np.random.uniform(-1, 1)
        self.update_error() # Activity change affects prediction error

    def update_weights(self, next_layer_errors: np.ndarray) -> None:
        """
        Update the neuron's weights.
        The weight update rule is typically a Hebbian-like rule, where weights are adjusted
        based on the neuron's own activity and the error signals from the neurons it projects to.

        :param next_layer_errors: 1D NumPy array of error signals from the next layer of neurons.
                                   Each element `j` is the error signal from neuron `j` in the next layer
                                   that this neuron projects to.
        """
        # The activation function self.activations[0] is applied to the current activity.
        # This means weight updates are proportional to the neuron's output.
        self.weights += self.update_rate * next_layer_errors * self.activations[0](self.activity)

    def get_output(self) -> np.ndarray:
        """
        Get the output of the neuron.
        This is a vector of weighted activations, representing the signals sent to the next layer.
        
        :return: 1D NumPy array, where each element is the product of an outgoing weight
                 and the neuron's current activation.
        """
        return self.weights * self.activations[0](self.activity)
