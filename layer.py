import numpy as np

from neuron import PC_Neuron # Corrected import statement


class PC_Layer():
    def __init__(self, width: int, output_width: int, activations: tuple = (lambda x: x, lambda x: 1), 
                 clamped: bool = False, update_rate: float = 0.1):
        """
        Initialize a Layer instance.

        :param width: Number of neurons in the layer.
        :param output_width: Number of outputs this layer will have (number of neurons in the next layer).
        :param activations: A tuple of two functions: the activation function and its derivative.
        :param clamped: Boolean indicating if the layer's activity is fixed.
        :param update_rate: The learning rate for updating neuron activity.
        """
        assert isinstance(width, int) and width > 0, "Width must be a positive integer."
        assert isinstance(output_width, int) and output_width >= 0, \
            "Output width must be a non-negative integer."
        assert isinstance(activations, tuple) and len(activations) == 2, \
            "Activations must be a tuple of two functions."
        assert callable(activations[0]), "Activation function must be callable."
        assert callable(activations[1]), "Derivative of activation function must be callable."
        assert isinstance(clamped, bool), "Clamped must be a boolean."

        self.neurons = [PC_Neuron(output_width, activations=activations, update_rate=update_rate) 
                        for _ in range(width)]
        self.clamped = clamped

    def update_error(self) -> None:
        """
        Update the prediction error for each neuron in the layer.
        """
        for neuron in self.neurons:
            neuron.update_error()
    
    def update_prediction(self, inputs: np.ndarray) -> None:
        """
        Update the prediction for each neuron in the layer.

        :param inputs: 2D NumPy array of outputs from the previous layer.
                       Shape: (num_neurons_previous_layer, num_neurons_current_layer).
                       inputs[:, i] provides the vector of inputs to neuron i in this layer.
        """
        if inputs is None or inputs.size == 0: 
            # This can happen for the first layer if not handled by PC_Model
            # or if a layer is designed to not receive inputs.
            # Each neuron's prediction would be based on its internal state or zero.
            for neuron in self.neurons:
                neuron.update_prediction(np.array([])) # Or handle as per model design
            return

        for i, neuron in enumerate(self.neurons):
            # Each neuron i in this layer receives inputs from all neurons in the previous layer.
            # The weights of neuron i determine how it combines these inputs.
            # Here, inputs[:, i] is the set of signals passed to neuron i from the previous layer.
            neuron.update_prediction(inputs[:, i])

    def update_activity(self, next_layer_errors: np.ndarray) -> None:
        """
        Update the activity for each non-clamped neuron in the layer.

        :param next_layer_errors: 1D NumPy array of error signals from the next layer.
                                   Each element `j` is the error signal from neuron `j` in the next layer
                                   that neurons in this layer project to.
        """
        if self.clamped:
            return
        
        # Each neuron in this layer updates its activity based on the error signals 
        # propagated back from the next layer. The exact way `next_layer_errors`
        # is used by each neuron depends on its `update_activity` method,
        # particularly how it uses its outgoing weights.
        for neuron in self.neurons:
            neuron.update_activity(next_layer_errors) 
    
    def reset_activity(self) -> None:
        """
        Reset the activity for each neuron in the layer.
        """
        for neuron in self.neurons:
            neuron.reset_activity()
    
    def update_weights(self, next_layer_errors: np.ndarray) -> None:
        """
        Update the weights for each neuron in the layer.

        :param next_layer_errors: 1D NumPy array of error signals from the next layer.
                                   Each element `j` is the error signal from neuron `j` in the next layer.
        """
        # Each neuron updates its weights based on its own activity and the errors from the next layer.
        for neuron in self.neurons:
            neuron.update_weights(next_layer_errors)
    
    def get_output(self) -> np.ndarray:
        """
        Get the output of the layer.
        The output of a layer is a collection of outputs from all its neurons.
        Each neuron's output is typically a vector (weights * activity).

        :return: 2D NumPy array. Shape: (num_neurons_current_layer, num_neurons_next_layer).
                 Element (i, j) is the output from neuron `i` in this layer to neuron `j` in the next.
        """
        return np.array([neuron.get_output() for neuron in self.neurons])
    
    def get_errors(self) -> np.ndarray:
        """
        Get the prediction errors of all neurons in the layer.

        :return: 1D NumPy array. Element `i` is the prediction error of neuron `i`.
        """
        return np.array([neuron.prediction_error for neuron in self.neurons])
    
    def get_energies(self) -> np.ndarray:
        """
        Get the energies (squared prediction error) for all neurons in the layer.

        :return: 1D NumPy array. Element `i` is the energy of neuron `i`.
        """
        return np.array([neuron.prediction_error**2 for neuron in self.neurons])
    
    def get_activity(self) -> np.ndarray:
        """
        Get the activity of all neurons in the layer.

        :return: 1D NumPy array. Element `i` is the activity of neuron `i`.
        """
        return np.array([neuron.activity for neuron in self.neurons])
    
    def set_activity(self, activity: np.ndarray) -> None:
        """
        Set the activity of the neurons in the layer.

        :param activity: 1D NumPy array of activity values. Must match the number of neurons.
        """
        assert isinstance(activity, np.ndarray), "Activity must be a NumPy array."
        assert len(activity) == len(self.neurons), \
            "Activity array length must match the number of neurons."
            
        for neuron, act_val in zip(self.neurons, activity):
            neuron.activity = act_val
            neuron.update_error() # Activity change affects prediction error
