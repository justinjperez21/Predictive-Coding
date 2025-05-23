from neuron import *
import numpy as np

class PC_Layer():
    def __init__(self, width, output_width, activations=(lambda x: x, lambda x: 1), clamped=False, update_rate=0.1):
        """
        Initialize a Layer instance.
        :param width: Number of neurons in the layer.
        :param output_width: Number of outputs this layer will have. Usually the number of neurons in the next layer.
        """
        assert isinstance(width, int) and width > 0, "Width must be a positive integer."
        assert isinstance(output_width, int) and output_width >= 0, "Output width must be a non-negative integer."
        assert isinstance(activations, tuple) and len(activations) == 2, "Activations must be a tuple of two functions."
        assert callable(activations[0]), "Activation function must be callable."
        assert callable(activations[1]), "Activation function must be callable."
        assert isinstance(clamped, bool), "Clamped must be a boolean."

        self.neurons = [PC_Neuron(output_width, activations=activations, update_rate=update_rate) for _ in range(width)]
        self.clamped = clamped

    def update_error(self):
        """
        Update the error for each neuron in the layer.
        """
        for neuron in self.neurons:
            neuron.update_error()
    
    def update_prediction(self, inputs):
        """
        Update the prediction for each neuron in the layer based on the inputs.
        :param inputs: 2d array of outputs from the previous layer of neurons.
        """
        for i, neuron in enumerate(self.neurons):
            curr_inputs = [next_layer_neuron_output[i] for next_layer_neuron_output in inputs]
            neuron.update_prediction(curr_inputs)

    def update_activity(self, next_layer_errors):
        """
        Update the activity for each neuron in the layer based on the errors from the next layer.
        :param next_layer_errors: Errors from the next layer of neurons.
        """
        if self.clamped:
            return
        for neuron in self.neurons:
            neuron.update_activity(next_layer_errors) #next_layer_errors[i] is the error from neuron i in the next layer
    
    def reset_activity(self):
    	for neuron in self.neurons:
            neuron.reset_activity()
    
    def update_weights(self, next_layer_errors):
        """
        Update the weights for each neuron in the layer based on the errors from the next layer.
        :param next_layer_errors: Errors from the next layer of neurons.
        """
        for neuron in self.neurons:
            neuron.update_weights(next_layer_errors) #next_layer_errors[i] is the error from neuron i in the next layer
    
    def get_output(self):
        """
        Get the output of the layer by getting the output of each neuron.
        :return: list of arrays of outputs from each neuron in the layer. Element i is the output array of neuron i.
        """
        return np.array([neuron.get_output() for neuron in self.neurons])
    
    def get_errors(self):
        """
        Get the errors of the layer by getting the error of each neuron.
        :return: 1d array of errors from each neuron in the layer. Element i is the error of neuron i.
        """
        return np.array([neuron.prediction_error for neuron in self.neurons])
    
    def get_energies(self):
        """
        Get the energies of the layer by getting the energy of each neuron.
        :return: 1d array of energies from each neuron in the layer. Element i is the energy of neuron i.
        """
        return np.array([neuron.prediction_error**2 for neuron in self.neurons])
    
    def get_activity(self):
        """
        Get the activity of the layer by getting the activity of each neuron.
        :return: 1d array of activity values for each neuron in the layer. Element i is the activity of neuron i.
        """
        return np.array([neuron.activity for neuron in self.neurons])
    
    def set_activity(self, activity):
        """
        Set the activity of the layer to a given value.
        :param activity: 1d array of activity values for each neuron in the layer.
        """
        for neuron, activity in zip(self.neurons, activity):
            neuron.activity = activity
            neuron.update_error()
