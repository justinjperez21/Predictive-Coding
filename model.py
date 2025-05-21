from layer import *

class PC_Model():
    def __init__(self, layer_widths, activations=(lambda x: x, lambda x: 1), clamped_layers=[], update_rate=0.01):
        """
        Initialize a PC_Model instance.
        :param layer_widths: List of widths in number of neurons for each layer(positive integers). The length of this list will determine the number of layers in the model. List should begin with most abstract layer and end with input layer.
        :param activation_function: A function to apply to the neuron's activity. Must map 1 number to 1 number.
        :param clamped_layers: List of layer indices that are clamped. Clamped layers will not update their activity. Their activity will be set to the input data. 
        """
        assert len(layer_widths) > 1, "Model must have at least 2 layers."
        assert all(isinstance(x, int) and x > 0 for x in layer_widths), "Layer widths must be positive integers."
        assert all(isinstance(x, int) and x >= 0 and x < len(layer_widths) for x in clamped_layers), "Clamped layers must be a list of indices of the layers in the model."
        assert isinstance(activations, tuple) and len(activations) == 2, "Activations must be a tuple of two functions."
        assert callable(activations[0]), "Activation function must be callable."
        assert callable(activations[1]), "Activation function must be callable."

        self.layers = []
        for i in range(len(layer_widths) - 1):
            clamped = i in clamped_layers
            self.layers.append(PC_Layer(layer_widths[i], layer_widths[i+1], activations=activations, clamped=clamped, update_rate=update_rate))
        self.layers.append(PC_Layer(layer_widths[-1], 0, activations=activations, clamped=True, update_rate=update_rate)) # Last layer has no outputs

        self.update_prediction() # Initialize predictions to their correct values

    def update_error(self):
        """
        Update the error for each layer in the model.
        """
        for layer in self.layers:
            layer.update_error()
    
    def update_prediction(self):
        """
        Update the prediction for each layer in the model based on the inputs.
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            layer.update_prediction(self.layers[i-1].get_output())
    
    def update_activity(self):
        """
        Update the activity for each layer in the model based on the errors from the next layer.
        """
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                continue
            layer.update_activity(self.layers[i+1].get_errors())
    
    def update_weights(self):
        """
        Update the weights for each layer in the model based on the errors from the next layer.
        """
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                continue
            layer.update_weights(self.layers[i+1].get_errors())

    def get_error(self):
        """
        Get the errors for each layer in the model.
        :return: summed total error of the model.
        """
        return np.sum([np.sum(layer.get_errors()) for layer in self.layers])

    def clamp_layer(self, layer_index, data=None):
        """
        Clamp a layer to a specific activity.
        :param layer_index: Index of the layer to clamp.
        :param data: Data to set the activity of the clamped layer. If None, the layer will be clamped to its current activity.
        """
        assert isinstance(layer_index, int) and layer_index >= 0 and layer_index < len(self.layers), "Layer index must be a valid index."
        assert isinstance(data, (np.ndarray, list)) or data is None, "Data must be a numpy array or list."
        assert len(data) == self.layers[layer_index].width if data is not None else True, "Data must have the same length as the layer width."
        self.layers[layer_index].clamped = True

        if data is not None:
            self.layers[layer_index].set_activity(data)
    
    def unclamp_layer(self, layer_index):
        """
        Unclamp a layer.
        :param layer_index: Index of the layer to unclamp.
        """
        assert isinstance(layer_index, int) and layer_index >= 0 and layer_index < len(self.layers), "Layer index must be a valid index."
        self.layers[layer_index].clamped = False
    
    def set_activity(self, layer_index, activity):
        """
        Set the activity of a layer to a given value.
        :param layer_index: Index of the layer to set the activity for.
        :param activity: 1d array of activity values for each neuron in the layer.
        """
        assert isinstance(layer_index, int) and layer_index >= 0 and layer_index < len(self.layers), "Layer index must be a valid index."
        assert isinstance(activity, (np.ndarray, list)), "Activity must be a numpy array or list."
        assert len(activity) == len(self.layers[layer_index].neurons), "Activity must have the same length as the layer width."
        self.layers[layer_index].set_activity(activity)
    
    def get_activity(self, layer_index):
        """
        Get the activity of a layer.
        :param layer_index: Index of the layer to get the activity for.
        :return: 1d array of activity values for each neuron in the layer.
        """
        assert isinstance(layer_index, int) and layer_index >= 0 and layer_index < len(self.layers), "Layer index must be a valid index."
        return self.layers[layer_index].get_activity()
    
    def settle(self, settle_ratio=1e-10):
        """
        Settle the model by updating the activity and predictions until the energy change is below a certain ratio.
        :param settle_ratio: Ratio change in energy to determine when to stop settling.
        """
        curr_energy = self.get_error()**2
        prev_energy = float('inf')
        while abs(1 - curr_energy/prev_energy) > settle_ratio:
            print(curr_energy)
            prev_energy = curr_energy
            self.update_activity()
            self.update_prediction()
            curr_energy = self.get_error()**2
        print(curr_energy)

    def settle_weights(self, settle_ratio=1e-10):
        """
        Settle the model by updating the weights until the energy change is below a certain ratio.
        :param settle_ratio: Ratio change in energy to determine when to stop settling.
        """
        curr_energy = self.get_error()**2
        prev_energy = float('inf')
        while abs(1 - curr_energy/prev_energy) > settle_ratio:
            print(curr_energy)
            prev_energy = curr_energy
            self.update_weights()
            curr_energy = self.get_error()**2
        print(curr_energy)