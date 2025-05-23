import numpy as np

from layer import PC_Layer # Corrected import statement


class PC_Model():
    def __init__(self, layer_widths: list[int], activations: tuple = (lambda x: x, lambda x: 1), 
                 clamped_layers: list[int] = [], update_rate: float = 0.01):
        """
        Initialize a PC_Model instance.

        :param layer_widths: List of widths (number of neurons) for each layer.
                             Ordered from most abstract layer to input layer.
        :param activations: Tuple: (activation_function, derivative_of_activation_function).
        :param clamped_layers: List of indices for layers whose activity is fixed by input data.
        :param update_rate: Learning rate for activity and weight updates.
        """
        assert len(layer_widths) > 1, "Model must have at least 2 layers."
        assert all(isinstance(x, int) and x > 0 for x in layer_widths), \
            "Layer widths must be positive integers."
        assert all(isinstance(x, int) and 0 <= x < len(layer_widths) for x in clamped_layers), \
            "Clamped layers must be a list of valid layer indices."
        assert isinstance(activations, tuple) and len(activations) == 2, \
            "Activations must be a tuple of two functions."
        assert callable(activations[0]), "Activation function must be callable."
        assert callable(activations[1]), "Derivative of activation function must be callable."

        self.layers: list[PC_Layer] = []
        num_model_layers = len(layer_widths)

        # Create hidden layers and the first layer (most abstract)
        for i in range(num_model_layers - 1):
            is_clamped = i in clamped_layers
            # output_width for layer i is the width of layer i+1
            self.layers.append(PC_Layer(width=layer_widths[i], 
                                        output_width=layer_widths[i+1], 
                                        activations=activations, 
                                        clamped=is_clamped, 
                                        update_rate=update_rate))
        
        # Create the last layer (typically the input layer)
        # It has no subsequent layer, so output_width is 0.
        # It's often clamped by default if it's the input layer.
        last_layer_index = num_model_layers - 1
        is_last_layer_clamped = last_layer_index in clamped_layers or True # Default to clamped if input
        self.layers.append(PC_Layer(width=layer_widths[last_layer_index], 
                                    output_width=0, 
                                    activations=activations, 
                                    clamped=is_last_layer_clamped, 
                                    update_rate=update_rate))

        self.update_prediction() # Initialize predictions throughout the model

    def update_error(self) -> None:
        """
        Update the prediction error for each layer in the model.
        """
        for layer in self.layers:
            layer.update_error()
    
    def update_prediction(self) -> None:
        """
        Update predictions for all layers.
        The prediction for layer `i` is based on the output of layer `i-1`.
        The first layer (index 0) does not have a preceding layer in this hierarchy,
        so its predictions are typically based on its own state or are not updated here.
        """
        for i, current_layer in enumerate(self.layers):
            if i == 0:  # First layer's predictions are intrinsic or handled differently
                # Potentially, the first layer's neurons might predict their own activity
                # or a baseline if they are not input-driven in a specific way.
                # For now, we assume its predictions are updated internally if needed.
                pass # Or current_layer.update_prediction(None) if designed to handle it
            else:
                previous_layer_output = self.layers[i-1].get_output()
                current_layer.update_prediction(previous_layer_output)
    
    def update_activity(self) -> None:
        """
        Update activity for all non-clamped layers.
        Activity for layer `i` is updated based on error signals from layer `i+1`.
        The last layer (index N-1) does not have a subsequent layer (N),
        so its activity update is not driven by errors from a "next" layer in this loop.
        """
        for i, current_layer in enumerate(self.layers):
            if current_layer.clamped:
                continue # Skip clamped layers

            if i < len(self.layers) - 1:  # If there is a next layer
                next_layer_errors = self.layers[i+1].get_errors()
                current_layer.update_activity(next_layer_errors)
            else:
                # Last layer has no "next_layer_errors" to receive in this manner.
                # Its activity might be updated based on its own prediction error if not clamped.
                # This is implicitly handled by the neuron's update_activity if it uses its own error.
                # For now, we assume update_activity is called appropriately.
                # If it requires next_layer_errors, it simply won't be called for the last layer here.
                 pass


    def reset_activity(self) -> None:
        """
        Reset activity for all layers in the model.
        """
        for layer in self.layers:
            layer.reset_activity()
    
    def update_weights(self) -> None:
        """
        Update weights for all layers that have a subsequent layer.
        Weights connecting layer `i` to layer `i+1` are updated based on
        error signals from layer `i+1`.
        """
        for i, current_layer in enumerate(self.layers):
            if i < len(self.layers) - 1:  # If there is a next layer to provide errors
                next_layer_errors = self.layers[i+1].get_errors()
                current_layer.update_weights(next_layer_errors)

    def get_error(self) -> float:
        """
        Calculate the total absolute prediction error across all layers.

        :return: Sum of absolute prediction errors.
        """
        total_error = 0.0
        for layer in self.layers:
            total_error += np.sum(np.abs(layer.get_errors()))
        return total_error
    
    def get_energy(self) -> float:
        """
        Calculate the total energy (sum of squared prediction errors) across all layers.

        :return: Sum of squared prediction errors.
        """
        total_energy = 0.0
        for layer in self.layers:
            total_energy += np.sum(layer.get_energies())
        return total_energy

    def clamp_layer(self, layer_index: int, data: np.ndarray = None) -> None:
        """
        Clamp a layer, fixing its activity (optionally to provided data).

        :param layer_index: Index of the layer to clamp.
        :param data: Optional 1D NumPy array to set the activity of the clamped layer.
                     Length must match the number of neurons in the layer.
        """
        assert isinstance(layer_index, int) and 0 <= layer_index < len(self.layers), \
            "Layer index must be a valid index."
        
        target_layer = self.layers[layer_index]
        target_layer.clamped = True

        if data is not None:
            assert isinstance(data, np.ndarray), "Data must be a NumPy array."
            assert data.ndim == 1, "Data must be a 1D array."
            assert len(data) == len(target_layer.neurons), \
                "Data length must match the number of neurons in the layer."
            target_layer.set_activity(data)
    
    def unclamp_layer(self, layer_index: int) -> None:
        """
        Unclamp a layer, allowing its activity to be updated dynamically.

        :param layer_index: Index of the layer to unclamp.
        """
        assert isinstance(layer_index, int) and 0 <= layer_index < len(self.layers), \
            "Layer index must be a valid index."
        self.layers[layer_index].clamped = False
    
    def set_activity(self, layer_index: int, activity: np.ndarray) -> None:
        """
        Set the activity of a specific layer.

        :param layer_index: Index of the layer.
        :param activity: 1D NumPy array of activity values.
        """
        assert isinstance(layer_index, int) and 0 <= layer_index < len(self.layers), \
            "Layer index must be a valid index."
        assert isinstance(activity, np.ndarray), "Activity must be a NumPy array."
        assert activity.ndim == 1, "Activity must be a 1D array."
        assert len(activity) == len(self.layers[layer_index].neurons), \
            "Activity array length must match the number of neurons in the layer."
            
        self.layers[layer_index].set_activity(activity)
    
    def get_activity(self, layer_index: int) -> np.ndarray:
        """
        Get the activity of a specific layer.

        :param layer_index: Index of the layer.
        :return: 1D NumPy array of activity values.
        """
        assert isinstance(layer_index, int) and 0 <= layer_index < len(self.layers), \
            "Layer index must be a valid index."
        return self.layers[layer_index].get_activity()
    
    def settle(self, settle_ratio: float = 1e-6, max_iterations: int = 100) -> None:
        """
        Iteratively update activities and predictions until the network settles.
        Settling occurs when the relative change in total energy is below `settle_ratio`
        or `max_iterations` is reached.

        :param settle_ratio: Relative energy change threshold for convergence.
        :param max_iterations: Maximum number of iterations to prevent infinite loops.
        """
        prev_energy = self.get_energy()
        
        for i in range(max_iterations):
            self.update_activity()
            self.update_prediction()
            current_energy = self.get_energy()
            
            # print(f"Settle Iteration {i+1}: Energy = {current_energy}") # For debugging

            if prev_energy == 0: # Avoid division by zero if energy starts at 0
                if current_energy == 0: # If still zero, it's stable
                    break
                else: # If energy increased from zero, use a small factor for change
                    relative_change = settle_ratio + 1 
            else:
                relative_change = abs(current_energy - prev_energy) / prev_energy

            if relative_change < settle_ratio:
                # print(f"Network settled after {i+1} iterations.")
                break
            
            prev_energy = current_energy
        # else:
            # print(f"Network did not fully settle after {max_iterations} iterations.")

    def settle_weights(self, settle_ratio: float = 1e-6, max_iterations: int = 50) -> None:
        """
        Iteratively update weights (and dependent predictions/energy) until the network settles.
        Settling occurs when the relative change in total energy is below `settle_ratio`
        or `max_iterations` is reached. Typically, `settle()` for activities
        would be called within the training loop before or after `settle_weights()`.

        :param settle_ratio: Relative energy change threshold for convergence.
        :param max_iterations: Maximum number of iterations.
        """
        prev_energy = self.get_energy()

        for i in range(max_iterations):
            self.update_weights()       # Update weights based on current activities and errors
            self.update_prediction()    # Predictions change due to new weights
            current_energy = self.get_energy() # Recalculate energy

            # print(f"Settle Weights Iteration {i+1}: Energy = {current_energy}") # For debugging

            if prev_energy == 0:
                if current_energy == 0:
                    break
                else:
                    relative_change = settle_ratio + 1 
            else:
                relative_change = abs(current_energy - prev_energy) / prev_energy

            if relative_change < settle_ratio:
                # print(f"Weights settled after {i+1} iterations.")
                break
            
            prev_energy = current_energy
        # else:
            # print(f"Weights did not fully settle after {max_iterations} iterations.")
