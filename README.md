# Predictive-Coding
A full implementation of a predictive coding model, with built-in methods and options for clamping and settling as well as support for any activation function.

# What is Predictive Coding?

Predictive Coding is a biologically plausible model for how learning works in humans. It is an energy based model, similar to Restricted Boltzmann Machines. Unlike with Artificial Neural Networks which use backpropogation for gradient descent, Predictive Coding networks update each of their nodes locally. Prediction is done by allowing the model to 'settle' given a certain set of constraints and training is done by updating weights between nodes once the model is settled. For a more detailed explanation, please see the 'Links' section.

# Links

Based on the explanation given in this video: https://www.youtube.com/watch?v=l-OLgbdZ3kk

For further reading and understanding, I recommend this github repo: https://github.com/BerenMillidge/Predictive_Coding_Papers
