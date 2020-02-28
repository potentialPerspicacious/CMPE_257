1. Building a NN from scratch. 

2. Neural network will have 4 inputs, one hidden layer with n hidden units (where n is a parameter of your program), and 3 output units. The hidden and output units should use the sigmoid activation function. The network should be fully connected — that is, every input unit connects to every hidden unit, and every hidden unit connects to every output unit. Every hidden and output unit also has a weighted connection from a bias unit, whose value is set to 1.

3. Task: Each output unit corresponds to one of the 3 classes. Set the target value tk for output unit k to 0.9 if the input class is the kth class, 0.1 otherwise.

4. Network classification: An example x is propagated forward from the input to the output. The class predicted by the network is the one corresponding to the most highly activated output unit. The activation function for each hidden and output unit is the sigmoid function.

5. Network training: Use back-propagation with stochastic gradient descent to train the network. Set the training rate to 0.01.

6. Preprocessing: Need to pre-process the data so that the input dimensions have similar scales.
Initial weights: Your network should start off with small (−.05 < θ < .05) random positive and negative weights.
