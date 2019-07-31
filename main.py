# Neural Network - Simple example

import numpy as np

# Sigmoid function
def sigmoid(x):
    # activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:

    # init of weights 
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    # maths operations
    def feedforward(self, inputs):
        # { (inputs * weights) + bias }, then use activation function
        tot = np.dot(self.weights, inputs) + self.bias
        return sigmoid(tot)
    

class NeuralNetwork:
    """
    Neural network with:
        - 2 inputs
        - 1 hidden layer with 2 neurons (h1,h2)
        - 1 output layer with 1 neuron (o1)
        
    Each neuron has:
        weights = [0,1]
        bias = 0
    """
    
    def __init__(self):
        weights = np.array([0,1])
        bias = 0
        
        # Network
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
        
        
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        
        #out_h1 and out_h2 are inputs for o1 neuron
        out_o1 = self.o1.feedforward([out_h1, out_h2])
        return out_o1
    
    
def mse_loss(y_true, y_pred):
    # 1/n * SUM((y_true - y_pred)^2)
    return ((y_true - y_pred)**2).mean()


def singleNeuron():
    print("\n\n--- EXAMPLE 1: single neuron ---\n")
    
    # Building the Neuron
    weights = np.array([0,1])
    bias = 4
    n1 = Neuron(weights, bias)
    
    x = np.array([2,3])
    print("weights:", weights)
    print("bias:", bias)
    print("input_array:", x)
    print("output_value:", n1.feedforward(x))


def manualNetwork():
    print("\n\n--- EXAMPLE 2: manual network ---\n")
    
    # Building the Network
    network = NeuralNetwork()
    
    x = np.array([2,3])
    print("input_array:", x)
    print("output_value:", network.feedforward(x))    
    
    
def test_mse():
    print("\n\n--- EXAMPLE 3: test MSE Loss ---\n")
    
    y_true = np.array([1,0,0,1])
    y_pred = np.array([0,0,0,0])

    print("MSE Loss:", mse_loss(y_true, y_pred))
    
    
# --- MAIN ---

singleNeuron()
manualNetwork()
test_mse()