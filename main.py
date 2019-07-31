# Neural Network - Simple example

import numpy as np

# Sigmoid function
def sigmoid(x):
    # activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # 1/n * SUM((y_true - y_pred)^2)
    return ((y_true - y_pred)**2).mean()


class Neuron:

    # init of weights
    def __init__(self, weights, bias, name, layer):
        self.weights = weights
        self.bias = bias
        self.layer = layer
        self.name = "L{}N{}".format(str(layer), str(name))
        
    # maths operations
    def feedforward(self, inputs):
        # { (inputs * weights) + bias }, then use activation function
        #print(self.weights, inputs, self.bias)
        tot = np.dot(self.weights, inputs) + self.bias
        return sigmoid(tot)
    
    def printInfo(self):
        print("\nName:", self.name)
        print("Layer:", self.layer)
        print("Weights:", self.weights)
        print("Bias:", self.bias)


class NeuralNetwork:
    """
    Neural network with:
        - 2 inputs data
        - 1 hidden layer with 2 neurons (h1,h2)
        - 1 output layer with 1 neuron (o1)
    """
    
    def __init__(self, layers):
        
        self.layers = layers
        self.num_neurons = []
        self.weights = []
        self.bias = []
        self.neurons = []
        for layer in range(layers):
            print("How many Neurons in Layer " + str(layer) + " ?")
            num = int(input("-> "))
            self.num_neurons.append(num)
              
            # init Weights, Biases and Neurons
            k=0
            for i in range(num):
               self.weights.append(np.random.normal())
               self.bias.append(np.random.normal())
               neur = Neuron(self.weights[k:k+2], self.bias[i], i, layer)
               self.neurons.append(neur)
               k+=2
        self.printInfo()
           
    def feedforward(self, x):
        out_h1 = self.neurons[0].feedforward(x)
        out_h2 = self.neurons[1].feedforward(x)
        
        #out_h1 and out_h2 are inputs for o1 neuron
        out_o1 = self.neurons[2].feedforward([out_h1, out_h2])
        return out_o1

    def printInfo(self):
        print("\n--- Network Info ---")
        print("Weights:", self.weights)
        print("Bias:", self.bias)
        
        for neuron in self.neurons:
            neuron.printInfo()

def basicNetwork():
    print("\n\n--- EXAMPLE 2: basic network ---\n")
    
    # Building the Network
    network = NeuralNetwork(2)
    
    x = np.array([2,3])
    print("\ninput_array:", x)
    print("output_value:", network.feedforward(x))    
    
    
# --- TEST MSE ---  
def test_mse():
    print("\n\n--- EXAMPLE 3: test MSE Loss ---\n")
    
    y_true = np.array([1,0,0,1])
    y_pred = np.array([0,0,0,0])

    print("MSE Loss:", mse_loss(y_true, y_pred))
    
    
# --- MAIN ---
    
#test_mse()
basicNetwork()
