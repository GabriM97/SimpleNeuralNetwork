# Neural Network - Simple example

import numpy as np

# SIGMOID FUNCTION
def sigmoid(x):
    # activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

# SIGMOID DERIVATIVE FUNCTION
def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

# MEDIUM SQUARE ERROR LOSS
def mse_loss(y_true, y_pred):
    # 1/n * SUM((y_true - y_pred)^2)
    return ((y_true - y_pred)**2).mean()


# --- NEURON CLASS ---
class Neuron:

    # init of weights
    def __init__(self, weights, bias, name, layer):
        self.weights = weights
        self.bias = bias
        self.h_layer = layer
        self.name = "L{}N{}".format(str(layer), str(name))
        
    # maths operations
    def feedforward(self, inputs):
        # { (inputs * weights) + bias }, then use activation function
        tot = np.dot(self.weights, inputs) + self.bias
        return sigmoid(tot)
    
    def setName(self, name):
        self.name = name
        
    def printInfo(self):
        print("\nName:", self.name)
        print("H_Layer:", self.h_layer)
        print("Weights:", self.weights)
        print("Bias:", self.bias)


# --- NEURAL NETWORK --- 
class NeuralNetwork:
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        
        #self.init_output()
        self.output_num = 1
        #self.init_hidden_layers()
        self.h_layers = 1
        
        self.init_neurons()
        self.init_output_neuron()
        self.printInfo()
       
    def feedforward(self, data):
        self.out_h = []
        self.outputs = []
        
        print("\n--- FEED FORWARD ---")
        idx = 0
        for x in data:
            self.out_h = x
            for layer in range(self.h_layers):
                print("\nLayer", layer)
                n_neur = self.num_neurons[layer+1]
                print("Num_Neurons:", n_neur)
                result = []
                for neuron in self.neurons[idx:n_neur]:
                    result.append(neuron.feedforward(self.out_h))
                self.out_h = result
                print("out_h:", self.out_h)
                idx += n_neur
            
            print("\nOUTPUT LAYER")
            result = []
            for neuron in self.neurons[-self.output_num:]:
                result.append(neuron.feedforward(self.out_h))
            self.outputs = result
            return self.outputs
    
    # INITIALIZATION FUNCTIONS
    def init_output(self):
        print("\nHow many Output?")
        self.output_num = int(input("-> "))
        
    def init_hidden_layers(self):
        print("\nHow many Hidden Layers?")
        self.h_layers = int(input("-> "))
        
    def init_neurons(self):
        self.weights = []
        self.bias = []
        self.num_neurons = []
        self.num_neurons.append(self.input_dim) # pos[0] = input_dimension
        self.neurons = []
        for layer in range(self.h_layers):
            print("\nHow many Neurons in Layer" + str(layer+1) + "?")
            num = int(input("-> "))
            self.num_neurons.append(num)
            
            n_prev_neur = self.num_neurons[layer] # number of neurons in previously layer
            for neuron in range(num):
                # init Weights, Biases
                for weight in range(n_prev_neur):
                    self.weights.append(np.random.normal())
                self.bias.append(np.random.normal())
                
                # init Neuron
                neur = Neuron(self.weights[-n_prev_neur:], self.bias[-1], neuron, layer)
                self.neurons.append(neur)
    
    def init_output_neuron(self):
        for output in range(self.output_num):
            # init Weights, Biases
            n_prev_neur = self.num_neurons[-1]
            for neuron in range(n_prev_neur):
                self.weights.append(np.random.normal())
            self.bias.append(np.random.normal())
            
            # init Neuron
            neur = Neuron(self.weights[-n_prev_neur:], self.bias[-1], neuron, len(self.num_neurons)-1)
            neur.setName("O"+str(output))
            self.neurons.append(neur)

    # PRINT INFO OF NETWORK  
    def printInfo(self):
        print("\n--- Network Info ---")
        print("\nInput_Dimension:", self.input_dim)
        print("Num_of_Outputs:", self.output_num)
        print("Num_of_Hidden_Layers:", self.h_layers)
        for layer in range(self.h_layers):
            print("\n -H_Layer " + str(layer+1) + ":")
            print("Neurons:", self.num_neurons[layer+1])
        
        print("\n")
        print("Weights:", self.weights)
        print("Bias:", self.bias, "\n")
        
        index = 0
        for layer in range(self.h_layers):
            print("\n\n -- #LAYER " + str(layer+1) + " --")
            num_neurons_per_layer = self.num_neurons[layer+1]
            for neuron in self.neurons[index:index+num_neurons_per_layer]:
                neuron.printInfo()
            index += num_neurons_per_layer
        
        print("\n\n -- #OUTPUT LAYER --")
        for neuron in self.neurons[-self.output_num:]:
            neuron.printInfo()

    # TRAINING FUNCTION
    def train(self, data, y_trues):
        print("training")
        

# --- MAIN METHODS ---

def basicNetwork():
    print("\n\n--- EXAMPLE 2: basic network ---\n")
    
    data = np.array([
      [-2, -1],  # Alice
      [25, 6],   # Bob
      [17, 4],   # Charlie
      [-15, -6], # Diana
    ])
    
    y_trues = np.array([
      1, # Alice
      0, # Bob
      0, # Charlie
      1, # Diana
    ])
    
    # Building the Network
    input_dim = 2
    network = NeuralNetwork(input_dim)
    
    print("output_value:", network.feedforward(data))
    
    
# --- TEST MSE ---  
def test_mse():
    print("\n\n--- EXAMPLE 3: test MSE Loss ---\n")
    
    y_true = np.array([1,0,0,1])
    y_pred = np.array([0,0,0,0])

    print("MSE Loss:", mse_loss(y_true, y_pred))
    
    
# --- MAIN ---
    
#test_mse()
basicNetwork()
