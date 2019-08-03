# Neural Network - Simple example

import numpy as np

round_digits = 4

# SIGMOID FUNCTION
def sigmoid(x):
    # activation function: f(x) = 1 / (1 + e^(-x))
    return round(1 / (1 + np.exp(-x)), round_digits)

# SIGMOID DERIVATIVE FUNCTION
def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return round(fx * (1 - fx), round_digits)

# MEDIUM SQUARE ERROR LOSS
def mse_loss(y_true, y_pred):
    # 1/n * SUM((y_true - y_pred)^2)
    return round(((y_true - y_pred)**2).mean(), round_digits)


# --- NEURON CLASS ---
class Neuron:

    # init of weights
    def __init__(self, weights, bias, output_neuron, seq, layer="/"):        
        if output_neuron:
            self.name = "O{}".format(str(seq))
        else:
            self.name = "L{}N{}".format(str(layer), str(seq))
        
        self.h_layer = layer
        self.output_neuron = output_neuron
        self.weights = weights
        self.bias = bias
    
    def setWeights(self, weights):
        self.weights = weights
    
    # maths operations
    def feedforward(self, inputs):
        # { (inputs * weights) + bias }, then use activation function
        tot = np.dot(self.weights, inputs) + self.bias
        return round(tot, round_digits), sigmoid(tot)
        
    def printInfo(self):
        print("\nName:", self.name)
        if not self.output_neuron:
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
        self.h_layers = 2
        
        self.init_neurons()
        self.init_output_neuron()
        self.printInfo()
       
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
                    self.weights.append(round(np.random.normal(), round_digits))
                self.bias.append(round(np.random.normal(), round_digits))
                
                # init Neuron
                neur = Neuron(self.weights[-n_prev_neur:], self.bias[-1], False, neuron, layer)
                self.neurons.append(neur)
    
    def init_output_neuron(self):
        for output in range(self.output_num):
            # init Weights, Biases
            n_prev_neur = self.num_neurons[-1]
            for neuron in range(n_prev_neur):
                self.weights.append(round(np.random.normal(), round_digits))
            self.bias.append(round(np.random.normal(), round_digits))
            
            # init Neuron
            neur = Neuron(self.weights[-n_prev_neur:], self.bias[-1], True, output)
            self.neurons.append(neur)


    # PRINT INFO OF NETWORK  
    def printInfo(self):
        print("\n--- Network Info ---")
        print("\nInput_Dimension:", self.input_dim)
        print("Num_of_Hidden_Layers:", self.h_layers)
        print("Num_of_Outputs:", self.output_num)
        
        for layer in range(self.h_layers):
            print("\n-H_Layer " + str(layer) + ":")
            print(" > Neurons:", self.num_neurons[layer+1])
        
        #print("\nWeights:", self.weights)
        #print("Bias:", self.bias, "\n")
        
        index = 0
        for layer in range(self.h_layers):
            print("\n\n -- #HIDDEN LAYER " + str(layer) + " --")
            num_neurons_per_layer = self.num_neurons[layer+1]
            for neuron in self.neurons[index:index+num_neurons_per_layer]:
                neuron.printInfo()
            index += num_neurons_per_layer
        
        print("\n\n -- #OUTPUT LAYER --")
        for neuron in self.neurons[-self.output_num:]:
            neuron.printInfo()

    def feedforward(self, x, loss_calc=True):
        out_h = []
        h_values = []
        x_sums = []
        idx = 0
        
        #print("\n\n--- FEED FORWARD ---")
        out_h = x
        for layer in range(self.h_layers):
            #print("\nLayer", layer)
            n_neur = self.num_neurons[layer+1]
            #print("Num_Neurons:", n_neur)
            result = []
            for neuron in self.neurons[idx:idx+n_neur]:
                x_value, tot = neuron.feedforward(out_h)
                x_sums.append(x_value)
                result.append(tot)
                
            out_h = result
            h_values.append(out_h)
            #print("h_values:", h_values)
            idx += n_neur
        
        #print("\nOUTPUT LAYER")
        result = []
        for neuron in self.neurons[-self.output_num:]:
            x_value, tot = neuron.feedforward(out_h)
            x_sums.append(x_value)
            result.append(tot)
            
        h_values.append(result)
        #print("h_values:", h_values)
        if loss_calc:
            return h_values[-1]
        else:
            return x_sums, h_values, result
    
    def partial_derivatives(self, x, y_true, y_pred, x_sums, h_values):
        print("\n- PARTIAL DERIVATIVES -")
        derivatives = []    #array used to return all the values
        
        d_L_d_ypred = -2 * (y_true - y_pred)   #y_pred is an array, could be containing multiple values
        derivatives.append(d_L_d_ypred)
        print("d_L_d_ypred:", d_L_d_ypred)
        
        d_h_d_w = []
        d_h_d_b = []
        index = 0
        neur_cnt = 0
        inputs = x
        for layer in range(self.h_layers):
            #print("\nLayer", layer)
            num_neur = self.num_neurons[layer+1]
            for neuron in self.neurons[index:index+num_neur]:
                dh_dw = []
                for val in inputs:
                    deriv = val * deriv_sigmoid(x_sums[neur_cnt])
                    dh_dw.append(round(deriv, round_digits))
                d_h_d_w.append(dh_dw)
                d_h_d_b.append(deriv_sigmoid(x_sums[neur_cnt]))
                neur_cnt+=1
            
            index+=num_neur                    
            inputs = h_values[layer]
            derivatives.append(d_h_d_w)
            derivatives.append(d_h_d_b)
        print("d_h_d_w:", d_h_d_w)
        print("d_h_d_b:", d_h_d_b)
            
        # Partial derivatives OUTPUT LAYER
        d_ypred_d_w = []
        d_ypred_d_b = []
        d_ypred_d_h = []
        #print("\nOutput Layer")
        #inputs = h_values[-2]    #do not need this
        for neuron in self.neurons[-self.output_num:]:
            dy_dw = []
            for val in inputs:
                deriv = val * deriv_sigmoid(x_sums[neur_cnt])
                dy_dw.append(round(deriv, round_digits))
            d_ypred_d_w.append(dy_dw)
            d_ypred_d_b.append(deriv_sigmoid(x_sums[neur_cnt]))
            
            dy_dh = []
            for weight in neuron.weights:
                deriv = weight * deriv_sigmoid(x_sums[neur_cnt])
                dy_dh.append(round(deriv, round_digits))
            d_ypred_d_h.append(dy_dh)
            neur_cnt+=1
        
        derivatives.append(d_ypred_d_w)
        derivatives.append(d_ypred_d_b)
        derivatives.append(d_ypred_d_h)
        print("d_ypred_d_w:", d_ypred_d_w)
        print("d_ypred_d_b:", d_ypred_d_b)
        print("d_ypred_d_h:", d_ypred_d_h)
        return derivatives

    def update_w_b(self, learn_rate, d_L_d_ypred, d_h_d_w, d_h_d_b, d_ypred_d_w, d_ypred_d_b, d_ypred_d_h):
        print("\n- UPDATE WEIGHTS AND BIASES -")
        index = 0
        neur_cnt = 0
        for layer in range(self.h_layers):
            print("\nLayer", layer)
            num_neur = self.num_neurons[layer+1]
            for neuron in self.neurons[index:index+num_neur]:
                new_weights = []
                print("\n", neuron)
                for weight in neuron.weights:
                    print("weight:", weight)
                    print("weights_before:", neuron.weights)
                    new_value = learn_rate
                    #new_weights.append()
                
                print("weights_after:", neuron.weights)
                
            index+=num_neur

    # TRAINING FUNCTION
    def train(self, data, y_trues):
        learn_rate = 0.1
        epochs = 1   #number of loops
        
        print("\n\n--- TRAINING ---")
        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                x_sums, h_values, y_pred = self.feedforward(x, False)
                #print("\nData:", x, y_true)
                #print("x_sums:", x_sums)
                #print("h_values:", h_values)
                #print("y_pred:", y_pred)
                
                # Partial derivatives
                derivatives = self.partial_derivatives(x, y_true, y_pred, x_sums, h_values)
                d_L_d_ypred = derivatives[0]
                d_h_d_w = derivatives[1]
                d_h_d_b = derivatives[2]    
                d_ypred_d_w = derivatives[3]
                d_ypred_d_b = derivatives[4]
                d_ypred_d_h = derivatives[5]
                
                #Update weights and biases
                self.update_w_b(learn_rate, d_L_d_ypred, d_h_d_w, d_h_d_b, d_ypred_d_w, d_ypred_d_b, d_ypred_d_h)
                
            if epoch % 2 == 0:
                print("\n\nLoss calculation")
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                print(y_pred)
                loss = mse_loss(y_trues, y_preds)
                print("Epoch {} - Loss: {}".format(epoch, loss))
                

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
    
    network.train(data, y_trues)
    
    
# --- TEST MSE ---  
def test_mse():
    print("\n\n--- EXAMPLE 3: test MSE Loss ---\n")
    
    y_true = np.array([1,0,0,1])
    y_pred = np.array([0,0,0,0])

    print("MSE Loss:", mse_loss(y_true, y_pred))
    
    
# --- MAIN ---
    
#test_mse()
basicNetwork()
