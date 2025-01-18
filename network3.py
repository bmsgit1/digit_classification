'''Network v3.0: network2.py with L2 regularisation, cross-entropy cost, and improved weight initialisation. 
Shallow neural network for handwritten digit classification, trained on MNIST dataset. 
Implemented in Numpy as a learning exercise, to build an intuitive understanding. 
The code and model are not optimised, and omit many desireable features'''
import random 
import numpy as np 

'''miscellaneous global functions'''
def sigmoid(Z):
    return 1.0/(1.0+np.exp(-Z)) #automatically applied elementwise by numpy if Z a matrix 

def sigmoid_prime(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

'''define CE and qaudratic cost functions for single training example'''
class CrossEntropyCost():
    @staticmethod 
    def cost_function(A, Y):
        '''return cost of output a with desired output y'''
        return np.nan_to_num(-Y*np.log(A) + (1-Y)*np.log(1-A)) 
    
    @staticmethod 
    def delta_L(Z, A, Y):
        '''return error from output layer'''
        return (A-Y) #output error is delta_L = a^L - y for CE cost
    
class QuadraticCost():
    @staticmethod
    def cost_function(A, Y):
        return 0.5*np.linalg.norm(A-Y)**2
    
    @staticmethod
    def delta_L(Z, A, Y):
        return (A-Y)*sigmoid_prime(Z)  

'''main network class'''
class Network():
    def __init__(self, sizes, mini_batch_size, cost=CrossEntropyCost): #cross-entropy cost set as default 
        self.num_layers = len(sizes)
        self.mini_batch_size = mini_batch_size
        self.sizes = sizes 
        self.cost = cost
        self.default_weight_initialiser() #non-standard Gaussian weights set as default 
    
    def default_weight_initialiser(self):
        self.biases = [np.random.randn(x,self.mini_batch_size) for x in self.sizes[1:]] #list of bias vectors for each layer 
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for y,x in zip(self.sizes[1:], self.sizes[:-1])] #list of weight matrices for each layer  

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, X):
        """BATCHED INFERENCE. Return the output of the whole network if "X" is input for whole mini-batch."""
        for B, w in zip(self.biases, self.weights): #loop through b and w of each layer in network 
            X = sigmoid(np.dot(w, X)+B) #calc activation for layer, move forward
        return X

    def SGD(self, training_data, num_epochs, eta, lmbda, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in range(num_epochs): #loop through multiple epochs of training 
            random.shuffle(training_data) #random reordering of training examples 
            mini_batches = [training_data[k:k+self.mini_batch_size] for k in range(0, n, self.mini_batch_size)] #generate mini-batches for epoch 
            for mini_batch in mini_batches: #loop over each mini-batch 
                self.update_mini_batch(mini_batch, eta, lmbda, n) 
            if test_data:
                eval = self.evaluate_accuracy(test_data)
                print("Epoch {0}: {1} / {2} ({3}%)".format(j, eval, n_test, ((eval/n_test)*100)))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying
        stochastic gradient descent using backpropagation to an entire mini batch at once.
        The "mini_batch" is a list of tuples "(x, y)", "eta"
        is the learning rate, "lmbda" is regularisation factor, "n" is training set size."""
        nabla_B = [np.zeros(B.shape) for B in self.biases] #initialise nB matrices for each layer to zero vector 
        nabla_w = [np.zeros(w.shape) for w in self.weights] #initialise nw matrices for each layer to zero matrix 
        X = np.transpose([np.reshape(x, 784) for x,y in mini_batch]) #create input activation and label matrices
        Y = np.transpose([np.reshape(y, 10) for x,y in mini_batch])

        delta_nabla_B, delta_nabla_w = self.backprop(X, Y) #perform backprop on training example, return list containing b and w gradients of each layer
        nabla_B = [nB+dnB for nB, dnB in zip(nabla_B, delta_nabla_B)] #update nB and nw with B and w gradients for entire mini-batch, for each layer  
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] #update weights and biases for entire epoch with nB and nw of mini-batch, for each layer. L2 reg included
        self.biases = [B-(eta/len(mini_batch))*nB for B, nB in zip(self.biases, nabla_B)]

    def backprop(self, X, Y):
        """Return a tuple (nabla_B, nabla_w) representing the
        gradient for the entire mini-batch's quadratic cost function C_x. nabla_B and
        nabla_w are layer-by-layer lists of numpy arrays."""
        nabla_B = [np.zeros(B.shape) for B in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        '''feedforward''' 
        activations = [X] 
        activation = X
        Zs = [] #list to store Z matrices of each layer 
        for w,B in zip(self.weights, self.biases):
            Z = np.dot(w, activation) + B
            Zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)
        
        '''output layer error'''
        delta_L = self.cost.delta_L(Zs[-1], activations[-1], Y) 
        nabla_B[-1] = delta_L #compute the gradients for outer layer 
        nabla_w[-1] = np.dot(delta_L, activations[-2].transpose())

        '''backpropagation of error'''
        delta_l = delta_L 
        for l in range(2, self.num_layers):
            delta_l = np.dot(self.weights[-l+1].transpose(), delta_l) * sigmoid_prime(Zs[-l])
            nabla_B[-l] = delta_l #compute the gradients for layer
            nabla_w[-l] = np.dot(delta_l, activations[-(l+1)].transpose())
        
        return (nabla_B, nabla_w)
    
    def evaluate_accuracy(self, test_data):
        """BATCHED INFERENCE. Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        batches = [test_data[k:k+self.mini_batch_size] for k in range(0, len(test_data), self.mini_batch_size)]
        results = []
        for batch in batches:
            X = np.transpose([np.reshape(x, 784) for x,y in batch]) #create input activation and label matrices
            Y = [y for x,y in batch]
            OUT = self.feedforward(X)
            for i in range(self.mini_batch_size):
                results.append((np.argmax(OUT[:, i]), Y[i])) #select columns representing output and y of each test image
        return sum(int(x == y) for x,y in results)

