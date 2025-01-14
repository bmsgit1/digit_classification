import mnist_loader
import network_fullmatrix
#import network
from time import perf_counter

start = perf_counter()
training_data, validation_data, test_data = mnist_loader.process_data()
mini_batch_size = 10
#net = network.Network([784, 30, 10]) #sizes of NN layers 
net = network_fullmatrix.Network([784, 30, 10], mini_batch_size)
net.SGD(training_data, 30, mini_batch_size, 3.0, test_data) #train NN with SGD
end = perf_counter()
print("Training took {} seconds".format(end-start))
