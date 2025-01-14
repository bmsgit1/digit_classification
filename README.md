# digit_classification
My implementation of a neural network that learns to classify handwritten digits, the main project in Michael Neilsen's fantastic book _Neural Networks and Deep Learning_ (http://neuralnetworksanddeeplearning.com/index.html). It is trained on the MNIST dataset (60000 images in a compressed Python pickle file mnist.pkl.gz) split into a 50000-image test set, 10000-image validation set (for setting hyperparameters) and 10000-image test set. I am using an Apple M2 CPU.

- network.py is a the chapter 1 implementation, a shallow neural net using mini-batch stochastic gradient descent with no optimisations. Example run of a three layer network with sizes [784, 30, 10], learning rate = 3.0, 30 epochs and mini batch size of 10 gives:  

  _% python3 training_run.py_   
  Epoch 0: 9042 / 10000 (90.42%)  
  Epoch 1: 9224 / 10000 (92.24%)  
  Epoch 2: 9255 / 10000 (92.55%)  
  Epoch 3: 9325 / 10000 (93.25%)  
  Epoch 4: 9355 / 10000 (93.55%)  
  Epoch 5: 9382 / 10000 (93.82000000000001%)  
  Epoch 6: 9370 / 10000 (93.7%)  
  Epoch 7: 9411 / 10000 (94.11%)  
  Epoch 8: 9389 / 10000 (93.89%)  
  Epoch 9: 9399 / 10000 (93.99%)  
  Epoch 10: 9427 / 10000 (94.27%)  
  Epoch 11: 9444 / 10000 (94.44%)  
  Epoch 12: 9422 / 10000 (94.22%)  
  Epoch 13: 9440 / 10000 (94.39999999999999%)  
  Epoch 14: 9453 / 10000 (94.53%)  
  Epoch 15: 9484 / 10000 (94.84%)  
  Epoch 16: 9455 / 10000 (94.55%)  
  Epoch 17: 9455 / 10000 (94.55%)  
  Epoch 18: 9479 / 10000 (94.78999999999999%)  
  Epoch 19: 9461 / 10000 (94.61%)  
  Epoch 20: 9481 / 10000 (94.81%)  
  Epoch 21: 9470 / 10000 (94.69999999999999%)  
  Epoch 22: 9478 / 10000 (94.78%)  
  Epoch 23: 9477 / 10000 (94.77%)  
  Epoch 24: 9475 / 10000 (94.75%)  
  Epoch 25: 9485 / 10000 (94.85%)  
  Epoch 26: 9500 / 10000 (95.0%)  
  Epoch 27: 9497 / 10000 (94.97%)  
  Epoch 28: 9493 / 10000 (94.93%)  
  Epoch 29: 9477 / 10000 (94.77%)  
  Training took 79.958410042 seconds  
  Max accuracy at Epoch 26  

- network_fullmatrix.py is the chapter 2 implementation. It is the same as network.py but with batched backpropagation, so the gradients for all training examples in a mini-batch are computed simultaneously.
  Example run of a three layer network with sizes [784, 30, 10], learning rate = 3.0, 30 epochs and mini batch size of 10 gives:  

  _% python3 training_run.py_   
  Epoch 0: 9055 / 10000 (90.55%)  
  Epoch 1: 9280 / 10000 (92.80000000000001%)  
  Epoch 2: 9300 / 10000 (93.0%)  
  Epoch 3: 9335 / 10000 (93.35%)  
  Epoch 4: 9354 / 10000 (93.54%)  
  Epoch 5: 9392 / 10000 (93.92%)  
  Epoch 6: 9400 / 10000 (94.0%)  
  Epoch 7: 9430 / 10000 (94.3%)  
  Epoch 8: 9443 / 10000 (94.43%)  
  Epoch 9: 9426 / 10000 (94.26%)  
  Epoch 10: 9434 / 10000 (94.34%)  
  Epoch 11: 9443 / 10000 (94.43%)  
  Epoch 12: 9475 / 10000 (94.75%)  
  Epoch 13: 9484 / 10000 (94.84%)  
  Epoch 14: 9437 / 10000 (94.37%)  
  Epoch 15: 9461 / 10000 (94.61%)  
  Epoch 16: 9423 / 10000 (94.23%)  
  Epoch 17: 9460 / 10000 (94.6%)  
  Epoch 18: 9482 / 10000 (94.82000000000001%)  
  Epoch 19: 9481 / 10000 (94.81%)  
  Epoch 20: 9451 / 10000 (94.51%)  
  Epoch 21: 9459 / 10000 (94.59%)  
  Epoch 22: 9486 / 10000 (94.86%)  
  Epoch 23: 9480 / 10000 (94.8%)  
  Epoch 24: 9484 / 10000 (94.84%)  
  Epoch 25: 9488 / 10000 (94.88%)  
  Epoch 26: 9417 / 10000 (94.17%)  
  Epoch 27: 9467 / 10000 (94.67%)  
  Epoch 28: 9469 / 10000 (94.69%)  
  Epoch 29: 9484 / 10000 (94.84%)  
  Training took 16.019150959 seconds

- The chapter 3 implementation will include improved weight and bias initialisation, cross-entropy cost, andf regularisation. 
  
