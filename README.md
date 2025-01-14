# digit_classification
My implementation of a neural network that learns to classify handwritten digits, the main project in Michael Neilsen's fantastic book _Neural Networks and Deep Learning_ (http://neuralnetworksanddeeplearning.com/index.html). It is trained on the MNIST dataset (60000 images) split into a 50000-image test set, 10000-image validation set (for setting hyperparameters) and 10000-image test set.

- network.py is a the chapter 1 implementation, a shallow neural net using mini-batch stochastic gradient descent with no optimisations.
  Three layer network with sizes [784, 30, 10], learning rate = 3.0, 30 epochs and mini batch size of 10 gives:
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

- 
  
