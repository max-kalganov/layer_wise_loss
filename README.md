# Layer-wise loss

## Intuition
Train each layer/block of layers independently

## Experiment notes

Input - MNIST flatten  
Output - 10 classes one hot   
Loss func - CategoricalCrossEntropy


### Pretraining

#### Additional layers: 
- num of nodes: num of output classes
- temporary weights: random with mean from N(0, 1) - fixed for a batch  

#### Training details: 
Each loss is for the corresponding layer training. All other layers weights are constants.


### FineTuning

1. Best loss value is selected as a head - other top layers are removed
2. Full network is finetuned until ready


## Example:
***for binary classification***

![experiment example for binary classification](data/experiment_example.png)

Details:
1. Train i - layer features(K_i) using Loss_i.
2. All other weights(K_j: j!=i) are fixed at that moment.
3. Temporary weights W_i in R^(K_i * 1) are all ones
and not trained during backprop.
