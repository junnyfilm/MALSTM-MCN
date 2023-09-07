
import numpy as np
import torch
import torch.nn as nn
from customdataset import loaders
import tqdm

def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for inputs, labels in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        output,_ = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss,accuracy