import numpy as np
import torch
import torch.nn as nn
from customdataset import loaders
import tqdm
from torch.autograd import Variable


def test(classifier, dataloader):
    # setup the network
    predlist=[]
    real=[]    

    classifier.eval()
    correct = 0.0
    
    max_tr_score = 0
    max_val_score = 0
    for batch_idx, (data) in enumerate(dataloader):
    
        signal1,label = data
        signal1,label = Variable(signal1.cuda()),Variable(label.cuda().long())
        out,_ = classifier(signal1)   

        pred = out.data.max(1, keepdim= True)[1]
        predlist.append(pred.cpu().detach().numpy().squeeze())
        real.append(label)        
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        # predlist.append(pred)
    print('\nAccuracy: {}/{} ({:.4f}%)\n'.format(
        correct, len(dataloader.dataset), 100. * float(correct) / len(dataloader.dataset)))
    acc=100. * float(correct) / len(dataloader.dataset)
    return acc,predlist,real



def test_crossval(classifier1,classifier2,classifier3,classifier4,classifier5, dataloader):
    # setup the network
    predlist=[]
    real=[]    

    classifier.eval()
    correct = 0.0
    
    max_tr_score = 0
    max_val_score = 0
    for batch_idx, (data) in enumerate(dataloader):
    
        signal1,label = data
        signal1,label = Variable(signal1.cuda()),Variable(label.cuda().long())
        
        
        out1,_ = classifier1(signal1)
        out2,_ = classifier2(signal1) 
        out3,_ = classifier3(signal1)
        out4,_ = classifier4(signal1)
        out5,_ = classifier5(signal1)

        
        out=(out1+out2+out3+out4+out5)/5
        
        pred = out.data.max(1, keepdim= True)[1]
        predlist.append(pred.cpu().detach().numpy().squeeze())
        real.append(label)        
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        # predlist.append(pred)
    print('\nAccuracy: {}/{} ({:.4f}%)\n'.format(
        correct, len(dataloader.dataset), 100. * float(correct) / len(dataloader.dataset)))
    acc=100. * float(correct) / len(dataloader.dataset)
    return acc,predlist,real