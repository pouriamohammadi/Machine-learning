# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 09:51:22 2021

@author: Asus
"""

#using neural network with pytorch

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Check System Devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    for d in range(device_count):
        device_name = torch.cuda.get_device_name(d)
        print(device_name)
#

device = torch.device('cuda:0')

class BankNote(Dataset):
    def __init__(self, data_path, mode):
        
        super(BankNote, self).__init__()
        
        # TODO
        # 1. Initialize internal data 
        
        raw_tr = np.loadtxt(os.path.join(data_path, 'train.csv'), delimiter=',')
        raw_te = np.loadtxt(os.path.join(data_path, 'test.csv'), delimiter=',')
        
        Xtr, ytr, Xte, yte = \
            raw_tr[:,:-1], raw_tr[:,-1].reshape([-1,1]), raw_te[:,:-1], raw_te[:,-1].reshape([-1,1])
        
        if mode == 'train':
            self.X, self.y = Xtr, ytr
        elif mode == 'test':
            self.X, self.y = Xte, yte
        else:
            raise Exception("Error: Invalid mode option!")
        
    def __getitem__(self, index):
        
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        
        return self.X[index,:], self.y[index,:]
    
    def __len__(self,):
        # Return total number of samples.
        return self.X.shape[0]
    
class Net(nn.Module):
    def __init__(self, config, act=nn.Tanh()):
        super(Net, self).__init__()
        layers_list = []
        for l in range(len(config)-2):
            in_dim = config[l]
            out_dim = config[l+1]
            
            layers_list.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            layers_list.append(nn.ReLU())
        # last layer
        layers_list.append(nn.Linear(in_features=config[-2], out_features=config[-1]))
        # containers: https://pytorch.org/docs/stable/nn.html#containers
        self.net = nn.ModuleList(layers_list)
    def forward(self, X):
        h = X
        for layer in self.net:
            h = layer(h)
        return h

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
        torch.nn.init.constant_(m.bias.data, 1)
        
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device=device)
            y = y.float().to(device=device)
            scores = model(x)
            # _, predictions = scores.max(1)
            predictions = torch.sign(scores)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        return num_correct , num_samples

# Train
dataset_train = BankNote('', mode='train')
dataset_test = BankNote('', mode='test')

# create dataloaders for training and testing
train_loader = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=False)

# training hypyterparameters
epochs=20
lr=1e-3
reg=1e-5
print ("give number of depth")
depth = int(input())
# model configuration
width_list=[5,10,25,50,100]

for hidden_layers in width_list:
# hidden_layers=5
    if depth == 9:
        config = [4,hidden_layers,hidden_layers,hidden_layers,hidden_layers,hidden_layers
                  ,hidden_layers,hidden_layers,hidden_layers,1]
    elif depth == 5:
        config = [4,hidden_layers,hidden_layers,hidden_layers,hidden_layers,1]
    else:
        config = [4,hidden_layers,hidden_layers,1]
    model = Net(config).to(device)
    model = model.to("cuda:0")
    print('\nhidden_layers=',hidden_layers)
    # apply initial w
    model.apply(init_weights)
    
    # instantiate optimizer and pass model's internal differentiable parameters
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    
    # loss function 
    loss_MSE = nn.MSELoss()
    
    # train
    train_loss=[]
    for ie in range(epochs+1):
        loss_array_train =[]
        model.train()  
        # print("Epoch", ie)
        for batch_idx, (X, y) in enumerate(train_loader):
            ## Before Training Starts, put model in Training Mode
            ## Put Model in Training Mode
            Xtr = X.float().to(device)
            ytr = y.float().to(device)
            #calculate MSE loss
            pred = model(Xtr)
            loss = loss_MSE(pred, ytr)
            # step 1: clear the grads
            optimizer.zero_grad()
            # step 2: backward the computational graph
            loss.backward()
            # step 3: take the gradient step
            optimizer.step()
            # save loss of each batch   
            loss_array_train.append(loss.cpu().detach().numpy())
        # print("Training loss: ",np.mean(loss_array_train))
        train_loss.append(np.mean(loss_array_train))   
         
    #test
    model.eval()
    loss_array_test =[]
    for batch_idx, (X, y) in enumerate(test_loader):
        Xte = X.float().to(device)
        yte = y.float().to(device)
        #calculate MSE loss
        pred = model(Xtr)
        loss_test = loss_MSE(pred, ytr)
        # save loss of each batch   
        loss_array_test.append(loss_test.cpu().detach().numpy())
    mean_loss=np.mean(loss_array_test)
    print("Test loss: ",mean_loss)
        
    # Get accuracy of the model
    num_correct, num_samples = check_accuracy(test_loader, model)
    accur = float(num_correct)/float(num_samples)
    print('Test accuracy =', accur) 
    
    ## Plot graph of loss vs epoch
    plt.plot(train_loss)
    plt.ylabel('Training loss')
    plt.xlabel('epochs')
    plt.show()
    
    
    
