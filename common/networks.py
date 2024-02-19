import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from typing import List
def compute_metrics(logits,true_labels):
    _, predicted = torch.max(logits, 1)  # Get the index of the maximum value in each row
    correct = (predicted == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy
class RBM:
    def __init__(self,input_size=320,hidden_size=10,device=None):
        super(RBM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.std = 1e-1
        self.W = nn.Parameter(self.std*torch.randn((hidden_size,input_size),device=device)) # Weights matrix
        self.b = nn.Parameter(torch.zeros(hidden_size,device=device)) # bias for observed data
        self.a = nn.Parameter(torch.zeros(input_size,device=device)) # bias for hidden data
        self.device = device
    
    def forward(self,inputs):
        #print(self.W.device)
        h = F.linear(inputs,self.W, self.b)
        out = torch.sigmoid(h)
        return out
    
    def backward(self,latent):
        v = F.linear(latent,self.W.T ,self.a)
        return torch.sigmoid(v)
    
    def gibbs(self,inputs,k=1):
        x = inputs.clone()
        for i in range(k):
            prob = self.forward(x)
            u = torch.rand(self.hidden_size,device=self.device)
            h = (prob >=u).float()
            prob_ = self.backward(h)
            u = torch.rand(self.input_size,device=self.device)
            x = (prob_ >=u).float()
        return x

    def train(self,data,n_steps,alpha,mode="full",batch_size=30,k=1):
        """
        Contrastive Divergence 1
        """
        if mode not in ['full', 'batch']:
            raise ValueError("mode must be either 'full' or 'batch'")

        if mode == "full":
            batch_size = len(data)
        elif mode == "batch":
            if batch_size is None:
                raise ValueError("batch_size must be specified when mode is 'batch'")
            batch_size = min(batch_size,len(data))
        for i in range(n_steps):
            data=data[torch.randperm(batch_size)] #shuffle the dataset at each iteration  and  use mini batch
            for x in data:
                x_ = self.gibbs(x,k)
                u,v = self.forward(x),self.forward(x_)
                self.W.data = self.W + alpha*(u.outer(x) - v.outer(x_))/batch_size
                self.b.data = self.b + alpha*(u-v)/batch_size
                self.a.data = self.a + alpha*(x-x_)/batch_size

    def generate(self,n_images,k=10):
        samples = torch.zeros((n_images,self.input_size),device=self.device)
        for i in range(n_images):
            x = torch.zeros(self.input_size,device =self.device)
            samples[i] = self.gibbs(x,k)
        return samples


class DBN:
    def __init__(self,input_size,hidden_sizes: List,device):
        super(DBN,self).__init__()
        self.input_size = input_size
        self.device = device
        self.layers = [RBM(input_size,hidden_sizes[0],device)]
        for i in range(len(hidden_sizes)-1):
            self.layers.append(RBM(hidden_sizes[i],hidden_sizes[i+1],device))
    
    def train(self,data,n_steps,alpha,mode="full",batch_size=30,k=1):
        data_ = data.clone()
        for i in trange(len(self.layers)):
            layer = self.layers[i]
            layer.train(data_,n_steps,alpha,mode,batch_size,k)
            data_ = layer.forward(data_)
    
    def generate(self,n_images,k):
        samples = torch.zeros((n_images,self.input_size),device=self.device)
        for i in range(n_images):
            layer = self.layers[-1]
            x = torch.zeros(layer.input_size,device = self.device)
            x_ = layer.gibbs(x,k)
            for layer in self.layers[-2::-1]:
                prob = layer.backward(x_)
                u = torch.rand(layer.input_size,device=self.device)
                x_ = (prob >=u).float()
            samples[i] = x_
        return samples
    
    def get_all_params(self):
        params = []
        for rbm in self.layers:
            params.extend([rbm.W, rbm.b, rbm.a])
        return params


class DNN(nn.Module):
    def __init__(self,input_size,hidden_sizes: List,out_size,device):
        super(DNN,self).__init__()
        self.input_size = input_size
        self.dbn = DBN(input_size,hidden_sizes,device)
        self.fc = nn.Linear(hidden_sizes[-1],out_size)
        self.device = device

    def pretrain(self,data,n_steps,alpha,mode="full",batch_size=30,k=1):
        self.dbn.train(data,n_steps,alpha,mode,batch_size,k)
    
    def forward(self,inputs):
        for layer in self.dbn.layers:
            inputs = layer.forward(inputs)
        inputs = self.fc(inputs)
        out = F.softmax(inputs,dim=-1)
        return out
    
    def train(self,dataloader,n_epochs,lr):
        optimizer = torch.optim.Adam([
                                {'params': self.parameters()},
                                {'params': self.dbn.get_all_params()}  # Add custom object's parameter
                            ],lr=lr)
        for i in trange(n_epochs):
            avg_loss, avg_acc = 0.0, 0.0
            n = 0.
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                preds = self(inputs)
                loss = F.cross_entropy(preds,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n+= inputs.size(0)
                acc = compute_metrics(preds,labels)
                avg_acc += acc*inputs.size(0)
                avg_loss += loss.item()*inputs.size(0)
            avg_loss, avg_acc = avg_loss / n, avg_acc / n
            """if i%2==0:
                print("epoch {0} Loss:= {1} Accuracy: {2}".format(i+1,avg_loss,avg_acc))"""

    @torch.no_grad()
    def evaluate(self,val_loader):
        avg_loss, avg_acc = 0.0, 0.0
        n = 0.
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            preds = self(inputs)
            loss = F.cross_entropy(preds,labels)
            n+= inputs.size(0)
            acc = compute_metrics(preds,labels)
            avg_acc += acc*inputs.size(0)
            avg_loss += loss.item()*inputs.size(0)
        avg_loss, avg_acc = avg_loss / n, avg_acc / n
        return avg_acc