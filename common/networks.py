import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader, TensorDataset

def compute_metrics(logits,true_labels):
    _, predicted = torch.max(logits, 1)  # Get the index of the maximum value in each row
    true_labels = true_labels.argmax(dim=1)
    correct = (predicted == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy

def calculate_mse(original_images, reconstructed_images):
    """
    Calculate the Mean Squared Error (MSE) between original and reconstructed images.
    
    Parameters:
    - original_images: A list or tenosr of original images.
    - reconstructed_images: A list or tensor of reconstructed images.
    
    Returns:
    - mean_squared_error: The average MSE over all the images.
    """

    if original_images.shape != reconstructed_images.shape:
        raise ValueError("Original and reconstructed images must have the same shape.")
    mse_per_image = np.mean((original_images - reconstructed_images) ** 2, axis=1)
    mean_squared_error = np.mean(mse_per_image)
    
    return mean_squared_error

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
        bs = x.size(0) 
        for i in range(k):
            prob = self.forward(x) #bs,hidden_size
            u = torch.rand(bs,self.hidden_size,device=self.device)
            h = (prob >=u).float()
            prob_ = self.backward(h)
            u = torch.rand(bs,self.input_size,device=self.device)
            x = (prob_ >=u).float()
        return x

    def train(self,data,n_steps,alpha,batch_size=30,k=1):
        """
        Contrastive Divergence 1
        """
        batch_size = min(batch_size,len(data))
        dataset = TensorDataset(data)  # `data` est votre tenseur
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for i in range(n_steps):
            for batch in data_loader:
                x = batch[0]
                x_ = self.gibbs(x,k)  # renvoye après un forward et un backward de GIBBS
                u,v = self.forward(x),self.forward(x_)
                self.W.data = self.W + alpha*(torch.matmul(u.T, x) - torch.matmul(v.T, x_))/batch_size
                self.b.data = self.b + alpha*(torch.sum(u - v, dim=0))/batch_size
                self.a.data = self.a + alpha*(torch.sum(x - x_, dim=0))/batch_size

    def generate(self,n_images,k=10):
        x = torch.zeros((n_images,self.input_size),device=self.device)
        samples = self.gibbs(x,k)
        return samples


class DBN:
    def __init__(self,input_size,hidden_sizes: List,device):
        super(DBN,self).__init__()
        self.input_size = input_size
        self.device = device
        self.layers = [RBM(input_size,hidden_sizes[0],device)]
        for i in range(len(hidden_sizes)-1):
            self.layers.append(RBM(hidden_sizes[i],hidden_sizes[i+1],device))
    
    def train(self,data,n_steps,alpha,batch_size=30,k=1):
        data_ = data.clone()
        for i in trange(len(self.layers)):
            layer = self.layers[i]
            layer.train(data_,n_steps,alpha,batch_size,k)
            data_ = layer.forward(data_)
            

        
    def generate(self,n_images,k):
        layer = self.layers[-1]
        x = torch.zeros((n_images,layer.input_size),device=self.device)
        #x = torch.rand((n_images,layer.input_size),device=self.device) pour tester avec une init aleatoire
        x_ = layer.gibbs(x,k)
        for layer in self.layers[-2::-1]:
            prob = layer.backward(x_)
            u = torch.rand(n_images,layer.input_size,device=self.device)
            x_ = (prob >=u).float()
        samples = x_
        return samples
    


class DNN(nn.Module):
    def __init__(self,input_size,hidden_sizes: List,out_size,device):
        super(DNN,self).__init__()
        self.input_size = input_size
        self.dbn = DBN(input_size,hidden_sizes,device)
        self.fc = nn.Sequential(nn.Identity(),nn.Dropout(0.2),nn.ReLU(),nn.Linear(hidden_sizes[-1],out_size))
        self.device = device
    
    
    
        # Manually register parameters of each RBM layer
        for i, rbm in enumerate(self.dbn.layers):
            self.register_parameter(name=f'W_{i}', param=rbm.W)
            self.register_parameter(name=f'b_{i}', param=rbm.b)
            
            
    def pretrain(self,data,n_steps,alpha,batch_size=30,k=1):
        self.dbn.train(data,n_steps,alpha,batch_size,k)
    
    def forward(self,inputs):
        for layer in self.dbn.layers:
            inputs = layer.forward(inputs)
        inputs = self.fc(inputs)
        out = F.softmax(inputs,dim=-1)
        return out
    
    def train(self,dataloader,n_epochs,lr):
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        for i in trange(n_epochs):
            avg_loss, avg_acc = 0.0, 0.0
            n = 0.
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                preds = self(inputs)
                loss = F.binary_cross_entropy(preds,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                """n+= inputs.size(0)
                acc = compute_metrics(preds,labels)
                avg_acc += acc*inputs.size(0)
                avg_loss += loss.item()*inputs.size(0)
            avg_loss, avg_acc = avg_loss / n, avg_acc / n
            if i%2==0:
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
            loss = F.binary_cross_entropy(preds,labels)
            n+= inputs.size(0)
            acc = compute_metrics(preds,labels)
            avg_acc += acc*inputs.size(0)
            avg_loss += loss.item()*inputs.size(0)
        avg_loss, avg_acc = avg_loss / n, avg_acc / n
        return avg_acc

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, device):
        super().__init__()

        self.device=device
        self.z_dim=z_dim
        self.input_dim=input_dim
        # encoder
        self.img_2hid1 = nn.Linear(self.input_dim, 256)
        self.img_2hid2 = nn.Linear(256, 128)
        self.img_2hid3 = nn.Linear(128, 20)
        


        self.hid_2mu = nn.Linear(20, self.z_dim)
        self.hid_2sigma = nn.Linear(20, self.z_dim)

        # decoder
        self.z_2hid1 = nn.Linear(z_dim, 20)
        self.z_2hid2 = nn.Linear(20, 128)
        self.z_2hid3 = nn.Linear(128, 256)
        self.hid_2img = nn.Linear(256, self.input_dim)

    def encode(self, x):
        h = F.relu(self.img_2hid1(x))
        h = F.relu(self.img_2hid2(h))
        h = F.relu(self.img_2hid3(h))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        new_h = F.relu(self.z_2hid1(z))
        new_h = F.relu(self.z_2hid2(new_h))
        new_h = F.relu(self.z_2hid3(new_h))
        x = torch.sigmoid(self.hid_2img(new_h))
        return x

    def forward(self, x):
      mu, sigma = self.encode(x)

      # Sample from latent distribution from encoder
      epsilon = torch.randn_like(sigma)
      z_reparametrized = mu + sigma*epsilon

      x = self.decode(z_reparametrized)
      return x, mu, sigma

    def generate_images(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim).to(self.device)
            samples = self.decode(z)
            samples = torch.where(samples < 0.5, torch.zeros_like(samples), torch.ones_like(samples))
        return samples

    def reconstruct_images(self, x):
        generated_images = []
        for image in x:
            with torch.no_grad():
                #image = torch.clamp(image.view(1, -1), 0, 1)
                mean, std = self.encode(image.to(self.device))
                epsilon = torch.randn_like(std)
                z = mean + epsilon * std
                out = self.decode(z)

                out = torch.where(out < 0.5, torch.zeros_like(out), torch.ones_like(out))
                generated_images.append(out)
        return generated_images

    def train(self,num_epochs, optimizer, loss_fn, train_loader):
    # Start training
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader))
            for i, x in loop:
                # Forward pass
                x = x.to(self.device).view(-1, self.input_dim)
                x_reconst, mu, sigma = self.forward(x)


                reconst_loss = loss_fn(x_reconst, x)
                kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

                # Backprop and optimize
                loss = reconst_loss + kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())