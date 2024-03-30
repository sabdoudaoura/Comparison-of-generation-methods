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
    """
    Compute the accuracy of the model given predicted logits and true labels.

    Parameters:
    - logits: A tensor of predicted logits.
    - true_labels: A tensor of true labels.

    Returns:
    - accuracy: The accuracy of the model as a float.
    """
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
    """
    A Restricted Boltzmann Machine (RBM) implementation.

    Parameters:
    - input_size: The size of the input data.
    - hidden_size: The size of the hidden layer.
    - device: The device to use for tensor operations.
    """
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
        """
        Compute the forward pass of the RBM (inputs to latent)

        Parameters:
        - inputs: The input data.

        Returns:
        - out: The output of the RBM.
        """
        #print(self.W.device)
        h = F.linear(inputs,self.W, self.b)
        out = torch.sigmoid(h)
        return out
    
    def backward(self,latent):
        """
        Compute the backward pass of the RBM (latentt to inputs)
        Parameters:
        - latent: The latent variables.

        Returns:
        - out: The output of the RBM.
        """
        v = F.linear(latent,self.W.T ,self.a)
        return torch.sigmoid(v)
    
    def gibbs(self,inputs,k=1):
        """
        Perform k steps of Gibbs sampling.

        Parameters:
        - inputs: The input data.
        - k: The number of Gibbs sampling steps.

        Returns:
        - x: The output after k steps of Gibbs sampling.
        """
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
        Train the RBM using Contrastive Divergence.

        Parameters:
        - data: The training data.
        - n_steps: The number of training steps.
        - alpha: The learning rate.
        - batch_size: The batch size.
        - k: The number of Gibbs sampling steps.
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
        """
        Generate samples from the RBM.

        Parameters:
        - n_images: The number of samples to generate.
        - k: The number of Gibbs sampling steps.

        Returns:
        - samples: The generated samples.
        """
        x = torch.randn((n_images,self.input_size),device=self.device)
        samples = self.gibbs(x,k)
        return samples

    def parameters(self):
        """
        Get the number of parameters in the RBM.

        Returns:
        - The number of parameters in the RBM.
        """
        return self.W.numel() + self.b.numel() + self.a.numel()

class DBN:
    """
    A Deep Belief Network (DBN) implementation.

    Parameters:
    - input_size: The size of the input data.
    - hidden_sizes: A list of sizes for the hidden layers.
    - device: The device to use for tensor operations.
    """
    def __init__(self,input_size,hidden_sizes: List,device):
        super(DBN,self).__init__()
        self.input_size = input_size
        self.device = device
        self.layers = [RBM(input_size,hidden_sizes[0],device)]
        for i in range(len(hidden_sizes)-1):
            self.layers.append(RBM(hidden_sizes[i],hidden_sizes[i+1],device))
    
    def train(self,data,n_steps,alpha,batch_size=30,k=1):
        """
        Train the DBN using layer-wise pretraining.

        Parameters:
        - data: The training data.
        - n_steps: The number of training steps.
        - alpha: The learning rate.
        - batch_size: The batch size.
        - k: The number of Gibbs sampling steps.
        """
        data_ = data.clone()
        for i in trange(len(self.layers)):
            layer = self.layers[i]
            layer.train(data_,n_steps,alpha,batch_size,k)
            data_ = layer.forward(data_)
            
    def generate(self,n_images,k):
        """
        Generate samples from the DBN.

        Parameters:
        - n_images: The number of samples to generate.
        - k: The number of Gibbs sampling steps.

        Returns:
        - samples: The generated samples.
        """
        layer = self.layers[-1]
        x = torch.randn((n_images,layer.input_size),device=self.device)
        x_ = layer.gibbs(x,k)
        for layer in self.layers[-2::-1]:
            prob = layer.backward(x_)
            u = torch.rand(n_images,layer.input_size,device=self.device)
            x_ = (prob >=u).float()
        samples = x_
        return samples
    
    def parameters(self):
        """
        Get the number of parameters in the DBN.

        Returns:
        - The number of parameters in the DBN.
        """
        return sum(layer.count_parameters() for layer in self.layers)

class DNN(nn.Module):
    """
    A Deep Neural Network (DNN) implementation.

    Parameters:
    - input_size: The size of the input data.
    - hidden_sizes: A list of sizes for the hidden layers.
    - out_size: The size of the output layer.
    - device: The device to use for tensor operations.
    """
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
        """
        Pretrain the DNN using a DBN.

        Parameters:
        - data: The training data.
        - n_steps: The number of pretraining steps.
        - alpha: The learning rate.
        - batch_size: The batch size.
        - k: The number of Gibbs sampling steps.
        """
        self.dbn.train(data,n_steps,alpha,batch_size,k)
    
    def forward(self,inputs):
        """
        Compute the forward pass of the DNN.

        Parameters:
        - inputs: The input data.

        Returns:
        - out: The output of the DNN.
        """

        for layer in self.dbn.layers:
            inputs = layer.forward(inputs)
        inputs = self.fc(inputs)
        out = F.softmax(inputs,dim=-1)
        return out
    
    def train(self,dataloader,n_epochs,lr):
        """
        Train the DNN using backpropagation.

        Parameters:
        - dataloader: The training data loader.
        - n_epochs: The number of training epochs.
        - lr: The learning rate.
        """
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

    @torch.no_grad()
    def evaluate(self,val_loader):
        """
        Evaluate the DNN on a validation set.

        Parameters:
        - val_loader: The validation data loader.

        Returns:
        - avg_acc: The average accuracy on the validation set.
        """
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
    """
    A VariationalAutoEncoder (VAE) implementation.

    Parameters:
    - input_dim: The size of the input data.
    - z_dim: The size of the bottle neck size.
    - device: The device to use for tensor operations.
    """
    def __init__(self, input_dim, z_dim, device):
        super().__init__()

        self.device=device
        self.z_dim=z_dim
        self.input_dim=input_dim
        # encoder
        self.img_2hid1 = nn.Linear(self.input_dim, 512)
        self.img_2hid2 = nn.Linear(512, 256)
        


        self.hid_2mu = nn.Linear(256, self.z_dim)
        self.hid_2sigma = nn.Linear(256, self.z_dim)

        # decoder
        self.z_2hid1 = nn.Linear(z_dim, 256)
        self.z_2hid3 = nn.Linear(256, 512)
        self.hid_2img = nn.Linear(512, self.input_dim)

    def encode(self, x):
        """
        Encoder function of the VAE.

        Parameters:
        - x: The training data.
        - mu : paramter of the hidden distribution
        - log_sigma : paramter of the hidden distribution
        """
        h = F.relu(self.img_2hid1(x))
        h = F.relu(self.img_2hid2(h))
        mu = self.hid_2mu(h)
        log_sigma = self.hid_2sigma(h)
        return mu, log_sigma

    def decode(self, z):
        """
        decoder function of the VAE.

        Parameters:
        - z:  Data in the latent space
        - x : The reconstructed data

        """
        new_h = F.relu(self.z_2hid1(z))
        new_h = F.relu(self.z_2hid3(new_h))
        x = torch.sigmoid(self.hid_2img(new_h))
        return x

    def forward(self, x):
      
      """
        Compute the forward pass of the VAE.

        Parameters:
        - inputs: The input data.

        Returns:
        - out: The output of the DNN
        - mu : encoder output
        - log_sigma : encoder output
        """
      mu, log_sigma = self.encode(x)

      # Sample from latent distribution from encoder
      epsilon = torch.randn_like(log_sigma)
      z_reparametrized = mu + torch.exp(0.5*log_sigma)*epsilon

      x = self.decode(z_reparametrized)
      return x, mu, log_sigma

    def generate_images(self, num_samples):
        """
        Generate samples from the VAE.

        Parameters:
        - num_samples: The number of samples to generate.


        Returns:
        - samples: The generated samples.
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim).to(self.device)
            samples = self.decode(z)
            
        return samples

    def reconstruct_images(self, x):
        generated_images = []
        for image in x:
            with torch.no_grad():
                
                mean, std = self.encode(image.to(self.device))
                epsilon = torch.randn_like(std)
                z = mean + epsilon * std
                out = self.decode(z)

                
                generated_images.append(out)
        return generated_images

    def train(self,num_epochs, optimizer, loss_fn, train_loader):
        """
        Training loop of the VAE

        Parameters:
        - num_epochs: The number of training epochs.
        - optimizer: The optimizer to be used.
        - loss_fn: The loss function to use.
        - train_loader: The dataloader to use to load the data.
        """
    # Start training
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader))
            for i, (x,y) in loop:
                # Forward pass
                x = x.to(self.device).view(-1, self.input_dim)
                x_reconst, mu, log_sigma = self.forward(x)

                reconst_loss = loss_fn(x_reconst, x).sum(dim=-1).mean()
          
                kl_div = - 0.5*torch.sum(1 + log_sigma - mu.pow(2) - torch.exp(log_sigma), dim=-1).mean(dim=0)
        
                # Backprop and optimize
                loss = reconst_loss + kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())