import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
from torch.nn.functional import one_hot

def binarize_image(image):
    # Binariser l'image : 0 pour les pixels < 127, 1 pour ceux >= 127
    return (image >= 127/255.).float()

def lire_alpha(data,label):
    matrix = data[label]
    X = [i.flatten() for i in matrix]
    return np.array(X)

def one_hot_collate(batch):
    data, labels = zip(*batch)
    data = torch.stack(data)  # Converts list of tensors to a single tensor
    
    # Convert labels to one-hot encoding
    labels = torch.tensor(labels)
    one_hot_labels = torch.zeros(len(labels), 10).scatter_(1, labels.unsqueeze(1), 1)
    
    return data, one_hot_labels

# Define a transform to normalize and flatten the data
transform = transforms.Compose([
    transforms.Resize((20, 16)),  # Resize the images to 20x16
    transforms.ToTensor(),
    transforms.Lambda(lambda x: binarize_image(x)),  # binarise
    transforms.Lambda(lambda x: torch.flatten(x)) , # Flatten the images
    
])

def load_mnist(train_subset_size=None):
    # Load the MNIST training data
    full_train_dataset = datasets.MNIST(root='', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='', train=False, download=True, transform=transform)

    # Select a subset of images
    if train_subset_size is not None:
        subset_indices = np.arange(train_subset_size)  # This gets the first 1000 indices
        train_dataset = Subset(full_train_dataset, subset_indices)
    else:
        train_dataset = full_train_dataset
    return train_dataset,test_dataset