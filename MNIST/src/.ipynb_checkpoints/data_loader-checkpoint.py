import torch
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def create_datasets(transform, train_size=0.8, batch_size=64):
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Determine sizes of train, validation, and test sets
    num_train_val = len(train_dataset)
    num_train = int(train_size * num_train_val)
    num_val = num_train_val - num_train
    num_test = len(test_dataset)
    

    # Generate indices for train, validation, and test sets
    indices = list(range(num_train_val))
    np.random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = list(range(num_test))

    # Create data samplers for train, validation, and test sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

    return train_loader, val_loader, test_loader