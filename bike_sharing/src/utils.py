import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np

def train_model(train_loader, model, criterion, optimizer):
    
    running_loss = 0.0
    
    for inputs, targets in train_loader:

        optimizer.zero_grad()
        
        outputs = model(inputs) # float
        targets = targets.float()
        
        loss = criterion(outputs, targets)
        loss.backward() # calculate descent gradient

        running_loss += loss.item()
        optimizer.step()
        
    return running_loss / len(train_loader)
        

def evaluate_model(val_loader, criterion, model):
    
    model.eval()
    val_loss = 0.0
    true_values = []
    predictions = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            
            true_values.extend(targets.numpy())
            predictions.extend(outputs.numpy())
            
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    return val_loss / len(val_loader), true_values, predictions


def train_evaluate_model (train_loader, val_loader, model, criterion, optimizer, num_epochs, print_interval):
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        
        # train 
        running_loss = train_model(train_loader, model, criterion, optimizer)
        train_losses.append(running_loss)
        if (epoch + 1) % print_interval == 0: 
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")
    
        
        # Evaluate
        val_loss, true_values, predictions = evaluate_model(val_loader, criterion, model)
        val_losses.append(val_loss)
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}")
   
    return train_losses, val_losses



def evaluate_plot(trues, predictions):
    
    plt.figure(figsize=(15,10))
    plt.plot(range(len(trues)),trues, label='True')
    plt.plot(range(len(predictions)), predictions, label = 'Predicted' )
    plt.title('True vs Predicted')
    plt.legend()
    plt.show()



def create_data_loaders(X, y, batch_size=64):
    
    dataset = CustomDataset(X, y)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def split_train_validation_test(data, target_column, train_size=0.8):
    
    data_sorted = data.sort_index()
    
    split_index = int(train_size * len(data_sorted))
    train_df = data_sorted.iloc[:split_index]
    val_df = data_sorted.iloc[split_index:]
    
  
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df.loc[:,[target_column]]

    X_val = val_df.drop(columns=[target_column])
    y_val = val_df.loc[:,[target_column]]
    
    return X_train, y_train, X_val, y_val


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X_sample = self.X[index]
        y_sample = self.y[index]
        
        return X_sample, y_sample