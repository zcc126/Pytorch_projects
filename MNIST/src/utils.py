import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_model(train_loader, model, criterion, optimizer):
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate_model(val_loader, model, criterion):
    model.eval()
    val_loss = 0.0
    true_values = []
    predictions = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs, 1)
            true_values.extend(targets.numpy())
            predictions.extend(predicted.numpy())
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    return val_loss / len(val_loader), true_values, predictions

def train_evaluate_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, print_interval):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Train 
        running_loss = train_model(train_loader, model, criterion, optimizer)
        train_losses.append(running_loss)
        if (epoch + 1) % print_interval == 0: 
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss}")
    
        # Evaluate
        val_loss, _, _ = evaluate_model(val_loader, model, criterion)
        val_losses.append(val_loss)
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}")
    return train_losses, val_losses
