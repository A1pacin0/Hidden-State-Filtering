import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model.model_llama import create_model  # Import the create_model function from model.py
import torch.nn as nn
import argparse
from safetensors import safe_open
from transformers import get_linear_schedule_with_warmup
import os

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def vicuna_cat(tensor_a, tensor_b):
    combined_tensor = torch.cat((tensor_a, tensor_b), dim=0)
    return combined_tensor

def val_cat(tensor_a, tensor_b, size, val):
    values_shape = list(tensor_a.shape)
    values_shape[0] = size  # Modify only the first dimension (number of rows)
    # Create a tensor filled with a specific value
    values_tensor = torch.full(values_shape, val, dtype=tensor_a.dtype, device=tensor_a.device)
    combined_tensor = torch.cat((tensor_a, values_tensor, tensor_b), dim=0)
    return combined_tensor

def k_val_cat(tensors, size, val):
    values_shape = list(tensors[0].shape)
    values_shape[0] = size  # Modify only the first dimension (number of rows)
    # Create a tensor filled with a specific value
    values_tensor = torch.full(values_shape, val, dtype=tensors[0].dtype, device=tensors[0].device)
    
    # Create a new list that sequentially contains the original tensors and the value-filled tensor
    combined_list = []
    for tensor in tensors:
        combined_list.append(tensor)
        combined_list.append(values_tensor)
    
    # Remove the redundant value-filled tensor at the end of the list
    combined_list.pop()

    # Concatenate all tensors
    combined_tensor = torch.cat(combined_list, dim=0)
    return combined_tensor

def load_safetensor_data(harmful_paths, harmless_paths, k):
    harmful_data = []
    harmless_data = []
    
    for harmful_path in harmful_paths:
        hidden_states_harmful = safe_open(harmful_path, framework='pt', device=0)
        for key in hidden_states_harmful.keys():
            tensors = [hidden_states_harmful.get_tensor(key)[-i] for i in range(k, 0, -1)]
            combined_tensor = k_val_cat(tensors, 1, 0)
            harmful_data.append(combined_tensor)

    for harmless_path in harmless_paths:
        hidden_states_harmless = safe_open(harmless_path, framework='pt', device=0)
        for key in hidden_states_harmless.keys():
            tensors = [hidden_states_harmless.get_tensor(key)[-i] for i in range(k, 0, -1)]
            combined_tensor = k_val_cat(tensors, 1, 0)
            harmless_data.append(combined_tensor)

    # Convert lists to tensors
    harmful_X = torch.stack(harmful_data)
    harmless_X = torch.stack(harmless_data)
    
    # Create labels
    harmful_y = torch.ones(harmful_X.size(0), 1)  # Label as 1, indicating unsafe
    harmless_y = torch.zeros(harmless_X.size(0), 1)  # Label as 0, indicating safe

    # Combine data
    X = torch.cat((harmful_X, harmless_X), dim=0)
    y = torch.cat((harmful_y, harmless_y), dim=0)
    return X, y

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(torch.device('cuda'), dtype=torch.float32)
            batch_y = batch_y.to(torch.device('cuda'), dtype=torch.float32)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total
    average_loss = total_loss / len(data_loader)
    return accuracy, average_loss

def main(args):
    # Load data
    X, y = load_safetensor_data(args.dataset_harmful, args.dataset_harmless, args.last_k)

    # Create dataset and data loaders
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, loss function, and optimizer
    input_size = X.shape[1]
    hidden_size = args.hidden_size
    output_size = 1
    model = create_model(input_size, hidden_size, output_size)
    model.to(torch.device('cuda'))
    if args.pretrained_model_path:
        model.load_state_dict(torch.load(args.pretrained_model_path))

    total_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = total_steps // 10
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=total_steps)
    ensure_dir_exists(args.checkpoint)

    # Train the model
    num_epochs = args.num_epochs
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(torch.device('cuda'), dtype=torch.float32)
            batch_y = batch_y.to(torch.device('cuda'), dtype=torch.float32)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']   
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', 'lr =', '{:.6f}'.format(current_lr))

        if (epoch + 1) % 5 == 0:
            accuracy, avg_loss = evaluate_model(model, test_loader, criterion)
            print(f'Validation Accuracy: {accuracy * 100:.2f}%, Validation Loss: {avg_loss:.4f}')

            # Save the best model weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_dir = f'{args.checkpoint}/k={args.last_k}'
                ensure_dir_exists(best_model_dir)
                torch.save(model.state_dict(), f'{best_model_dir}/best_model.pth')
                print(f'Saved best model with accuracy: {best_accuracy * 100:.2f}%')

    final_model_dir = f'{args.checkpoint}/k={args.last_k}'
    ensure_dir_exists(final_model_dir)
    torch.save(model.state_dict(), f'{final_model_dir}/final_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP model.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to the pretrained model.")
    parser.add_argument("--dataset_harmful", type=str, nargs='+', help="Path to the harmful dataset.")
    parser.add_argument("--dataset_harmless", type=str, nargs='+', help="Path to the harmless dataset.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of neurons in the hidden layer.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save the model.")
    parser.add_argument("--last_k", type=int, default=1, help="Number of last hidden states to consider.")
    args = parser.parse_args()
    main(args)
