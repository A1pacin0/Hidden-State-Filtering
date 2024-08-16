import torch
import argparse
from safetensors import safe_open
from model.model_llama import create_model  # Import the create_model function from model.py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np

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

def load_safetensor_data(harmful_paths=None, harmless_paths=None, k=1):
    harmful_data = []
    harmless_data = []
    
    if harmful_paths:
        for harmful_path in harmful_paths:
            hidden_states_harmful = safe_open(harmful_path, framework='pt', device=0)
            for key in hidden_states_harmful.keys():
                tensors = [hidden_states_harmful.get_tensor(key)[-i] for i in range(k, 0, -1)]
                combined_tensor = k_val_cat(tensors, 1, 0)
                harmful_data.append(combined_tensor)

    if harmless_paths:
        for harmless_path in harmless_paths:
            hidden_states_harmless = safe_open(harmless_path, framework='pt', device=0)
            for key in hidden_states_harmless.keys():
                tensors = [hidden_states_harmless.get_tensor(key)[-i] for i in range(k, 0, -1)]
                combined_tensor = k_val_cat(tensors, 1, 0)
                harmless_data.append(combined_tensor)

    # Convert lists to tensors
    harmful_X = torch.stack(harmful_data) if harmful_data else torch.empty(0)
    harmless_X = torch.stack(harmless_data) if harmless_data else torch.empty(0)
    
    # Create labels
    harmful_y = torch.ones(harmful_X.size(0), 1)  # Label as 1, indicating unsafe
    harmless_y = torch.zeros(harmless_X.size(0), 1)  # Label as 0, indicating safe

    # Combine data
    if harmful_X.size(0) > 0 and harmless_X.size(0) > 0:
        X = torch.cat((harmful_X, harmless_X), dim=0)
        y = torch.cat((harmful_y, harmless_y), dim=0)
    elif harmful_X.size(0) > 0:
        X = harmful_X
        y = harmful_y
    else:
        X = harmless_X
        y = harmless_y

    return X, y

def save_metrics_to_file(filepath, auc, fixed_text):
    with open(filepath, 'a') as file:
        file.write(f"{fixed_text}\n")
        file.write(f"AUC: {auc:.4f}\n")
        file.write("\n")
    print(f"Metrics saved to {filepath}")

def predict(args):
    # Load data
    X, y = load_safetensor_data(args.dataset_harmful, args.dataset_harmless, args.last_k)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    input_size = X.shape[1]
    hidden_size = args.hidden_size
    output_size = 1
    model = create_model(input_size, hidden_size, output_size)
    model.to(torch.device('cuda'))
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    all_predictions = []
    all_true_labels = []

    # Prediction
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(torch.device('cuda'), dtype=torch.float32)
            batch_y = batch_y.to(torch.device('cuda'), dtype=torch.float32)
            outputs = model(batch_X)
            all_predictions.extend(outputs.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())

    # Convert lists to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)

    # Calculate AUC
    auc = roc_auc_score(all_true_labels, all_predictions)

    print(f"AUC: {auc:.4f}")

    save_metrics_to_file(args.output_file, auc, args.fixed_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a trained MLP model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--dataset_harmful", type=str, nargs='+', default=None, help="Path to the harmful dataset.")
    parser.add_argument("--dataset_harmless", type=str, nargs='+', default=None, help="Path to the harmless dataset.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of neurons in the hidden layer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for prediction.")
    parser.add_argument("--output_file", type=str, required=True, help="File path to save the evaluation metrics.")
    parser.add_argument("--fixed_text", type=str, default="Autodan+adv", help="Fixed text to be included in the output file.")
    parser.add_argument("--last_k", type=int, default=1, help="Number of last hidden states to consider.")
    args = parser.parse_args()
    predict(args)
