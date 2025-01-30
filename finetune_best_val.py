import torch
from tqdm import tqdm
import json
import os
from args import parse_arguments
from datasets.common import maybe_dictionarize
from utils import load_zeroshot, get_chosen_dataset, build_zeroshot, get_balanced_dataloader


# Number of epochs for each dataset
EPOCHS = {
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SVHN": 4,
}

# Fine-tuning function
def fine_tune_clip_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dictionary to store the best validation accuracy for each dataset
    best_validation_results = {}

    # Iterate over all datasets in EPOCHS
    for dataset_name in EPOCHS.keys():
        print(f"\nFine-tuning on the dataset: {dataset_name}")

        if not os.path.isfile("./encoders/zeroshot.pt"):
            build_zeroshot ("DTD", device, args)

        model = load_zeroshot(dataset_name, args, device)
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss_function = torch.nn.CrossEntropyLoss()

        if args.balanced:
            train_loader = get_balanced_dataloader(dataset_name+'Val', model, args, is_train=True)
            val_loader = get_balanced_dataloader(dataset_name+'Val', model, args, is_train=False)
        else:
            train_loader = get_chosen_dataset(dataset_name+'Val', model, args, is_train=True)
            val_loader = get_chosen_dataset(dataset_name+'Val', model, args, is_train=False)

        # Use SGD optimizer with learning rate 1e-4
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss_function = torch.nn.CrossEntropyLoss()
        

        # Initialize variables for tracking the best model
        best_val_accuracy = float("-inf")  # Start with a very low validation accuracy
        best_model_state = None  

        # Training loop
        for epoch in range(EPOCHS[dataset_name]):
            avg_loss, accuracy = dataset_training(model, train_loader, optimizer, loss_function, device)
            print(f"Epoch {epoch + 1}/{EPOCHS[dataset_name]}  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.4f}")

            # Validation step after each epoch
            val_loss, val_accuracy = validate_model(model, val_loader, loss_function, device)
            print(f"Validation Loss: {val_loss:.4f}  Accuracy: {val_accuracy:.4f}")

            # Update the best model if the current validation accuracy is higher
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.image_encoder.state_dict()

        # Save the best fine-tuned model using the same logic as in `finetune.py`
        if best_model_state:
            if args.balanced:
                finetuned_save_path = "./encoders/balanced"
            else:
                finetuned_save_path = "./encoders/validation_based"
            
            if not os.path.isdir(finetuned_save_path):
                os.makedirs(finetuned_save_path, exist_ok=True)
            
            finetuned_save_path += f"/{dataset_name}_finetuned.pt"

            model.image_encoder.load_state_dict(best_model_state)
            model.image_encoder.save(finetuned_save_path)
            print(f"Best model for {dataset_name} saved at: {finetuned_save_path}")

        # Save the best validation accuracy for this dataset
        best_validation_results[dataset_name] = {"Best Validation Accuracy": best_val_accuracy}
        print(f"Best validation accuracy for {dataset_name}: {best_val_accuracy:.4f}")

    # Save the best validation accuracies to a JSON file
    if args.balanced:
        best_val_results_file = f"./json_results/balanced"
    else:
        best_val_results_file = f"./json_results/validation_based"
    if not os.path.isdir(best_val_results_file):
        os.makedirs(best_val_results_file, exist_ok=True)

    best_val_results_file += f"/{dataset_name}_best_val.json"
    
    with open(best_val_results_file, "w") as f:
        json.dump(best_validation_results, f, indent=4)
    print(f"Best validation results saved in: {best_val_results_file}")

# Train the model for one epoch
def dataset_training(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training ..."):
        data = maybe_dictionarize(batch)
        inputs, labels = data["images"].to(device), data["labels"].to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted_labels = predictions.max(1)
        correct_predictions += predicted_labels.eq(labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, correct_predictions / total_samples

# Validate the model
def validate_model(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation ..."):
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"].to(device), data["labels"].to(device)

            predictions = model(inputs)
            loss = loss_function(predictions, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted_labels = predictions.max(1)
            correct_predictions += predicted_labels.eq(labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, correct_predictions / total_samples

if __name__ == "__main__":
    args = parse_arguments()
    fine_tune_clip_model(args)
