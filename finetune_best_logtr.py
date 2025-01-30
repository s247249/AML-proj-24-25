import torch
from tqdm import tqdm
from args import parse_arguments
from datasets.common import maybe_dictionarize
from utils import load_zeroshot, train_diag_fim_logtr, get_chosen_dataset, build_zeroshot, get_balanced_dataloader
import json
import os

EPOCHS = {
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SVHN": 4,
}

def fine_tune_clip_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        best_logtr_fim = float("-inf")
        best_model_state = None

        for epoch in range(EPOCHS[dataset_name]):
            avg_loss, accuracy = dataset_training(model, train_loader, optimizer, loss_function, device)
            print(f"Epoch {epoch + 1}/{EPOCHS[dataset_name]}  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.4f}")

            val_loss, val_accuracy = validate_model(model, val_loader, loss_function, device)
            print(f"Validation Loss: {val_loss:.4f}  Accuracy: {val_accuracy:.4f}")

            samples_to_compute = 2000
            logtr_fim = train_diag_fim_logtr(args, model, dataset_name, samples_to_compute)
            print(f"Diagonal FIM log-trace: {logtr_fim:.4f}")

            if logtr_fim > best_logtr_fim:
                best_logtr_fim = logtr_fim
                best_model_state = model.image_encoder.state_dict()

        if best_model_state:
            finetuned_save_path = "./encoders/log_trace"
            
            if not os.path.isdir(finetuned_save_path):
                os.makedirs(finetuned_save_path, exist_ok=True)
            
            finetuned_save_path += f"/{dataset_name}_finetuned.pt"

            model.image_encoder.load_state_dict(best_model_state)  
            model.image_encoder.save(finetuned_save_path)  
            print(f"Best model for {dataset_name} saved at: {finetuned_save_path}")

        logtr_fim_file = "./json_results/log_trace"
        if not os.path.isdir(logtr_fim_file):
            os.makedirs(logtr_fim_file, exist_ok=True)
        
        logtr_fim_file += f"/{dataset_name}_best_logtr_fim.json"

        with open(logtr_fim_file, "w") as f:
            json.dump({"Dataset": dataset_name, "Best LogTr FIM": best_logtr_fim}, f, indent=4)
        print(f"Best LogTr FIM for {dataset_name} saved in: {logtr_fim_file}")
        


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


# python finetune_best_logtr.py --data-location /content/dataset/ --save /content/checkpoint --batch-size 32