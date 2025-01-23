import torch
from tqdm import tqdm
from torch.utils.data import random_split
from args import parse_arguments
from datasets.common import maybe_dictionarize, get_dataloader
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head

# Epoch for each dataset
EPOCHS = {
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SVHN": 4
}

# Fine-tuning function
def fine_tune_clip_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over all datasets in EPOCHS
    for dataset_name in EPOCHS.keys():
        print(f"\nFine-tuning on the dataset : {dataset_name}")

        image_encoder = ImageEncoder(args).to(device)

        # Saving pretrained weights
        pretrained_save_path = f"{args.save}/{dataset_name}_zeroshot.pt"
        image_encoder.save(pretrained_save_path)
        print(f"Pre-trained weights saved at : {pretrained_save_path}")

        # Charging the head_classification
        classification_head = get_classification_head(args, dataset_name + "Val")
        model = ImageClassifier(image_encoder, classification_head).to(device)
        model.freeze_head() 
        
        # Using SGD Optimizer with learning rate 1e-4
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss_function = torch.nn.CrossEntropyLoss()

        # Charging train, val, and test datasets 
        train_dataset = get_dataset(
            dataset_name + "Val", preprocess=model.train_preprocess,
            location=args.data_location,batch_size=args.batch_size
        )
        val_dataset = get_dataset(
            dataset_name + "Val", preprocess=model.val_preprocess,
            location=args.data_location, batch_size=args.batch_size
        )

        train_loader = get_dataloader(train_dataset, is_train=True, args=args)
        val_loader = get_dataloader(val_dataset, is_train=False, args=args)
    

        # Training
        for epoch in range(EPOCHS[dataset_name]):
            avg_loss, accuracy = dataset_training(model, train_loader, optimizer, loss_function, device)
            print(f"Epoch {epoch + 1}/{EPOCHS[dataset_name]}  Loss : {avg_loss:.4f}  Accuracy : {accuracy:.4f}")

            # Validation after each epoch
            val_loss, val_accuracy = validate_model(model, val_loader, loss_function, device)
            print(f"Validation Part,  Loss : {val_loss:.4f}  Accuracy : {val_accuracy:.4f}")

        # Saving the fine-tuned model
        finetuned_save_path = f"{args.save}/{dataset_name}_finetuned.pt"
        model.image_encoder.save(finetuned_save_path)
        print(f"{dataset_name} used for fine-tuning, saving at : {finetuned_save_path}")

# Train the model on one epoch
def dataset_training(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    prediction_ok = 0
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
        prediction_ok += predicted_labels.eq(labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, prediction_ok / total_samples


# Model validation
def validate_model(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0.0
    prediction_ok = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation ..."):
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"].to(device), data["labels"].to(device)

            predictions = model(inputs)
            loss = loss_function(predictions, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted_labels = predictions.max(1)
            prediction_ok += predicted_labels.eq(labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, prediction_ok / total_samples

if __name__ == "__main__":
    args = parse_arguments()
    fine_tune_clip_model(args)


# python finetune.py --data-location /content/dataset/ --save /content/checkpoint --batch-size 64