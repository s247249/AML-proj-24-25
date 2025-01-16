from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head

import torch
import torch.optim as optim
import torch.nn as nn
from utils import fine_tune_model


datasets = {
  "DTD": 76,
  "EuroSAT": 12,
  "GTSRB": 11,
  "MNIST": 5,
  "RESISC45": 15,
  "SVHN": 4
  }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_arguments() # Result of CLI argument parsing


# Instantiate a full model architecture
encoder = ImageEncoder(args) # Pre-trained CLIP ViT backbone
encoder.to(device)

# Get MNIST open-vocabulary classifier
head = get_classification_head(args, "MNISTVal")
model = ImageClassifier(encoder, head) # Build full model
model.freeze_head() # Freeze the classification head

# Added
save_path = "/content/AML-proj-24-25/results/"
model.image_encoder.save(save_path + "MNIST_zeroshot.pt")

model.to(device)

# Obtain the Train split of the "MNIST" dataset
dataset_train = get_dataset(
  "MNISTVal", preprocess=model.train_preprocess,
  location=args.data_location, batch_size=32, num_workers=2)
train_loader = get_dataloader(dataset, is_train=True, args=args)

# Obtain the Validation split of the "MNIST" dataset
dataset_val = get_dataset(
  "MNISTVal", preprocess=model.val_preprocess,
  location=args.data_location, batch_size=32, num_workers=2)
val_loader = get_dataloader(dataset, is_train=False, args=args)

# Obtain the Test split of the "MNIST" dataset
dataset_test = get_dataset(
  "MNIST", preprocess=model.val_preprocess,
  location=args.data_location, batch_size=32, num_workers=2)
test_loader = get_dataloader(dataset, is_train=False, args=args)

# Loss function
criterion = nn.CrossEntropyLoss()
# SGD Optimizer with lr=1-4
optimizer = optim.SGD(model.image_encoder.parameters(), lr=1e-4)

epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}

fine_tune_model(model, train_loader, val_loader, epochs["MNIST"], optimizer, criterion, device)

# Save fine-tuned weights (don’t need to store classification heads)
model.image_encoder.save(save_path + "MNIST_finetuned.pt")
