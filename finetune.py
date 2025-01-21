import torch
import torch.optim as optim
import torch.nn as nn

from utils import get_chosen_dataset, fine_tune_model

from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head


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

dataset = args.train_dataset

# Instantiate a full model architecture
encoder = ImageEncoder(args) # Pre-trained CLIP ViT backbone
encoder.to(device)

# Get chosen_dataset open-vocabulary classifier
head = get_classification_head(args, dataset+"Val")
model = ImageClassifier(encoder, head) # Build full model
model.freeze_head() # Freeze the classification head

# Added
save_path = "/content/AML-proj-24-25/encoders/"
model.image_encoder.save(save_path + dataset+"_zeroshot.pt")

model.to(device)

train_loader = get_chosen_dataset(dataset+'Val', model, args, is_train=True)
val_loader = get_chosen_dataset(dataset+'Val', model, args, is_train=False)

# Loss function
loss_fn = nn.CrossEntropyLoss()
# SGD Optimizer with lr=1-4
optimizer = optim.SGD(model.image_encoder.parameters(), lr=1e-4)

epochs = {"DTD": 76, "EuroSAT": 12, "GTSRB": 11, "MNIST": 5, "RESISC45": 15, "SVHN": 4}

fine_tune_model(model, train_loader, val_loader, epochs[dataset], optimizer, loss_fn, device)

# Save fine-tuned weights (don’t need to store classification heads)
model.image_encoder.save(save_path + dataset+"_finetuned.pt")
