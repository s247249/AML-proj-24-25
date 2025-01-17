import torch
import json

from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from utils import get_chosen_dataset,load_model, evaluate_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse_arguments()

datasets = {
  "DTD": 76,
  "EuroSAT": 12,
  "GTSRB": 11,
  "MNIST": 5,
  "RESISC45": 15,
  "SVHN": 4
  }

chosen_dataset = args.eval_dataset

model = load_model(chosen_dataset, args)
model.to(device)


val_loader = get_chosen_dataset(chosen_dataset+'Val', model, args, is_train=False)
test_loader = get_chosen_dataset(chosen_dataset, model, args, is_train=False)

val_accuracy = evaluate_accuracy(model, val_loader, device)
test_accuracy = evaluate_accuracy(model, test_loader, device)

# Save results to JSON file
results = {
    'validation_accuracy': val_accuracy + '%',
    'test_accuracy': test_accuracy + '%'
}

with open(""+"_results.json", 'w') as f:
    json.dump(results, f)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")