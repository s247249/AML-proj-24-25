import torch
import json

from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from utils import get_chosen_dataset, rebuild_zeroshot, load_model, evaluate_accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse_arguments()


for dataset in args.eval_datasets:
  # Used for colab
  rebuild_zeroshot (dataset, device, args)

  model = load_model(dataset, args)
  model.to(device)

  train_loader = get_chosen_dataset(dataset+'Val', model, args, is_train=True)
  val_loader = get_chosen_dataset(dataset+'Val', model, args, is_train=False)
  test_loader = get_chosen_dataset(dataset, model, args, is_train=False)

  from utils import train_diag_fim_logtr
  samples_nr = 2000 # How many per-example gradients to accumulate
  logdet_hF = train_diag_fim_logtr(args, model, dataset, samples_nr)

  train_accuracy = evaluate_accuracy(model, train_loader, device)
  val_accuracy = evaluate_accuracy(model, val_loader, device)
  test_accuracy = evaluate_accuracy(model, test_loader, device)

  # Save results to JSON file
  results = {
    'training_accuracy': train_accuracy,
    'validation_accuracy': val_accuracy,
    'test_accuracy': test_accuracy
  }

  with open("/content/AML-proj-24-25/json_results/"+dataset+"_results.json", 'w') as f:
      json.dump(results, f)

  print(f"\nDataset: {dataset}")
  print(f"Training Accuracy: {train_accuracy:.4f}")
  print(f"Validation Accuracy: {val_accuracy:.4f}")
  print(f"Test Accuracy: {test_accuracy:.4f}")