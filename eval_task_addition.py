import torch
import json

from args import parse_arguments
from utils import get_chosen_dataset, rebuild_zeroshot, load_merged_model, evaluate_accuracy, find_best_alpha
from heads import get_classification_head
from modeling import ImageClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse_arguments()

datasets = [
  "DTD",
  "EuroSAT",
  "GTSRB",
  "MNIST",
  "RESISC45",
  "SVHN"
  ]

json_dir = "/content/AML-proj-24-25/json results/"
results_dict = {}

for dataset in datasets:
    # rebuild zeroshot models (for colab)
    rebuild_zeroshot (dataset, device, args)
    with open(json_dir + dataset +"_results.json", 'r') as f:
        results = json.load(f)
    results_dict[dataset] = results

encoders_dir = "/content/AML-proj-24-25/encoders/"

print("Searching for best alpha value")
alpha, avg_norm_acc = find_best_alpha(encoders_dir, results_dict, datasets, args, device)

avg_abs_accuracy = 0.0
for dataset in datasets:

    merged_model = load_merged_model(encoders_dir, dataset, alpha, args)
    merged_model.to(device)

    # load the dataset
    test_loader = get_chosen_dataset(dataset, merged_model, args, is_train=False)

    accuracy = evaluate_accuracy(merged_model, test_loader,  device) / 100

    avg_abs_accuracy += accuracy

avg_abs_accuracy = avg_abs_accuracy / len(datasets)

results = {
    'alpha': alpha,
    'avg_norm_acc': avg_norm_acc,
    'avg_abs_accuracy': avg_abs_accuracy
}


with open(json_dir + "alpha_results.json", 'w') as f:
    json.dump(results, f)

print(f"{results}")




