import torch
import json
import os

from args import parse_arguments
from utils import get_chosen_dataset, build_zeroshot, load_merged_encoder, evaluate_accuracy, find_best_alpha
from modeling import ImageClassifier
from heads import get_classification_head


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

json_dir = "/content/AML-proj-24-25/json_results/"
results_dict = {}

# rebuild zeroshot models (for colab)
if not os.path.isfile("/content/AML-proj-24-25/encoders/zeroshot.pt"):
    build_zeroshot (datasets[0], device, args)

for dataset in datasets:
    with open(json_dir + dataset +"_results.json", 'r') as f:
        results = json.load(f)
    results_dict[dataset] = results

encoders_dir = "/content/AML-proj-24-25/encoders/"

print("Searching for best alpha value")
alpha, avg_norm_acc = find_best_alpha(encoders_dir, results_dict, datasets, args, device)
avg_abs_accuracy = 0.0


merged_encoder = load_merged_encoder(encoders_dir, alpha)
for dataset in datasets:

    head = get_classification_head(args, dataset+"Val")
    merged_model = ImageClassifier(merged_encoder, head)
    merged_model.freeze_head()
    merged_model.to(device)

    # load the dataset
    test_loader = get_chosen_dataset(dataset, merged_model, args, is_train=False)
    # evaluate accuracy for specific dataset
    accuracy = evaluate_accuracy(merged_model, test_loader,  device)

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




