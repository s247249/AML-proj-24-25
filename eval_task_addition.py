import torch
import json

from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from utils import get_chosen_dataset, rebuild_zeroshot, load_model, evaluate_accuracy, find_best_alpha
from task_vectors import NonLinearTaskVector
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

encoders_dir = "/content/AML-proj-24-25/encoders/"
task_paths = [
    (encoders_dir+"DTD_zeroshot.pt", encoders_dir+"DTD_finetuned.pt"),
    (encoders_dir+"EuroSAT_zeroshot.pt", encoders_dir+"EuroSAT_finetuned.pt"),
    (encoders_dir+"GTSRB_zeroshot.pt", encoders_dir+"GTSRB_finetuned.pt"),
    (encoders_dir+"MNIST_zeroshot.pt", encoders_dir+"MNIST_finetuned.pt"),
    (encoders_dir+"RESISC45_zeroshot.pt", encoders_dir+"RESISC45_finetuned.pt"),
    (encoders_dir+"SVHN_zeroshot.pt", encoders_dir+"SVHN_finetuned.pt"),
]   

task_vectors = []
for pt_path, ft_path in task_paths:
    task_vector = NonLinearTaskVector(pt_path, ft_path)
    task_vectors.append(task_vector)

task_vec_add = sum(task_vectors)

for dataset in datasets:
    rebuild_zeroshot (dataset, device, args)

alpha, avg_norm_acc = find_best_alpha(encoders_dir, datasets, task_vec_add, args, device)

avg_abs_accuracy = 0.0
for dataset in datasets:

    pt_path = encoders_dir+dataset+"_zeroshot.pt"
    merged_encoder = task_vec_add.apply_to(pt_path, scaling_coef=alpha)
    head = get_classification_head(args, dataset+"Val")
    merged_model = ImageClassifier(merged_encoder, head)

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

with open("/content/AML-proj-24-25/json_results/alpha_results.json", 'w') as f:
    json.dump(results, f)

print(f"{results}")




