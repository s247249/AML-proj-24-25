import torch
import json
import numpy as np
from args import parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader
from modeling import ImageClassifier
from heads import get_classification_head
from task_vectors import NonLinearTaskVector
from datasets.common import maybe_dictionarize
from tqdm import tqdm
import utils



if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset names
    dataset_names = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]

    # Paths for weights
    base_path = f"/content/Model/Batch{args.batch_size}"
    task_vectors = []

    # # Load task vectors and compute fine-tuned accuracies
    fine_tuned_accuracies_validation_split = {}
    fine_tuned_accuracies_test_split = {}
    fine_tuned_accuracies_train_split = {}


    # Initialize fine-tuned accuracies for testing
    # fine_tuned_accuracies_validation_split = {name: 0.9 for name in dataset_names}  
    # fine_tuned_accuracies_test_split = {name: 0.85 for name in dataset_names}    
    # fine_tuned_accuracies_train_split = {name: 0.85 for name in dataset_names}   

    # Creating the task vector list, the normal accuracy of the model on each dataset (relevant for computing the normal accuracy)
    for dataset_name in dataset_names:
        pt_path = f"{base_path}/zeroshot/{dataset_name}_zeroshot.pt"
        ft_path = f"{base_path}/finetune/{dataset_name}_finetuned.pt"
        task_vector = NonLinearTaskVector(pt_path, ft_path)
        task_vectors.append(task_vector)

        classification_head = get_classification_head(args, dataset_name + "Val")
        model = ImageClassifier(task_vector.apply_to(pt_path, scaling_coef=1.0), classification_head).to(device)
        
        # Validation, Test and Train loader
        validation_loader = utils.get_split_loader(dataset_name,"Validation", model,args=args)
        test_loader = utils.get_split_loader(dataset_name,"Test",model, args=args)
        train_loader = utils.get_split_loader(dataset_name,"Train",model, args=args)


        fine_tuned_accuracies_validation_split[dataset_name] = utils.evaluate_model(model, validation_loader, device, split="Validation")
        fine_tuned_accuracies_test_split[dataset_name] = utils.evaluate_model(model, test_loader, device, split="Test")
        fine_tuned_accuracies_train_split[dataset_name] = utils.evaluate_model(model, train_loader, device, split="Train")
          

    # Find the best alpha
    best_alpha, best_ana, alpha_results = utils.find_best_alpha(
        task_vectors, dataset_names, fine_tuned_accuracies_validation_split,"Validation", args
    )
    # best_alpha, best_ana, alpha_results = 0.5, 0.95, [{"Alpha": 0.5, "Average Normalized Accuracy": 0.95}]

  
    # Save alpha results
    save_path = f"{args.save}"
    alpha_results_file = f"{save_path}/alpha_results_task_addition_batch:{args.batch_size}_scale:{best_alpha}.json"
    with open(alpha_results_file, "w") as f:
        json.dump(alpha_results, f, indent=4)
    print(f"Alpha results saved in: {alpha_results_file}")

    # Evaluate on test split with the best alpha
    test_results = utils.evaluate_on_split(best_alpha, task_vectors, dataset_names, fine_tuned_accuracies_test_split,"Test", args)

    # Save test results
    test_results_file = f"{save_path}/test_results_task_addition_batch:{args.batch_size}_scale:{best_alpha}.json"
    with open(test_results_file, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved in: {test_results_file}")

    # Evaluate on train split with the best alpha
    train_results = utils.evaluate_on_split(best_alpha, task_vectors, dataset_names, fine_tuned_accuracies_train_split,"Train", args)

    # Save test results
    train_results_file = f"{save_path}/train_results_task_addition_batch:{args.batch_size}_scale:{best_alpha}.json"
    with open(train_results_file, "w") as f:
        json.dump(train_results, f, indent=4)
    print(f"Test results saved in: {train_results_file}")

# python eval_task_addition.py --data-location /content/dataset/ --save /content/task_addition_result/ --batch-size 8