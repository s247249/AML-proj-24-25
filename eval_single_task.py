import torch
import json
from datasets.registry import get_dataset
from args import parse_arguments
from datasets.common import get_dataloader
from modeling import ImageClassifier
from heads import get_classification_head
from task_vectors import NonLinearTaskVector
from datasets.common import maybe_dictionarize
from tqdm import tqdm
from utils import *

if __name__ == "__main__":
    args = parse_arguments()    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of dataset names
    dataset_names = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    # Scaling used 
    scaling = 0.3

    # Initialize results dictionary
    all_results = {}
    all_results[scaling] = {
        "Alpha used": scaling
    }
    
    
    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")

        # Paths for pretrained and fine-tuned checkpoints
        pt_path = f"/content/Model/Batch{args.batch_size}/zeroshot/{dataset_name}_zeroshot.pt"
        ft_path = f"/content/Model/Batch{args.batch_size}/finetune/{dataset_name}_finetuned.pt"

        # Load the fine-tuned encoder
        task_vector = NonLinearTaskVector(pt_path, ft_path)
        fine_tuned_encoder = task_vector.apply_to(pt_path, scaling)

        # Load the classification head
        classification_head = get_classification_head(args, dataset_name + "Val")
        model = ImageClassifier(fine_tuned_encoder, classification_head).to(device)

        # Obtain the Train split 
        train_loader = get_split_loader(dataset_name, "Train", model, args=args)

        # Obtain the Test split
        test_loader = get_split_loader(dataset_name, "Test", model, args=args)

        # Evaluate on train and test sets
        train_accuracy = evaluate_model(model, train_loader, device, "Train")
        # Compute logtr_hF
        samples_nr = 2000  # Number of samples to accumulate gradients
        logtr_hF = train_diag_fim_logtr(args, model, dataset_name, samples_nr)

        test_accuracy = evaluate_model(model, test_loader, device, "Test")
        
        # Store results for the current dataset
        all_results[dataset_name] = {
            "Test": test_accuracy,
            "Train": train_accuracy,
            "LogTr_hF": logtr_hF
        }

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"LogTr_hF: {logtr_hF}")

    # Save all results to JSON
    save_path = f"{args.save}"  # Define the save path
    output_file = f"{save_path}/eval_single_task_batch:{args.batch_size}_scaling:{scaling}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Results saved in: {output_file}")

# python eval_single_task.py --data-location /content/dataset/ --save /content/eval_all_datasets_result/ --batch-size 32
