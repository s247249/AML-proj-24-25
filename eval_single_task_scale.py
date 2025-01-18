import torch
import json
from args import parse_arguments
from datasets.registry import get_dataset
from args import parse_arguments
from datasets.common import get_dataloader
from modeling import ImageClassifier
from heads import get_classification_head
from task_vectors import NonLinearTaskVector
from datasets.common import maybe_dictionarize
from tqdm import tqdm
from utils import train_diag_fim_logtr

def evaluate_model(model, dataloader, device,eval_set=0):
    model.eval()
    total_samples = 0
    correct_predictions = 0
    set = "train set"
    if eval_set: 
        set = "eval set"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {set}  ..."):
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"].to(device), data["labels"].to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct_predictions += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    return correct_predictions / total_samples


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose dataset name
    dataset_name = "SVHN"  

    # Paths for pretrained and fine-tuned checkpoints
    pt_path = f"/content/drive/MyDrive/Model/zeroshot/{dataset_name}_zeroshot.pt"
    ft_path = f"/content/drive/MyDrive/Model/finetune/{dataset_name}_fine_tuned.pt"

    # Load the fine-tuned encoder
    task_vector = NonLinearTaskVector(pt_path, ft_path)

    # Initialize best results and results for all alpha values
    best_alpha = None
    best_logtr_hF = float("-inf")
    best_results = None
    all_results = []  

    # Iterate over alpha values
    alpha_values = [round(i * 0.05, 2) for i in range(21)]  
    for alpha in alpha_values:
        print(f"Testing with alpha = {alpha}")

        # Apply scaling to the task vector
        scaled_encoder = task_vector.apply_to(pt_path, scaling_coef=alpha)

        # Load the classification head
        classification_head = get_classification_head(args, dataset_name + "Val")
        model = ImageClassifier(scaled_encoder, classification_head).to(device)

        train_dataset = get_dataset(
        dataset_name + "Val", preprocess=model.train_preprocess,
        location=args.data_location, batch_size=32
        )
   
        test_dataset = get_dataset(
        dataset_name, preprocess=model.val_preprocess,
        location=args.data_location, batch_size=32
        )

        train_loader = get_dataloader(train_dataset, is_train=True, args=args)
        test_loader = get_dataloader(test_dataset, is_train=False, args=args)

        # Evaluate on train and test sets
        train_accuracy = evaluate_model(model, train_loader, device)
        test_accuracy = evaluate_model(model, test_loader, device,1)

        # Compute logtr_hF
        samples_nr = 2000  # Number of samples to accumulate gradients
        logtr_hF = train_diag_fim_logtr(args, model, dataset_name, samples_nr)

        print(f"Alpha = {alpha}, LogTr_hF = {logtr_hF:.4f}")

        # Save results for this alpha
        current_results = {
            "Alpha": alpha,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "LogTr_hF": logtr_hF
        }
        all_results.append(current_results)

        # Update the best results
        if logtr_hF > best_logtr_hF:
            best_logtr_hF = logtr_hF
            best_alpha = alpha
            best_results = current_results

    # Save all results to JSON
    save_path = f"{args.save}"  # Define the save path
    all_results_file = f"{save_path}/all_results_{dataset_name}_scaling.json"
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # Save the best results to JSON
    best_results_file = f"{save_path}/best_results_{dataset_name}_scaling.json"
    with open(best_results_file, "w") as f:
        json.dump(best_results, f, indent=4)

    print(f"Best Alpha: {best_alpha}")
    print(f"Best LogTr_hF: {best_logtr_hF:.4f}")
    print(f"Results saved in: {all_results_file}")
    print(f"Best results saved in: {best_results_file}")


   
python eval_single_task_scale.py --data-location /content/dataset/ --save /content/eval_single_task_scale_result/
    
