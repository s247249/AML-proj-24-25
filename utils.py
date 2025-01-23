import os
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
import modeling 
import heads




def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu")
    if device is not None:
        model = model.to(device)
    return model


class DotDict(dict): 
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def train_diag_fim_logtr(
            args,
            model,
            dataset_name: str,
            samples_nr: int = 2000):
    
    model.cuda()
    if not dataset_name.endswith('Val'):
        dataset_name += 'Val'

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=0
    )
    data_loader = torch.utils.data.DataLoader(
        dataset.train_dataset, 
        batch_size=args.batch_size, 
        num_workers=0, shuffle=False)
    
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    progress_bar = tqdm(total=samples_nr)
    seen_nr = 0

    while seen_nr < samples_nr:
        data_iterator = iter(data_loader)
        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data = next(data_loader)
        data = maybe_dictionarize(data)
        x, y = data['images'], data['labels']
        x, y = x.cuda(), y.cuda()

        logits = model(x)
        outdx = torch.distributions.Categorical(logits=logits).sample().unsqueeze(1).detach()
        samples = logits.gather(1, outdx)

        idx, batch_size = 0, x.size(0)

        for idx in range(batch_size):
            model.zero_grad()
            torch.autograd.backward(samples[idx], retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad and hasattr(param, 'grad') and param.grad is not None:
                    fim[name] += (param.grad * param.grad)
                    fim[name].detach_()
            seen_nr += 1
            progress_bar.update(1)
            if seen_nr >= samples_nr: break

    fim_trace = 0.0
    for name, grad2 in fim.items():
        fim_trace += grad2.sum()
    fim_trace = torch.log(fim_trace / samples_nr).item()

    return fim_trace

# Access dataset more easily
def get_split_loader(dataset_name, split, model, args):
    if split == "Validation":
        dataset = get_dataset(
            dataset_name + "Val", preprocess=model.val_preprocess,
            location=args.data_location, batch_size=args.batch_size, num_workers=2
        )
        loader = get_dataloader(dataset, is_train=False, args=args)
    elif split == "Test":
        dataset = get_dataset(
            dataset_name, preprocess=model.val_preprocess,
            location=args.data_location, batch_size=args.batch_size, num_workers=2
        )
        loader = get_dataloader(dataset, is_train=False, args=args)
    else:  # Assuming "Train" split
        dataset = get_dataset(
            dataset_name + "Val", preprocess=model.val_preprocess,
            location=args.data_location, batch_size=args.batch_size, num_workers=2
        )
        loader = get_dataloader(dataset, is_train=True, args=args)

    return loader



# Function to evaluate the model accuracy 
def evaluate_model(model, dataloader, device, split):
    model.eval()
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {split} split ..."):
            data = maybe_dictionarize(batch)
            inputs, labels = data["images"].to(device), data["labels"].to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct_predictions += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    return correct_predictions / total_samples

"""
eval_task_addition_scale.py related function
"""
# Function to find the best alpha based on normalized accuracy
def find_best_alpha(task_vectors, dataset_names, fine_tuned_accuracies,split, args):
    alpha_values = [round(i * 0.05, 2) for i in range(21)]
    best_alpha = None
    best_ana = float("-inf")
    results = []

    for alpha in alpha_values:
        print(f"Testing with alpha = {alpha}")
        normalized_accuracies, _ = compute_absolute_and_normalized(
            alpha, task_vectors, dataset_names, fine_tuned_accuracies,split, args
        )
        current_ana = np.mean(normalized_accuracies)
        results.append({"Alpha": alpha, "Average Normalized Accuracy": current_ana})

        if current_ana > best_ana:
            best_ana = current_ana
            best_alpha = alpha

        print(f"Alpha = {alpha}, Average Normalized Accuracy the {split} split = {current_ana:.4f}")

    return best_alpha, best_ana, results


"""
eval_task_addition.py related functions
"""

# Function to find the best alpha based on normalized accuracy
def find_best_alpha(task_vectors, dataset_names, fine_tuned_accuracies,split, args):
    alpha_values = [round(i * 0.05, 2) for i in range(21)]
    best_alpha = None
    best_ana = float("-inf")
    results = []

    for alpha in alpha_values:
        print(f"Testing with alpha = {alpha}")
        normalized_accuracies, _ = compute_absolute_and_normalized(
            alpha, task_vectors, dataset_names, fine_tuned_accuracies,split, args
        )
        current_ana = np.mean(normalized_accuracies)
        results.append({"Alpha": alpha, "Average Normalized Accuracy": current_ana})

        if current_ana > best_ana:
            best_ana = current_ana
            best_alpha = alpha

        print(f"Alpha = {alpha}, Average Normalized Accuracy the {split} split = {current_ana:.4f}")

    return best_alpha, best_ana, results


# Function to compute absolute accuracies given alpha, task vectors, datasets, but also the logtr if the split is "Train"
def compute_absolute_accuracy_and_logtr(alpha, task_vectors, dataset_names, split, args):
    merged_vector = sum(task_vector * alpha for task_vector in task_vectors)
    merged_encoder = merged_vector.apply_to(f"/content/Model/Batch{args.batch_size}/zeroshot/DTD_zeroshot.pt", scaling_coef=1.0)
  
    absolute_accuracies = []
    logtr_all = {}
    for dataset_name in dataset_names:
        # Get the classification head for each datasets
        classification_head = heads.get_classification_head(args, dataset_name + "Val")
        model = modeling.ImageClassifier(merged_encoder, classification_head).to(args.device)
        loader= get_split_loader(dataset_name, split, model, args)
        accuracy = evaluate_model(model, loader, args.device, split)
        absolute_accuracies.append(accuracy)
        if split == "Train":
            # Compute logtr_hF
            samples_nr = 2000  # Number of samples to accumulate gradients
            logtr = train_diag_fim_logtr(args, model, dataset_name, samples_nr)
            logtr_all[dataset_name] = logtr

    if split == "Train":
        return absolute_accuracies, logtr_all
    return absolute_accuracies

# Function to compute absolute accuracy, normalized accuracy and logtr (for Train split)
def compute_absolute_and_normalized(alpha, task_vectors, dataset_names, fine_tuned_accuracies, split, args):
    if split == "Train":
        absolute_accuracies, logtr = compute_absolute_accuracy_and_logtr(alpha, task_vectors, dataset_names, split, args)
    else:
        absolute_accuracies = compute_absolute_accuracy_and_logtr(alpha, task_vectors, dataset_names, split, args)  

    normalized_accuracies = []

    for i in range(len(dataset_names)):
        abs_acc = absolute_accuracies[i]
        dataset_name = dataset_names[i]
        norm_acc = abs_acc / fine_tuned_accuracies[dataset_name]
        normalized_accuracies.append(norm_acc)

    if split == "Train":
        return absolute_accuracies, normalized_accuracies, logtr

    return absolute_accuracies, normalized_accuracies

# Function to evaluate on test splits and compute averages
def evaluate_on_split(best_alpha, task_vectors, dataset_names, fine_tuned_accuracies,split, args):
    # Compute absolute and normalized accuracies
    if split == "Train":
         absolute_accuracies, normalized_accuracies, logtr = compute_absolute_and_normalized(best_alpha, task_vectors, dataset_names, fine_tuned_accuracies, split, args)
    else:
        absolute_accuracies, normalized_accuracies = compute_absolute_and_normalized(
        best_alpha, task_vectors, dataset_names, fine_tuned_accuracies,split, args
    )

    # Initialize results container
    results = []
    results.append({
            f"Best alpha evaluated on {split} split": best_alpha,
        })
    
    if split == "Train":
        for dataset_name, abs_acc, norm_acc in zip(dataset_names, absolute_accuracies, normalized_accuracies):
            results.append({ 
                "Dataset": dataset_name,
                f"Absolute Accuracy on {split} split": abs_acc,
                f"Normalized Accuracy on {split} split": norm_acc,
                "logtr": logtr[dataset_name]
            })

        # Calculate averages
        avg_absolute_accuracy = sum(absolute_accuracies) / len(absolute_accuracies)
        avg_normalized_accuracy = sum(normalized_accuracies) / len(normalized_accuracies)

        # Append averages to the results
        results.append({
            f"Average Absolute Accuracy on {split} split": avg_absolute_accuracy,
            f"Average Normalized Accuracy on {split} split": avg_normalized_accuracy
        })


        
    
    else:
        for dataset_name, abs_acc, norm_acc in zip(dataset_names, absolute_accuracies, normalized_accuracies):
            results.append({
                "Dataset": dataset_name,
                f"Absolute Accuracy on {split} split": abs_acc,
                f"Normalized Accuracy on {split} split": norm_acc
            })

        # Calculate averages
        avg_absolute_accuracy = sum(absolute_accuracies) / len(absolute_accuracies)
        avg_normalized_accuracy = sum(normalized_accuracies) / len(normalized_accuracies)

        # Append averages to the results
        results.append({
            f"Average Absolute Accuracy on {split} split": avg_absolute_accuracy,
            f"Average Normalized Accuracy on {split} split": avg_normalized_accuracy
        })

    return results


