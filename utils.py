import os
import pickle

import numpy as np
import math
import torch
from tqdm.auto import tqdm
from datasets.common import get_dataloader, maybe_dictionarize, SubsetSampler
from datasets.registry import get_dataset, GenericDataset

from task_vectors import NonLinearTaskVector
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from collections import Counter
from torch.utils.data import Subset
import random


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
    if args.balanced:
        data_loader = rebalance_dataset(dataset.train_dataset, args, is_train=False)
    else:
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

##################################################
# Added Functions:
##################################################


def get_chosen_dataset(chosen_dataset, model, args, is_train=False):
    """
    Function to load the requested dataset split (training, validation or test).
    Args:
        chosen_dataset: string containing the name of the dataset. Add 'Val' for training or validation split
        model: the model to be used
        args: provided args
        is_train: default=False. Set to True to obtain the training split 
    """

    if is_train:
        prep = model.train_preprocess
    else:
        prep = model.val_preprocess
    
    dataset = get_dataset(
    chosen_dataset, preprocess=prep,
    location=args.data_location, batch_size=args.batch_size, num_workers=2)
    dataset_loader = get_dataloader(dataset, is_train=is_train, args=args)

    return dataset_loader


def fine_tune_model(model, train_loader, val_loader, num_epochs, optimizer, loss_fn, device):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            data = maybe_dictionarize(batch)
            images, labels = data["images"].to(device), data["labels"].to(device)
            
            model.train()

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print statistics for the epoch
        train_accuracy = 100 * correct / total
        print(f"\n\tEpoch {epoch + 1}/{num_epochs}: \nTraining Loss: {running_loss / len(train_loader)} \nTraining Accuracy: {train_accuracy}%")
        

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                data = maybe_dictionarize(batch)
                images, labels = data["images"].to(device), data["labels"].to(device)
                
                model.eval()
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Loss: {val_loss / len(val_loader)} \nValidation Accuracy: {val_accuracy}%")


def build_zeroshot(chosen_dataset, device, args):
    # Remaking the zeroshot checkpoint to speed up operations in colab

    # Instantiate a full model architecture
    encoder = ImageEncoder(args) # Pre-trained CLIP ViT backbone
    encoder.to(device)

    # Get chosen_dataset open-vocabulary classifier
    head = get_classification_head(args, chosen_dataset+"Val")
    model = ImageClassifier(encoder, head) # Build full model
    model.freeze_head() # Freeze the classification head

    # model.image_encoder.save("/content/AML-proj-24-25/encoders/zeroshot.pt")
    model.image_encoder.save("./encoders/zeroshot.pt")


# To avoid circular import in finetune_best_logtr.py and finetune_best_eval.py
def load_zeroshot(dataset_name, args, device):
    encoder = ImageEncoder(args) # Pre-trained CLIP ViT backbone
    encoder.to(device)

    # Get chosen_dataset open-vocabulary classifier
    head = get_classification_head(args, dataset_name+"Val")
    model = ImageClassifier(encoder, head) # Build full model
    model.freeze_head() # Freeze the classification head
    return model


def load_model(chosen_dataset, args):
    # ft_model_path = "/content/AML-proj-24-25/encoders"
    # merged_model_path = "/content/AML-proj-24-25/encoders"
    ft_model_path = "./encoders"
    
    if not args.batch_size==32:
        ft_model_path += "/bs_" +str(args.batch_size)
    elif not args.lr==1e-4:
        ft_model_path += "/lr_" + str(args.lr)
    elif not args.wd==0.0:
        ft_model_path += "/wd_" + str(args.lr)
    elif args.balanced:
        ft_model_path += "/balanced"
    else:
        ft_model_path += "/base"

    # pt_path = "/content/AML-proj-24-25/encoders/zeroshot.pt"
    pt_path = "./encoders/zeroshot.pt"
    ft_path = ft_model_path+"/"+chosen_dataset+"_finetuned.pt"

    task_vector = NonLinearTaskVector(pt_path, ft_path)
    
    # Get chosen_dataset open-vocabulary classifier
    head = get_classification_head(args, chosen_dataset+"Val")

    # The merged_model is already scaled
    # This if statement avoids erroneus input of alpha
    if args.merged:
        encoder = load_merged_encoder(ft_model_path + "/", alpha=args.alpha)
    else:
        encoder = task_vector.apply_to(pt_path, scaling_coef=args.alpha)
    model = ImageClassifier(encoder, head)
    model.freeze_head()
    return model


def load_merged_encoder(encoders_dir, alpha):
    # pt_path = "/content/AML-proj-24-25/encoders/zeroshot.pt"
    pt_path = "./encoders/zeroshot.pt"
    encoders_paths = [
        encoders_dir+"DTD_finetuned.pt",
        encoders_dir+"EuroSAT_finetuned.pt",
        encoders_dir+"GTSRB_finetuned.pt",
        encoders_dir+"MNIST_finetuned.pt",
        encoders_dir+"RESISC45_finetuned.pt",
        encoders_dir+"SVHN_finetuned.pt",
    ]   

    # Get task vectors
    task_vectors = []
    for ft_path in encoders_paths:
        task_vector = NonLinearTaskVector(pt_path, ft_path)
        task_vectors.append(task_vector)
    
    # Add task vectors
    task_vec_add = task_vectors[0]
    for i in range(1, len(task_vectors)):
        task_vec_add += task_vectors[i]

    # Build the merged encoder
    merged_encoder = task_vec_add.apply_to(pt_path, scaling_coef=alpha)
    return merged_encoder


def evaluate_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            data = maybe_dictionarize(batch)
            images, labels = data["images"].to(device), data["labels"].to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    
    return accuracy


def find_best_alpha(encoders_dir, results_dict, datasets, args, device):
    
    best_alpha = -1.0
    best_avg_norm_accuracy = 0.0
    
    
    
    for alpha in np.arange(0.0, 1.05, 0.05):
        norm_accuracy = 0.0
        trunc_alpha = math.trunc(alpha*100) / 100
        print(f"\nChecking results for value alpha = {trunc_alpha} ")

        merged_encoder = load_merged_encoder(encoders_dir, trunc_alpha)
        
        #test chosen alpha on all tasks
        for dataset in datasets: 
            # build the merged_model for specific dataset           
            head = get_classification_head(args, dataset+"Val")
            merged_model = ImageClassifier(merged_encoder, head)
            merged_model.freeze_head()
            merged_model.to(device)

            # load the dataset
            val_loader = get_chosen_dataset(dataset+'Val', merged_model, args, is_train=False)
            # evaluate accuracy for specific dataset
            merged_accuracy = evaluate_accuracy(merged_model, val_loader,  device)

            base_accuracy = results_dict[dataset].get('validation_accuracy')
            norm_accuracy += merged_accuracy / base_accuracy

        avg_norm_accuracy = norm_accuracy / len(datasets)

        if avg_norm_accuracy > best_avg_norm_accuracy:
            best_alpha = trunc_alpha
            best_avg_norm_accuracy = avg_norm_accuracy

    
    return best_alpha, best_avg_norm_accuracy * 100


def get_balanced_dataloader(chosen_dataset, model, args, is_train=False):
    
    if is_train:
        prep = model.train_preprocess
    else:
        prep = model.val_preprocess

    # Load the initial dataset
    dataset = get_dataset(
    chosen_dataset, preprocess=prep,
    location=args.data_location, batch_size=args.batch_size, num_workers=2)
    
    # Get a balanced dataset loader
    if is_train:
        dataset_loader = rebalance_dataset(dataset.train_dataset, args, is_train)
    else:
        dataset_loader = rebalance_dataset(dataset.test_dataset, args, is_train)

    return dataset_loader


# def rebalance_dataset(dataset, args, is_train=False):
#     if isinstance(dataset, Subset):
#         dataset = dataset.dataset

#     # Some dataseets use ._samples instead of .sample
#     if hasattr(dataset, '_samples'):
#         samples_attr = '_samples'
#     elif hasattr(dataset, 'samples'):
#         samples_attr = 'samples'

        
        
#     labels = [label for _, label in getattr(dataset, samples_attr)]
#     class_counts = Counter(labels)
    
#     # Least represented class count
#     min_count = min(class_counts.values())

#     # Create a list of indices for rebalancing
#     class_indices = {class_label: [] for class_label in class_counts}
#     for idx, (_, label) in enumerate(getattr(dataset, samples_attr)):
#         class_indices[label].append(idx)

#     # Create a list to hold the new indices with balanced representation
#     balanced_indices = []
#     for class_label, indices in class_indices.items():
#         # Take only `min_count` samples from this class (determinism)
#         balanced_indices.extend(indices[:min_count])
    
#     if is_train:
#         random.shuffle(balanced_indices)

#     sampler = SubsetSampler(balanced_indices)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
#     return dataloader



# def rebalance_dataset(dataset, args, is_train=False):
#     if isinstance(dataset, Subset):
#         dataset = dataset.dataset

#     # Handle datasets with different attribute names for samples
#     if hasattr(dataset, '_samples'):
#         samples_attr = '_samples'
#     elif hasattr(dataset, 'samples'):
#         samples_attr = 'samples'
#     elif hasattr(dataset, 'data') and hasattr(dataset, 'targets'):  # Special case for MNIST
#         samples_attr = 'mnist'
#     else:
#         raise AttributeError(f"Dataset does not have expected attributes for sample data.")
    
#     # If using MNIST, we use 'data' and 'targets' instead of 'samples'
#     if samples_attr == 'mnist':
#         data = dataset.data
#         labels = dataset.targets
#     else:
#         data = getattr(dataset, samples_attr)
#         labels = [label for _, label in data]

#     # Ensure that labels are integers (this is important for consistency)
#     if samples_attr == 'mnist':
#         labels = labels.tolist()  # Convert tensor to a list of integers
#     else:
#         labels = [label for _, label in data]  # For other datasets, we already have this in list format

#     class_counts = Counter(labels)
    
#     # Least represented class count
#     min_count = min(class_counts.values())

#     # Create a list of indices for rebalancing
#     class_indices = {class_label: [] for class_label in class_counts}
    
#     if samples_attr == 'mnist':  # Special case for MNIST
#         for idx, label in enumerate(labels):
#             class_indices[label].append(idx)  # Here, label is already an int
#     else:
#         for idx, (_, label) in enumerate(data):
#             class_indices[label].append(idx)

#     # Create a list to hold the new indices with balanced representation
#     balanced_indices = []
#     for class_label, indices in class_indices.items():
#         # Take only `min_count` samples from this class (determinism)
#         balanced_indices.extend(indices[:min_count])
    
#     if is_train:
#         random.shuffle(balanced_indices)

#     sampler = SubsetSampler(balanced_indices)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
#     return dataloader

def rebalance_dataset(dataset, args, is_train=False):
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    # Handle datasets with different attribute names for samples
    if hasattr(dataset, '_samples'):
        samples_attr = '_samples'
    elif hasattr(dataset, 'samples'):
        samples_attr = 'samples'
    elif hasattr(dataset, 'data') and hasattr(dataset, 'targets'):  # Special case for MNIST
        samples_attr = 'mnist'
    elif hasattr(dataset, 'data') and hasattr(dataset, 'labels'):  # Special case for SVHN
        samples_attr = 'svhn'
    else:
        raise AttributeError(f"Dataset does not have expected attributes for sample data.")
    
    # If using MNIST, we use 'data' and 'targets' instead of 'samples'
    if samples_attr == 'mnist':
        data = dataset.data
        labels = dataset.targets
    elif samples_attr == 'svhn':  # Special case for SVHN
        data = dataset.data
        labels = dataset.labels
    else:
        data = getattr(dataset, samples_attr)
        labels = [label for _, label in data]

    # Ensure that labels are integers (this is important for consistency)
    if samples_attr == 'mnist' or samples_attr == 'svhn':
        labels = labels.tolist()  # Convert tensor to a list of integers
    else:
        labels = [label for _, label in data]  # For other datasets, we already have this in list format

    class_counts = Counter(labels)
    
    # Least represented class count
    min_count = min(class_counts.values())

    # Create a list of indices for rebalancing
    class_indices = {class_label: [] for class_label in class_counts}
    
    if samples_attr == 'mnist' or samples_attr == 'svhn':  # Special case for MNIST and SVHN
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)  # Now label is an int
    else:
        for idx, (_, label) in enumerate(data):
            class_indices[label].append(idx)

    # Create a list to hold the new indices with balanced representation
    balanced_indices = []
    for class_label, indices in class_indices.items():
        # Take only `min_count` samples from this class (determinism)
        balanced_indices.extend(indices[:min_count])
    
    if is_train:
        random.shuffle(balanced_indices)

    sampler = SubsetSampler(balanced_indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    return dataloader