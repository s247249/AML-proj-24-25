import os
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset

from task_vectors import NonLinearTaskVector
from modeling import ImageClassifier
from heads import get_classification_head


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
    location=args.data_location, batch_size=32, num_workers=2)
    dataset_loader = get_dataloader(dataset, is_train=False, args=args)

    return dataset_loader


def fine_tune_model(model, train_loader, val_loader, num_epochs, optimizer, loss_fn, device):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            data = maybe_dictionarize(batch)
            images, labels = data["images"].to(device), data["labels"].to(device)
            # Training
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
        print(f"\tEpoch {epoch + 1}/{num_epochs} \nTraining Loss: {running_loss / len(train_loader)} \nTraining Accuracy: {train_accuracy}%")
        

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

def load_model(chosen_dataset, args):
    pt_path, ft_path = "/path/to/"+chosen_dataset+"_zeroshot.pt", "/path/to/"+chosen_dataset+"_finetuned.pt"
    task_vector1 = NonLinearTaskVector(pt_path, ft_path)
    
    # Get chosen_dataset open-vocabulary classifier
    head = get_classification_head(args, chosen_dataset+"Val")
    encoder = task_vector1.apply_to(pt_path, scaling_coef=1.0)
    model = ImageClassifier(encoder, head)
    return model


def evaluate_accuracy(model, dataloader, device):
    model.eval()

    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            data = maybe_dictionarize(batch)
            images, labels = data["images"].to(device), data["labels"].to(device)
            
            model.eval()
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    accuracy = 100 * val_correct / val_total
    print(f"Validation Accuracy: {accuracy}%")
    
    return accuracy