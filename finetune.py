import torch
import torch.optim as optim
import torch.nn as nn
import os

from utils import get_chosen_dataset, fine_tune_model, get_balanced_dataloader

from args import parse_arguments
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head


datasets = {
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SVHN": 4
    }
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_arguments() # Result of CLI argument parsing

for dataset in args.train_dataset:
    # Instantiate a full model architecture
    encoder = ImageEncoder(args) # Pre-trained CLIP ViT backbone
    encoder.to(device)

    # Get chosen_dataset open-vocabulary classifier
    head = get_classification_head(args, dataset+"Val")
    model = ImageClassifier(encoder, head) # Build full model
    model.freeze_head() # Freeze the classification head

    save_path = "./encoders"
    model.image_encoder.save(save_path + "/zeroshot.pt")

    model.to(device)

    if args.balanced:
        train_loader = get_balanced_dataloader(dataset+'Val', model, args, is_train=True)
        val_loader = get_balanced_dataloader(dataset+'Val', model, args, is_train=False)
    else:
        train_loader = get_chosen_dataset(dataset+'Val', model, args, is_train=True)
        val_loader = get_chosen_dataset(dataset+'Val', model, args, is_train=False)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # SGD Optimizer
    optimizer = optim.SGD(model.image_encoder.parameters(), lr=args.lr, weight_decay=args.wd)

    fine_tune_model(model, train_loader, val_loader, datasets[dataset], optimizer, loss_fn, device)

    if args.balanced:
        save_path += "/balanced"
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
    elif not args.batch_size==32:
        save_path += "/bs_" +str(args.batch_size)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
    elif not args.lr==1e-4:
        save_path += "/lr_" + str(args.lr)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
    elif not args.wd==0.0:
        save_path += "/wd_" + str(args.wd)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
    else:
        save_path += "/base"
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)



    # Save fine-tuned weights (don’t need to store classification heads)
    model.image_encoder.save(save_path + "/" + dataset+"_finetuned.pt")
