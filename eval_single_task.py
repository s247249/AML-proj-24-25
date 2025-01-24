import torch
import json
import os

from args import parse_arguments
from utils import train_diag_fim_logtr, get_chosen_dataset, build_zeroshot, load_model, load_merged_encoder, evaluate_accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse_arguments()

testing = True

save_path = "/content/AML-proj-24-25/json_results"

if not args.batch_size==32:
    save_path += "/bs_" + str(args.batch_size)
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
# Change directory for merged model
if args.merged:
    save_path += "/merged"
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

save_path += "/"

# Used for colab# rebuild zeroshot model (for colab)
if not os.path.isfile("/content/AML-proj-24-25/encoders/zeroshot.pt"):
    build_zeroshot ("DTD", device, args)


for dataset in args.eval_datasets:
    if args.merged:
        model = load_model(dataset, args, merged=True)
    else:
        model = load_model(dataset, args)
    model.to(device)

    val_loader = get_chosen_dataset(dataset+'Val', model, args, is_train=False)
    test_loader = get_chosen_dataset(dataset, model, args, is_train=False)

    if testing:
        # Evaluate on training set
        train_loader = get_chosen_dataset(dataset+'Val', model, args, is_train=True)
        train_accuracy = evaluate_accuracy(model, train_loader, device)
        # Evaluate on log FIM
        samples_nr = 2000 # How many per-example gradients to accumulate
        logdet_hF = train_diag_fim_logtr(args, model, dataset, samples_nr)

    val_accuracy = evaluate_accuracy(model, val_loader, device)
    test_accuracy = evaluate_accuracy(model, test_loader, device)

    # Save results to JSON file
    results = {
        'validation_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }

    if testing:
        results = {
            'test_accuracy': test_accuracy,
            'validation_accuracy': val_accuracy,
            'train_accuracy': train_accuracy,
            'logdet_hF': logdet_hF
        }
    
    else:
        with open(save_path+dataset+"_results.json", 'w') as f:
            json.dump(results, f)

        print(f"\nDataset: {dataset}")
        if testing:
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Logarithm of the diagonal Fisher Information Matrix trace: {logdet_hF}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}\n")