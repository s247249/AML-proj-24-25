import torch
import json
import os

from args import parse_arguments

if __name__ == '__main__':
    
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
    
    json_dir = "./json_results"
    
    if not args.batch_size==32:
        json_dir += "/bs_" + str(args.batch_size)
    elif not args.lr==1e-4:
        json_dir += "/lr_" + str(args.lr)
    elif not args.wd==0.0:
        json_dir += "/wd_" + str(args.wd)
    else:
        json_dir += "/base"

    ft_json_dir = str(json_dir) + "/"
    ft_results_dict = {}
    for dataset in datasets:
        with open(ft_json_dir + dataset +"_results.json", 'r') as f:
            ft_results = json.load(f)
        ft_results_dict[dataset] = ft_results
    
    # Change directory for merged model
    if args.merged:
        json_dir += "/merged"
    # If evaluating alpha scaled model
    elif not args.alpha==1.0:
        json_dir += "/scaled"
    
    json_dir += "/"
    
    results_dict = {}
    for dataset in datasets:
        if args.merged:
            with open(json_dir + dataset +"_merged_results.json", 'r') as f:
                results = json.load(f)
        else:
            with open(json_dir + dataset +"_results.json", 'r') as f:
                results = json.load(f)
        results_dict[dataset] = results


    abs_task_test_accuracy = 0.0
    abs_task_train_accuracy = 0.0
    abs_task_log_accuracy = 0.0
    
    norm_test_accuracy = 0.0
    norm_train_accuracy = 0.0

    norm_test_accuracy_dict = {}
    norm_train_accuracy_dict = {}

    for dataset in datasets: 
        # Sum absolute individual accuracies
        abs_task_test_accuracy += results_dict[dataset].get('test_accuracy')
        abs_task_train_accuracy += results_dict[dataset].get('train_accuracy')
        abs_task_log_accuracy += results_dict[dataset].get('logdet_hF')
    
        # Get finetuned models accuracies 
        ft_test_accuracy = ft_results_dict[dataset].get('test_accuracy')
        ft_train_accuracy = ft_results_dict[dataset].get('train_accuracy')

        # Single task normalized accuracies
        norm_test_accuracy_dict[dataset] = 100 * results_dict[dataset].get('test_accuracy') / ft_test_accuracy
        norm_train_accuracy_dict[dataset] = 100 * results_dict[dataset].get('train_accuracy') / ft_train_accuracy
        
        # Sum normalized accuracies
        norm_test_accuracy += norm_test_accuracy_dict[dataset]
        norm_train_accuracy += norm_train_accuracy_dict[dataset]
        
    
    avg_abs_test_accuracy = abs_task_test_accuracy / len(datasets)
    avg_abs_train_accuracy = abs_task_train_accuracy / len(datasets)
    avg_abs_log_accuracy = abs_task_log_accuracy / len(datasets)

    avg_norm_test_accuracy = norm_test_accuracy / len(datasets)
    avg_norm_train_accuracy = norm_train_accuracy / len(datasets)

    if args.merged:
        results = {
            'avg_abs_test_accuracy': avg_abs_test_accuracy,
            'avg_norm_test_accuracy': avg_norm_test_accuracy,
            'avg_abs_train_accuracy': avg_abs_train_accuracy,
            'avg_norm_train_accuracy': avg_norm_train_accuracy,
            'avg_abs_log_accuracy': avg_abs_log_accuracy,
            'norm_test_accuracy_dict' : norm_test_accuracy_dict,
            'norm_train_accuracy_dict': norm_train_accuracy_dict
        }
    else:
        results = {
            'avg_abs_test_accuracy': avg_abs_test_accuracy,
            'avg_abs_train_accuracy': avg_abs_train_accuracy,
            'avg_abs_log_accuracy': avg_abs_log_accuracy
        }

    if args.merged:
        print("merged")
        print(f"dir: {json_dir}")
        with open(json_dir + "merged_avg.json", 'w') as f:
            json.dump(results, f, indent=4)
    # If evaluating alpha-scaled model
    elif not args.alpha==1.0:
        print("scaled")
        print(f"dir: {json_dir}")
        with open(json_dir + "scaled_avg.json", 'w') as f:
            json.dump(results, f, indent=4)

    else:
        print("finetuned")
        print(f"dir: {ft_json_dir}")
        with open(ft_json_dir + "ft_avg.json", 'w') as f:
            json.dump(results, f, indent=4)
    
    print(f"{results}")


    

    
    