# 1) finetune on specified datasets
# specifying arguments like lr or merged differently from their default will allow the evaluation of the specified checkpoints
# warning: --save now only determines the location to save the heads
python finetune.py \
--data-location=./data/ \
--save=./heads/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--train-dataset=MNIST,GTSRB

# 2) eval single task
# specifying arguments like lr or merged differently from their default will allow the evaluation of the specified checkpoints
# warning: --save now only determines the location to save the heads
python eval_single_task.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--alpha=1.0 \
--merged=False \
--balanced=False \
--eval-datasets=DTD,EuroSAT


# 3) task addition
# calculates the best alpha and gives SOME results
python eval_task_addition.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/\
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--balanced=False \


# 4) used to calculate more precise averages
# If merged=True, it will also calculate its normalized performance on the single tasks
python calculate_avg.py \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0 \
--merged=False \
--balanced=False \
--alpha=1.0


# 5) finetunes using log-trace diagonal of FIM as early stopping criteria
python finetune_best_logtr.py \
--data-location=./data/ \
--save=./heads/ \
--balanced=False


# 6) finetunes using validation accuracy as early stopping criteria
python finetune_best_val.py \
--data-location=./data/ \
--save=./heads/ \
--balanced=False