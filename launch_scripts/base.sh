# 1) finetune on all datasets
# 2) eval single task
# 3) task addition

python finetune.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0

! python finetune.py \
--data-location=/content/AML-proj-24-25/data/ \
--save=/content/AML-proj-24-25/heads/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0

python eval_single_task.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \

! python eval_single_task.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/ \
--eval-datasets DTD

python eval_task_addition.py \
--data-location=/path/to/datasets/ \
--save=/path/to/save/