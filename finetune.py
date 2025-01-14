from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head

args = parse_arguments() # Result of CLI argument parsing

# Instantiate a full model architecture
encoder = ImageEncoder(args) # Pre-trained CLIP ViT backbone

# Get MNIST open-vocabulary classifier
head = get_classification_head(args, "MNISTVal")
model = ImageClassifier(encoder, head) # Build full model
model.freeze_head() # Freeze the classification head

# Obtain the Train split of the "MNIST" dataset
dataset = get_dataset(
  "MNISTVal", preprocess=model.train_preprocess,
  location=args.data_location, batch_size=128, num_workers=2)
loader = get_dataloader(dataset, is_train=True, args=args)

# Obtain the Validation split of the "MNIST" dataset
dataset = get_dataset(
  "MNISTVal", preprocess=model.val_preprocess,
  location=args.data_location, batch_size=128, num_workers=2)
loader = get_dataloader(dataset, is_train=False, args=args)

# Obtain the Test split of the "MNIST" dataset
dataset = get_dataset(
  "MNIST", preprocess=model.val_preprocess,
  location=args.data_location, batch_size=128, num_workers=2)
loader = get_dataloader(dataset, is_train=False, args=args)

# How to iterate on a split
for batch in loader:
  data = maybe_dictionarize(batch)
  x, y = data["images"], data["labels"]