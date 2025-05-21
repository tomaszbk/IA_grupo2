import torchvision.transforms.v2 as transforms
from torch import float32

# 1. Define the transformations
# The transformations will be applied sequentially.
# - Resize to 128x128
# - Convert to Grayscale
# - Convert to PyTorch Tensor (scales pixel values to [0, 1])
# - (Optional) Normalize: For grayscale, a common practice is to normalize
#   with mean 0.5 and std 0.5 to bring values to [-1, 1].
#   Or, calculate mean and std from your specific dataset.
#   For this example, we'll use a simple 0.5, 0.5 normalization.
preprocessing_pipeline = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalizes to [-1, 1]
    ]
)


augmentation_pipeline = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25, fill=255),
    ]
)

augment_and_preprocess_pipeline = transforms.Compose(
    [
        augmentation_pipeline,
        preprocessing_pipeline,
    ]
)
