from collections import Counter
from datetime import datetime

import torch
from callbacks import LogMisclassifiedImages
from cnn_model import BottleCNN
from pipelines import augment_and_preprocess_pipeline, preprocessing_pipeline
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

# MLflow logger
mlflow_logger = MLFlowLogger(
    experiment_name="bottle_cnn",
    run_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
    tracking_uri="file:./mlruns",  # local mlruns folder; change URI if you have a remote server
)

# Configuración general
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 128
DATA_DIR = "./data/"
MODEL_PATH = "model.pth"
TRAIN_SPLIT_PERCENT = 0.8
SEED = 42

# Log all constants to MLflow
constants = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "img_size": IMG_SIZE,
    "train_split_percent": TRAIN_SPLIT_PERCENT,
    "seed": SEED,
}
for name, value in constants.items():
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, name, value)

# Dataset
dataset = datasets.ImageFolder(root=DATA_DIR, transform=preprocessing_pipeline)

# Log percentages of each class
counts = Counter(dataset.targets)
total = len(dataset)
for class_name, class_idx in dataset.class_to_idx.items():
    pct = counts[class_idx] / total * 100
    mlflow_logger.experiment.log_param(
        mlflow_logger.run_id, f"percent_{class_name}", f"{pct:.2f}"
    )

dataset_augmented = datasets.ImageFolder(
    root=DATA_DIR, transform=augment_and_preprocess_pipeline
)

combined_dataset = torch.utils.data.ConcatDataset([dataset, dataset_augmented])


n_train = int(0.8 * len(combined_dataset))
n_test = len(combined_dataset) - n_train

gen = torch.Generator().manual_seed(SEED)
train_ds, test_ds = random_split(combined_dataset, [n_train, n_test], generator=gen)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


mlflow_logger.experiment.log_param(mlflow_logger.run_id, "train_size", len(train_ds))
mlflow_logger.experiment.log_param(mlflow_logger.run_id, "test_size", len(test_ds))

# Modelo
callback = LogMisclassifiedImages(
    mlflow_logger, max_images=20, class_names=dataset.classes
)
model = BottleCNN()
trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    logger=mlflow_logger,
    callbacks=[callback],
)

# 4) Train on train_ds only
trainer.fit(model, train_loader)

# 5) Evaluate on test_ds
trainer.test(model, test_loader)

# Guardar el modelo como .pth después de entrenar
torch.save(model.state_dict(), "models/model.pth")
print("✅ Modelo guardado como model.pth")
