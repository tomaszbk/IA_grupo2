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

# Configuración general
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 128
DATA_DIR = "./data/"
MODEL_PATH = "model.pth"
TRAIN_SPLIT_PERCENT = 0.8
SEED = 42

def train_model(model_class, use_all_data):

    # Definir nombre del experimento automáticamente
    if use_all_data:
        experiment_name = f"{model_class.__name__}_all_data"
    else:
        experiment_name = f"{model_class.__name__}_train_test_split"

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        tracking_uri="file:./mlruns",
    )

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

    if use_all_data:
        train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = None
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "train_size", len(combined_dataset))
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "test_size", 0)
    else:
        n_train = int(TRAIN_SPLIT_PERCENT * len(combined_dataset))
        n_test = len(combined_dataset) - n_train
        gen = torch.Generator().manual_seed(SEED)
        train_ds, test_ds = random_split(combined_dataset, [n_train, n_test], generator=gen)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "train_size", len(train_ds))
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, "test_size", len(test_ds))

    callback = LogMisclassifiedImages(
        mlflow_logger, max_images=20, class_names=dataset.classes
    )
    model = model_class()
    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        logger=mlflow_logger,
        callbacks=[callback],
    )

    trainer.fit(model, train_loader)
    if test_loader is not None:
        trainer.test(model, test_loader)

    if use_all_data:
         torch.save(model.state_dict(), MODEL_PATH)
         print(f"Modelo guardado en {MODEL_PATH}")

# Ejemplo de uso:
if __name__ == "__main__":
    train_model(BottleCNN, use_all_data=False)  