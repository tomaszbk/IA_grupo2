import os

import torch
import torchvision.utils as vutils
from pytorch_lightning import Callback


class LogMisclassifiedImages(Callback):
    def __init__(self, mlflow_logger, max_images=20, class_names=None):
        super().__init__()
        self.mlflow_logger = mlflow_logger
        self.max_images = max_images
        self.class_names = class_names or []
        self.cache = []
        self.out_dir = None

    def on_test_start(self, trainer, pl_module):
        run_id = self.mlflow_logger.run_id
        # each run writes into its own subfolder
        self.out_dir = os.path.join("misclassified", run_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self.cache = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        x, y = batch
        preds = torch.argmax(pl_module(x), dim=1)
        mis_mask = preds != y
        for img, p, label in zip(x[mis_mask], preds[mis_mask], y[mis_mask]):
            if len(self.cache) < self.max_images:
                self.cache.append((img.cpu(), int(p.cpu()), int(label.cpu())))
            else:
                break

    def on_test_end(self, trainer, pl_module):
        if not self.cache:
            return

        for i, (img, p, label) in enumerate(self.cache):
            pred_name = self.class_names[p]
            true_name = self.class_names[label]
            fname = f"img{i}_pred_{pred_name}_lbl_{true_name}.png"
            vutils.save_image(img, os.path.join(self.out_dir, fname), normalize=True)

        # logs only this run's folder
        self.mlflow_logger.experiment.log_artifacts(
            self.mlflow_logger.run_id, self.out_dir, artifact_path="misclassified"
        )
