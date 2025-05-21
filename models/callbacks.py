import os

import torch
import torchvision.utils as vutils
from pytorch_lightning import Callback


class LogMisclassifiedImages(Callback):
    """
    During test, collects up to `max_images` misclassified samples,
    writes them out as individual PNGs, then logs the whole folder
    as an MLflow artifact under `misclassified/`.
    """

    def __init__(self, mlflow_logger, max_images=20):
        super().__init__()
        self.mlflow_logger = mlflow_logger
        self.max_images = max_images
        self.cache = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        x, y = batch
        logits = pl_module(x)
        preds = torch.argmax(logits, dim=1)
        mis_mask = preds != y
        if mis_mask.any():
            for img, p, label in zip(x[mis_mask], preds[mis_mask], y[mis_mask]):
                if len(self.cache) < self.max_images:
                    self.cache.append((img.cpu(), int(p.cpu()), int(label.cpu())))
                else:
                    break

    def on_test_end(self, trainer, pl_module):
        if not self.cache:
            return

        out_dir = "misclassified"
        os.makedirs(out_dir, exist_ok=True)
        for i, (img, p, label) in enumerate(self.cache):
            path = os.path.join(out_dir, f"img{i}_pred{p}_label{label}.png")
            vutils.save_image(img, path, normalize=True)

        # log entire folder
        run_id = self.mlflow_logger.run_id
        self.mlflow_logger.experiment.log_artifacts(
            run_id, out_dir, artifact_path="misclassified"
        )
