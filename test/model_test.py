import os
import unittest

import pytorch_lightning as pl

from models.cnn_model import BottleCNN
from models.mlp_model import BottleMLP
from models.train import train_model


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set seed for reproducibility
        pl.seed_everything(42)

    def test_train_cnn_model(self):
        """Test training CNN model with train/test split."""
        print("Testing CNN model training...")

        try:
            train_model(BottleCNN, use_all_data=False)
            self.assertTrue(True, "CNN model training completed successfully")
        except Exception as e:
            self.fail(f"CNN model training failed: {str(e)}")

    def test_train_mlp_model(self):
        """Test training MLP model with train/test split."""
        print("Testing MLP model training...")

        try:
            train_model(BottleMLP, use_all_data=False)
            self.assertTrue(True, "MLP model training completed successfully")
        except Exception as e:
            self.fail(f"MLP model training failed: {str(e)}")

    def test_train_cnn_all_data(self):
        """Test training CNN model with all data."""
        print("Testing CNN model training with all data...")

        try:
            train_model(BottleCNN, use_all_data=True)

            # Check if model file was created
            model_path = "models/BottleCNN.pth"
            self.assertTrue(
                os.path.exists(model_path), "CNN model file should be created"
            )

        except Exception as e:
            self.fail(f"CNN model training with all data failed: {str(e)}")

    def test_train_mlp_all_data(self):
        """Test training MLP model with all data."""
        print("Testing MLP model training with all data...")

        try:
            train_model(BottleMLP, use_all_data=True)

            # Check if model file was created
            model_path = "models/BottleMLP.pth"
            self.assertTrue(
                os.path.exists(model_path), "MLP model file should be created"
            )

        except Exception as e:
            self.fail(f"MLP model training with all data failed: {str(e)}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
