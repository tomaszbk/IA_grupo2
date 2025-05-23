import uuid

import gradio as gr
import mlflow
import torch
import torch.nn.functional as F
from cnn_model import BottleCNN
from mlp_model import BottleMLP
from PIL import Image
from pipelines import preprocessing_pipeline

# Configuración
CLASS_NAMES = ["Rota", "En buen estado"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Bottle_Predictions")


# Función de predicción
def predict(img: Image.Image, model_class_name: str):
    # Convert string to actual class
    model_class = BottleCNN if model_class_name == "BottleCNN" else BottleMLP

    model_path = f"models/{model_class.__name__}.pth"
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # start a new MLflow run for this prediction
    with mlflow.start_run(run_name=f"pred-{uuid.uuid4()}"):
        # log the raw input image under artifacts
        mlflow.log_image(img, "input.png")
        # preprocess and infer
        input_tensor = preprocessing_pipeline(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        label = CLASS_NAMES[predicted_class]
        # log prediction parameters and metrics
        mlflow.log_param("predicted_class", label)
        mlflow.set_tag("predicted_class", label)
        mlflow.log_param("model", model.__class__.__name__)
        mlflow.log_metric("confidence", confidence)
        return f"{label} ({confidence * 100:.2f}% de confianza)"


# Interfaz Gradio
gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(
            choices=["BottleCNN", "BottleMLP"], value="BottleCNN", label="Modelo"
        ),
    ],
    outputs="text",
    title="Clasificador de Botellas",
).launch()
