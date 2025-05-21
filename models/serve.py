import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from cnn_model import BottleCNN
from pipelines import preprocessing_pipeline

# Configuración
MODEL_PATH = "model.pth"  # <- usando el archivo .pth
CLASS_NAMES = ["Rota", "En buen estado"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = BottleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Función de predicción
def predict(img: Image.Image):
    input_tensor = preprocessing_pipeline(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        label = CLASS_NAMES[predicted_class]
        return f"{label} ({confidence * 100:.2f}% de confianza)"

# Interfaz Gradio
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Clasificador de Botellas"
).launch()