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


# Global variable to store current run info for feedback
current_run_info = {"run_id": None, "img": None, "predicted_class": None}


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
    with mlflow.start_run(run_name=f"pred-{uuid.uuid4()}") as run:
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

        # Store run info for potential feedback
        global current_run_info
        current_run_info = {
            "run_id": run.info.run_id,
            "img": img,
            "predicted_class": label,
        }

        # Reset button colors when new prediction is made
        return (
            f"{label} ({confidence * 100:.2f}% de confianza)",
            gr.update(variant="secondary"),  # reset thumbs_up_btn to normal
            gr.update(variant="secondary"),  # reset thumbs_down_btn to normal
            gr.update(visible=False),  # hide detailed_feedback_section
        )


# Función para manejar feedback del usuario (thumbs up/down)
def submit_thumbs_feedback(is_correct: bool):
    global current_run_info

    if current_run_info["run_id"] is None:
        return (gr.update(), gr.update(), gr.update(visible=False))

    # Log the feedback to the existing run
    with mlflow.start_run(run_id=current_run_info["run_id"]):
        mlflow.set_tag("prediction_correct", "yes" if is_correct else "no")
        if is_correct:
            # If thumbs up, the predicted class is the correct class
            mlflow.set_tag("correct_class", current_run_info["predicted_class"])
        else:
            mlflow.set_tag(
                "correct_class",
                "En buen estado"
                if current_run_info["predicted_class"] == "Rota"
                else "Rota",
            )
        # Don't reset run info for thumbs down - keep it for detailed feedback

    if is_correct:
        # Only reset run info if thumbs up (feedback complete)
        current_run_info = {"run_id": None, "img": None, "predicted_class": None}
        # Return button updates - thumbs up green, thumbs down normal, hide detailed section
        return (
            gr.update(variant="primary"),  # thumbs_up_btn - green
            gr.update(variant="secondary"),  # thumbs_down_btn - normal
            gr.update(visible=False),  # detailed_feedback_section - hidden
        )
    else:
        # Return button updates - thumbs up normal, thumbs down red, show detailed section
        return (
            gr.update(variant="secondary"),  # thumbs_up_btn - normal
            gr.update(variant="stop"),  # thumbs_down_btn - red
        )


# Interfaz Gradio
with gr.Blocks(title="Clasificador de Botellas") as demo:
    gr.Markdown("# 🍼 Clasificador de Botellas")
    gr.Markdown("Sube una imagen de botella y selecciona el modelo para clasificarla")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Imagen de botella")
            model_dropdown = gr.Dropdown(
                choices=["BottleCNN", "BottleMLP"], value="BottleCNN", label="Modelo"
            )
            predict_btn = gr.Button("🔍 Predecir", variant="primary")

        with gr.Column():
            prediction_output = gr.Textbox(label="Predicción", interactive=False)

            # Quick thumbs feedback
            gr.Markdown("### ¿Es correcta la predicción?")
            with gr.Row():
                thumbs_up_btn = gr.Button("👍 Correcto", variant="secondary")
                thumbs_down_btn = gr.Button("👎 Incorrecto", variant="secondary")

    # Event handlers
    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[
            prediction_output,
            thumbs_up_btn,
            thumbs_down_btn,
        ],
    )

    thumbs_up_btn.click(
        fn=lambda: submit_thumbs_feedback(True),
        outputs=[thumbs_up_btn, thumbs_down_btn],
        show_progress=False,
    )

    thumbs_down_btn.click(
        fn=lambda: submit_thumbs_feedback(False),
        outputs=[thumbs_up_btn, thumbs_down_btn],
        show_progress=False,
    )

demo.launch()
