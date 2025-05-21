import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os
from cnn_model import BottleCNN
from pipelines import preprocessing_pipeline

# Configuración
MODEL_PATH = "model.ckpt"
IMAGE_PATH = "test_images/botella1.jpg"  # Cambia este path si querés usar otro
IMAGE_SIZE = (128, 128)  # Debe coincidir con el tamaño usado en entrenamiento
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Rota", "En buen estado"]  # Asegurate que coincida con las carpetas de ImageFolder



# Cargar modelo
model = BottleCNN.load_from_checkpoint(MODEL_PATH)
model.eval()

# Leer imagen
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = IMAGE_PATH

if not os.path.exists(img_path):
    print(f"❌ Imagen no encontrada: {img_path}")
    sys.exit(1)

image = Image.open(img_path).convert('RGB')
input_tensor = preprocessing_pipeline(image).unsqueeze(0).to(DEVICE)

# Inferencia
with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

# Resultado
print(f"✅ Resultado: {CLASS_NAMES[predicted_class]} ({confidence * 100:.2f}% de confianza)")
