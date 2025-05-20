Descargar uv:
Bash:
`curl -LsSf https://astral.sh/uv/install.sh | sh`
windows:
`wget -qO- https://astral.sh/uv/install.sh | sh`

Setup:
`uv sync --extra cpu`
or
`uv sync --extra cu128`


# Proyecto clasificador_cnn:

Este proyecto implementa una red neuronal convolucional (CNN) en PyTorch para clasificar imágenes de botellas de agua como "en buen estado" o "rotas".

## Instalación de dependencias (ejecutar en terminal):
pip install torch torchvision opencv-python matplotlib

## Estructura del proyecto
```
clasificador_cnn/
├── data/
│ └── train/
│   ├── broken/ -> Fotos de botellas en rotas
│   └── good/ -> Fotos de botellas en buen estado
├── model.pth -> Modelo entrenado (se genera luego de entrenar)
├── cnn_model.py -> Definición de la CNN
├── train.py -> Script de entrenamiento
└── predict.py -> Script para predecir una imagen
```

## Entrenar modelo
Para entrenar el modelo, ejecutá:
```
python train.py
```

Esto va a entrenar el modelo con las imágenes de data/train y guardar el modelo en model.pth

## Predicción de una imagen

Para clasificar una imagen nueva, usá:
```
python predict.py ruta/a/la/imagen.jpg
```
El script mostrará algo como:

>Resultado: Rota (98.23% de confianza)