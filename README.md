# Instalacion y ejecucion
Descargar uv:

Bash:
`curl -LsSf https://astral.sh/uv/install.sh | sh`
windows:
`wget -qO- https://astral.sh/uv/install.sh | sh`

o

`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Setup:

`uv sync --extra cpu`
or
`uv sync --extra cu128`

Ejecutar la UI de mlflow:

`mlflow ui --backend-store-uri ./mlruns`

Entrenar el modelo CNN:

`uv run --extra cpu models/train.py`

o con CUDA usar

`uv run --extra cu128 models/train.py`

Agregar una dependencia:

`uv add <nombre_dependencia>`

# Proyecto clasificador_cnn:

Este proyecto implementa una red neuronal convolucional (CNN) en PyTorch para clasificar imágenes de botellas de agua como "en buen estado" o "rotas".


## Estructura del proyecto
```
models/
├── model.pth -> Modelo entrenado (se genera luego de entrenar)
├── cnn_model.py -> Definición de la CNN
├── train.py -> Script de entrenamiento
└── predict.py -> Script para predecir una imagen
```

## Entrenar modelo

Esto va a entrenar el modelo con las imágenes de data/train y guardar el modelo en model.pth

## Predicción de una imagen

Para clasificar una imagen nueva, usá:
```
uv run models/predict.py ruta/a/la/imagen.jpg
```
El script mostrará algo como:

>Resultado: Rota (98.23% de confianza)