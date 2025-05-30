# Trabajo Práctico IA - Grupo 2 - 2025
## Introducción
Este proyecto desarrolla e integra dos modelos de aprendizaje automático: una red neuronal convolucional (CNN) y un perceptrón multicapa (MLP), diseñados para clasificar imágenes de botellas de plástico en dos categorías: "en buen estado" o "rotas". El objetivo es facilitar la detección automática de defectos en botellas a partir de imágenes.

Para el análisis del entrenamiento y evaluación de los modelos, se utiliza MLflow como herramienta de seguimiento y visualización de métricas, permitiendo monitorear en tiempo real variables como la pérdida, la precisión, y otros indicadores relevantes tanto por paso como por época.

Además, el proyecto cuenta con una interfaz web construida con Gradio, que permite a cualquier usuario cargar imágenes desde su computadora y obtener predicciones instantáneas, eligiendo si desea utilizar el modelo CNN o el MLP. Esta interfaz hace que el sistema sea accesible incluso para usuarios sin conocimientos técnicos, ofreciendo una experiencia práctica y visual para probar los modelos entrenados.

## Instalacion del software necesario
### Python
Primero es necesario instalar Python. Desde el siguiente link se puede descargar la última versión:<br /> 
https://www.python.org/downloads/

<br /> 

### MLflow
```bash
pip install mlflow
```

<br /> 

### Gradio
```bash
pip install gradio
```

<br /> 

### uv

Con bash
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
<br /> Con Windows
```
wget -qO- https://astral.sh/uv/install.sh | sh

# o también se puede hacer con:

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
<br /> Luego debemos ejecutar lo siguiente
```bash
uv sync --extra cpu

# o también se puede hacer con:

uv sync --extra cu128
```

## Ejecución

Primero se debe ir al directorio del proyecto
```bash
cd \...\IA_grupo2
```
<br /> Ejecutar la UI de MLflow
```bash
mlflow ui --backend-store-uri ./mlruns
```
<br /> Luego debemos entrenar a los dos algoritmos para obtener sus modelos
```bash
uv run -m models.train

# o con CUDA usar

uv run -m --extra cu128 models.train
```
<br /> Una vez que tengamos los dos modelos podemos ejecutar la aplicacion web con
```bash
uv run models/serve.py
```
Desde la aplicación web podemos cargar imagenes para determinar si estan rotas o en buen estado, usando los modelos de CNN y MLP.