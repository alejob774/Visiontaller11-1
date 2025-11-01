"""
============================================================
app.py
API REST con FastAPI que usa inferencia.py
para detectar casas en imágenes.
============================================================
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
from inferencia import cargar_modelo, predecir_casas

# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

RUTA_PESOS = os.environ.get("YOLO_WEIGHTS", r"C:\Users\prestamour.UROSARIO\Documents\AAVC\src\models\yolov8n-obb.pt")

try:
    modelo = cargar_modelo(RUTA_PESOS)
except FileNotFoundError:
    raise RuntimeError(f"No se encontró el modelo en: {RUTA_PESOS}")

# ============================================================
# CONFIGURACIÓN DE LA APLICACIÓN FASTAPI
# ============================================================
app = FastAPI(
    title="API de Detección de Casas - YOLOv8",
    description="Recibe una imagen y devuelve bounding boxes y scores en formato JSON.",
    version="1.0"
)

# ============================================================
# ENDPOINT /predict
# ============================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve detecciones en JSON.
    [
        {"class": "house", "score": 0.92, "bbox": [x1,y1,x2,y2]},
        ...
    ]
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El archivo debe ser una imagen (.jpg, .png, etc.)")

    try:
        image_bytes = await file.read()
        image_path = "temp_image.jpg"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Ejecutar inferencia
        detecciones = predecir_casas(modelo, image_path)

        # Eliminar archivo temporal
        os.remove(image_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {str(e)}")

    return JSONResponse(content=detecciones)
