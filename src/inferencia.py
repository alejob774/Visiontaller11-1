"""
============================================================
inferencia.py
Ejecuta inferencia con un modelo YOLOv8 (normal u OBB)
y devuelve las detecciones en formato JSON.
============================================================

Ejemplo de uso:
---------------
from inferencia import cargar_modelo, predecir_casas

modelo = cargar_modelo("runs/obb/train/casas_colombia_yolo_obb/weights/best.pt")
resultado = predecir_casas(modelo, "dataset/images/val/ejemplo.jpg")
print(resultado)
"""

from ultralytics import YOLO
import os

def cargar_modelo(ruta_pesos: str):
    """
    Carga el modelo YOLOv8 desde la ruta indicada.
    """
    if not os.path.exists(ruta_pesos):
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {ruta_pesos}")

    print(f"🚀 Cargando modelo YOLO desde: {ruta_pesos}")
    modelo = YOLO(ruta_pesos)
    return modelo


def predecir_casas(modelo, ruta_imagen: str):
    """
    Realiza inferencia sobre una imagen local usando el modelo YOLO cargado.
    Devuelve una lista con diccionarios tipo:
    [
      {"class": "house", "score": 0.92, "bbox": [x1, y1, x2, y2]},
      ...
    ]
    """
    import numpy as np

    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    print(f"🖼️ Ejecutando detección sobre: {ruta_imagen}")
    results = modelo(ruta_imagen)

    # Validar resultados
    if results is None or len(results) == 0:
        print("⚠️ El modelo no devolvió resultados.")
        return []

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        print("⚠️ No se detectaron objetos en la imagen.")
        return []

    # Procesar las detecciones
    detecciones = []
    for box in res.boxes:
        coords = [float(x) for x in box.xyxy[0].cpu().numpy().tolist()]
        score = float(box.conf.cpu().numpy().item())
        cls_id = int(box.cls.cpu().numpy().item())
        cls_name = modelo.names.get(cls_id, str(cls_id))

        detecciones.append({
            "class": cls_name,
            "score": round(score, 4),
            "bbox": [round(x, 2) for x in coords]
        })

    return detecciones
