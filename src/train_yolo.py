"""
============================================================
train_yolo.py
Entrena un modelo YOLOv8 (OBB o normal) desde un archivo ZIP
que contiene el dataset en formato YOLO (carpetas train/valid).
============================================================

Ejemplo de uso (en Colab o terminal):
-------------------------------------
!python train_yolo.py --zip_path "/content/CasasColombia.v1i.yolov8-obb.zip" --epochs 50 --imgsz 640
"""

import argparse
import zipfile
import os
from ultralytics import YOLO


def unzip_dataset(zip_path, extract_dir):
    """Descomprime el archivo ZIP del dataset en la carpeta indicada."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"No se encontró el archivo: {zip_path}")

    print(f"🔄 Descomprimiendo dataset desde: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"✅ Dataset descomprimido en: {extract_dir}")
    print("📂 Contenido:", os.listdir(extract_dir))

    # Buscar el archivo data.yaml dentro del dataset
    data_yaml = None
    for root, _, files in os.walk(extract_dir):
        if "data.yaml" in files:
            data_yaml = os.path.join(root, "data.yaml")
            break

    if not data_yaml:
        raise FileNotFoundError("No se encontró el archivo data.yaml en el dataset descomprimido.")

    print(f"📄 Archivo YAML encontrado en: {data_yaml}")
    return data_yaml


def train_yolo(zip_path, epochs=50, imgsz=640, batch=8, model_type="yolov8n-obb.pt"):
    """Entrena un modelo YOLOv8 utilizando un archivo ZIP con el dataset."""
    extract_dir = "/content/dataset_extract" if "/content" in zip_path else "./dataset_extract"
    os.makedirs(extract_dir, exist_ok=True)

    # 1️⃣ Descomprimir dataset
    data_yaml = unzip_dataset(zip_path, extract_dir)

    # 2️⃣ Cargar modelo preentrenado
    print(f"🚀 Cargando modelo base: {model_type}")
    model = YOLO(model_type)

    # 3️⃣ Entrenar modelo
    print("🏋️ Iniciando entrenamiento ...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="casas_colombia_yolo_obb",
    )

    print("✅ Entrenamiento completado.")
    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo YOLOv8 con dataset ZIP")
    parser.add_argument("--zip_path", type=str, required=True, help="Ruta al archivo .zip del dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Tamaño del batch (default: 8)")
    parser.add_argument("--model", type=str, default="yolov8n-obb.pt", help="Modelo base YOLOv8 (default: yolov8n-obb.pt)")

    args = parser.parse_args()

    # Entrenar modelo con los parámetros recibidos
    train_yolo(
        zip_path=args.zip_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_type=args.model
    )
