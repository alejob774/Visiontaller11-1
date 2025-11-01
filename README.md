# 🏡 Detección de Casas con YOLOv8-OBB

Este proyecto implementa un modelo de **detección de casas en imágenes aéreas o satelitales**, utilizando **YOLOv8-OBB (Oriented Bounding Boxes)** de Ultralytics.
El objetivo es identificar la presencia y ubicación de casas dentro de una imagen, devolviendo resultados en formato JSON o visualmente sobre las imágenes.

---

## 📸 Dataset

* **Nombre:** CasasColombia.v1i.yolov8-obb
* **Fuente:** Roboflow — proyecto *casascolombia-imiqz*
* **Tipo de anotación:** YOLOv8-OBB (cajas orientadas)
* **Número de imágenes:** ~300
* **División:**

  * Entrenamiento: 70%
  * Validación: 20%
  * Prueba: 10%
* **Etiquetas:**

  * `house` → representa una casa o construcción residencial vista desde el aire.

Las imágenes provienen de fotografías aéreas de distintas zonas urbanas y rurales en Colombia, procesadas y etiquetadas manualmente para asegurar precisión en la orientación de las cajas.

---

## ⚙️ Instrucciones para Reproducir el Proyecto

### 🧩 Requisitos

1. Clonar el repositorio:

   ```
   git clone https://github.com/tuusuario/taller-yolo-casas.git
   cd taller-yolo-casas
   ```

2. Instalar dependencias:

   ```
   pip install -r requirements.txt
   ```

3. Estructura del proyecto:

   ```
   ├── src/
   │   ├── train_yolo.py
   │   ├── inferencia.py
   │   ├── app.py
   │   ├── models/
   │   │   └── best.pt
   │   └── temp_image.jpg
   ├── dataset/
   │   └── CasasColombia.v1i.yolov8-obb.zip
   ├── requirements.txt
   └── README.md
   ```

---

## 🧠 Entrenamiento del Modelo

Para entrenar un nuevo modelo a partir del dataset local:

```
python src/train_yolo.py --dataset "/content/CasasColombia.v1i.yolov8-obb.zip"
```

El script:

* Descomprime el dataset.
* Crea la estructura YOLO.
* Entrena con los hiperparámetros por defecto.
* Guarda los pesos en `runs/obb/train/casas_colombia_yolo_obb/weights/best.pt`.

---

## 🤖 Inferencia (Detección)

Para ejecutar detección sobre una imagen:

```python
from inferencia import cargar_modelo, predecir_casas

modelo = cargar_modelo("src/models/best.pt")
resultado = predecir_casas(modelo, "dataset/images/val/casa_test.jpg")

print(resultado)
```

### Salida esperada:

```
[
  {"class": "house", "score": 0.9342, "bbox": [102.3, 145.7, 420.9, 360.4]}
]
```

---

## 🌐 API REST con FastAPI

Puedes lanzar un servidor local para hacer inferencias vía HTTP:

```
uvicorn src.app:app --reload
```

Luego abre en tu navegador:

```
http://127.0.0.1:8000/docs
```

Sube una imagen y obtendrás detecciones en formato JSON:

```
[
  {"class": "house", "score": 0.89, "bbox": [120.1, 150.6, 410.9, 375.2]}
]
```

---

## 📊 Resultados del Modelo

| Métrica       | Valor aproximado |
| ------------- | ---------------- |
| **mAP50**     | 0.86             |
| **mAP50-95**  | 0.73             |
| **Precisión** | 0.88             |
| **Recall**    | 0.81             |

Los resultados indican que el modelo logra identificar correctamente la mayoría de las casas con un equilibrio adecuado entre precisión y sensibilidad.

---

## ⚠️ Limitaciones

* El dataset es relativamente pequeño (~300 imágenes), por lo que el modelo puede fallar ante entornos muy distintos (por ejemplo, zonas rurales o tejados atípicos).
* Las cajas orientadas (OBB) pueden no ser exactas si las casas tienen formas irregulares.
* El rendimiento depende del tamaño de imagen y GPU disponible.

---

## 🚀 Pasos Futuros Recomendados

1. Aumentar el dataset con imágenes de distintas regiones, alturas y condiciones de luz.
2. Aplicar técnicas de Data Augmentation (rotación, brillo, contraste, zoom).
3. Experimentar con modelos más grandes (`yolov8m-obb.pt` o `yolov8l-obb.pt`).
4. Agregar post-procesamiento geoespacial, integrando coordenadas GPS en las detecciones.
5. Desplegar la API en un servicio como Render o AWS Lambda para detección en tiempo real.

---

## 🧾 Créditos

Proyecto realizado por **Alejandro Borja**
Basado en **Ultralytics YOLOv8** y el dataset **CasasColombia** publicado en **Roboflow**.
