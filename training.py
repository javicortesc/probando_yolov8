from ultralytics import YOLO

# Cargar el modelo base YOLOv8n
model = YOLO("yolov8n.pt")  # Puedes usar "yolov8s.pt" o "yolov8m.pt" para modelos más grandes

# Entrenar el modelo
results = model.train(
    data="data.yaml",  # Archivo YAML con la configuración del dataset
    epochs=100,        # Número de épocas
    imgsz=640,         # Tamaño de la imagen
    batch=16,          # Tamaño del batch
    name="epp_yolov8"  # Nombre del experimento
)