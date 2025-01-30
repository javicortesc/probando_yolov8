from ultralytics import YOLO

model = YOLO("runs/detect/epp_yolov8/weights/best.pt")  # Cargar el modelo entrenado
metrics = model.val()  # Validar
print(metrics.box.map)  # Precisión (mAP)