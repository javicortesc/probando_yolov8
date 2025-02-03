from ultralytics import YOLO
import cv2

# Cargar el modelo preentrenado YOLOv8n (no requiere entrenamiento previo)
model = YOLO("runs/detect/epp_yolov8/weights/best.pt")  # ¡Modelo válido y disponible!

# Iniciar la cámara web
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Detección de objetos
    results = model.predict(frame, conf=0.5)
    annotated_frame = results[0].plot()
    
    cv2.imshow("Detección de Objetos", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()