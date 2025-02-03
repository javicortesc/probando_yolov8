from django.shortcuts import render
from django.http import StreamingHttpResponse
from ultralytics import YOLO
import cv2
import threading
from .models import Incident # Importación de modelos

# Create your views here.
model = YOLO("runs/detect/epp_yolov8/weights/best.pt")  # Cargar mi modelo

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detección de objetos
            results = model.predict(frame, conf=0.5)
            annotated_frame = results[0].plot()

            # Lógica de alertas
            humano_detectado = False
            casco_detectado = False
            chaleco_detectado = False

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    if class_name == "human":
                        humano_detectado = True
                    elif class_name == "helmet":
                        casco_detectado = True
                    elif class_name == "vest":
                        chaleco_detectado = True
            
            # Guardar incidencia si no se detecta casco o chaleco
            if humano_detectado:
                if not casco_detectado and not chaleco_detectado:
                    Incident.objects.create(camera = "Cámara 1", incident_type = "Sin casco ni chaleco")
                
                elif not casco_detectado:
                    Incident.objects.create(camera = "Cámara 1", incident_type = "Sin casco")
                
                elif not chaleco_detectado:
                    Incident.objects.create(camera = "Cámara 1", incident_type = "Sin chaleco")
                
            # Convertir el frame a JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Vista para transmitir video en tiempo real
def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def home(request):
    return render(request, 'epp_detection/home.html')
