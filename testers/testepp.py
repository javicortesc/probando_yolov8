from multiprocessing import Process, Queue
import time
import cv2
from roboflow import Roboflow

# Inicializar Roboflow
rf = Roboflow(api_key="vgPxy9MFYv5TBEmnAMYh")  # Reemplaza con tu API key de Roboflow
project = rf.workspace().project("ppe-detection-yfmym")
model = project.version(1).model

# Función para capturar frames
def capture_frames(queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ancho de 640 píxeles
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto de 480 píxeles
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        if queue.qsize() < 2:  # Limitar el tamaño de la cola
            queue.put(frame)
    cap.release()

# Función para procesar frames
def process_frames(queue):
    frame_skip = 2  # Procesar 1 de cada 2 frames
    frame_count = 0
    
    while True:
        if not queue.empty():
            frame = queue.get()
            frame_count += 1
            
            # Saltar frames si es necesario
            if frame_count % frame_skip != 0:
                continue
            
            # Realizar la detección con el modelo de Roboflow
            predictions = model.predict(frame, confidence=40, overlap=30).json()
            
            # Dibujar las cajas y etiquetas en el frame
            for prediction in predictions['predictions']:
                x = int(prediction['x'] - prediction['width'] / 2)
                y = int(prediction['y'] - prediction['height'] / 2)
                width = int(prediction['width'])
                height = int(prediction['height'])
                label = prediction['class']
                confidence = prediction['confidence']
                
                # Dibujar la caja
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Mostrar el frame con las detecciones
            cv2.imshow("Detección de EPP", frame)
            
            # Salir al presionar 'q'
            if cv2.waitKey(1) == ord("q"):
                break
    cv2.destroyAllWindows()

# Punto de entrada del programa
if __name__ == "__main__":
    queue = Queue()
    
    # Crear procesos para captura y procesamiento
    capture_process = Process(target=capture_frames, args=(queue,))
    process_process = Process(target=process_frames, args=(queue,))
    
    # Iniciar procesos
    capture_process.start()
    process_process.start()
    
    # Esperar a que los procesos terminen
    capture_process.join()
    process_process.join()