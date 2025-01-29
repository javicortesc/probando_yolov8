from ultralytics import YOLO
import numpy
# load pretrained yolo model
model = YOLO("yolov8n.pt")

# Prediccion en imagen
#results = model("faena.jpeg")

#results = model("bus.jpg")

results = model("faena2.jpg")

# Mostrar resultados
results[0].show()

# Imprimir resultados
print("Objectos detectados: ", results[0].names)