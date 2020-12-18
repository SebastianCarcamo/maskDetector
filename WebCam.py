import os
import numpy as np
import imutils
import time
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

faceDetectorPath = "./files/face_detector/deploy.prototxt"
maskModelPath = "./files/model"
weights = "./files/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

def load_faceDetector(pathPtxt,pathWeights):
    return cv2.dnn.readNet(pathPtxt,pathWeights)

def obtain_detections(faceNet, blob):
    faceNet.setInput(blob)
    return faceNet.forward()

def computeFace(faceImg):
    face = cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB) # BGR a RGB
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face) # volver la imagen un array
    face = preprocess_input(face) # preprocesarla
    return face

def maskDetection(frame, faceNet, maskNet):
    # construir un blob usando las dimensiones del frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    # obtener detecciones de caras
    detections = obtain_detections(faceNet, blob)

    faces = []
    preds = []
    locs = []

    # recorrer todas las caras detectadas
    for i in range(detections.shape[2]):
        #extraer la confianza
        confidence = detections[0, 0, i, 2]

        # establecemos un minimo de confianza que debe pasar
        if confidence > 0.5:
            # obtenemos los bounding boxes
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # los bounding boxes deben estar en el frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX] # extraer la cara
            face = computeFace(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        #solo ingresa aca si se detecto al menos 1 cara
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        # se hacen las predicciones en batch

    return (locs, preds)

# cargar el detector de caras
faceNet = load_faceDetector(faceDetectorPath, weights)

# cargar el detector de caras con mascarilla
maskNet = load_model(maskModelPath)

# Prender la camara y dejarla lista
print("Camara prendida (deberia encender la luz)")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    
    # detecta si hay caras en el frame y hace las predicciones
    (locs, preds) = maskDetection(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, noMask) = pred

        # define el label y el color
        if mask > noMask:
            label = "Con mascarilla"
            color = (0, 255, 0)
        else:
            label = "Sin mascarilla"
            color = (0, 0, 255)

        # muestra el label y el bounding box en la cara detectada
        cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # muestra la camara
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # si se presiona q se apaga todo
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()