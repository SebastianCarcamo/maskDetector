import os
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

mainPath = "./files/"
# Hiperparametros
INIT_LR = 1e-4
EPOCHS = 20
BATCH = 32

data = []
labels = []

imagePaths = list(paths.list_images(mainPath + "dataset"))

# Para cada imagen
for imagePath in imagePaths:
    # extraemos los labels de los nombres de los folders
    label = imagePath.split(os.path.sep)[-2]

    # cargado de la imagen
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)
    # vamos llenando data y labels con la informacion correspondiente

data = np.array(data, dtype="float32")
labels = np.array(labels)

# onehot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels) # separamos la data 80/20 para train/test

# generador de data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# entrenamiento del modelo
model.fit(aug.flow(trainX, trainY, batch_size=BATCH),
  steps_per_epoch=len(trainX) // BATCH,
  validation_data=(testX, testY),
  validation_steps=len(testX) // BATCH,
  epochs=EPOCHS)

# guardar el modelo para no tener que hacer el proceso mas de 1 vez
model.save(mainPath + "model", save_format="h5")