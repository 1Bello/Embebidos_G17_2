
import os
import zipfile
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])



optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# Vamos a usar las siguientes tecnicas para aumentar el dataset:
train_datagen = ImageDataGenerator( # no es que genere imagenes sino que le aplica una transformacion cada vez que la imagen es llamada
      rescale=1./255,
      rotation_range=20,
      zoom_range=0.2,
      horizontal_flip=True,
      )

train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(96, 96),
        batch_size = 64, # El batch nos dice cada cuanto tiempo se van a actualizar los pesos, si no se define es 1 y se actualizan despues de cada imagen. Mas rapido y tomamos promedios. Para considerarse una epoca tenemos que pasar por todos los btchs
        class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        'test',
        target_size=(96, 96),
        class_mode='categorical')


NUM_EPOCHS = 50
history = model.fit(
      train_generator,
      steps_per_epoch= len(train_generator) // train_generator.batch_size, # cuántos batches del conjunto de entrenamiento se utilizan para entrenar el modelo durante una epoch. ceil(num_samples / batch_size)
      epochs=NUM_EPOCHS,
      verbose=1,
      validation_data=validation_generator)

model.save("trained_model.keras")
# Buena practica y buen grafico para usar en su presentacion
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xlim([0,NUM_EPOCHS])
plt.ylim([0.4,1.0])
plt.show()
