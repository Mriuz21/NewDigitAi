import tkinter as tk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gui import GUIApplication 

# # Data Augmentation
# data_augmentation = tf.keras.Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255),
#   layers.experimental.preprocessing.RandomRotation(0.1),
# ])

# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test,y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train,axis=1)
# x_test = tf.keras.utils.normalize(x_test,axis=1)

# # Creating the model
# model = tf.keras.models.Sequential()

# # Making the layers
# model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(10,activation='softmax'))

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# # Adding Early Stopping and Model Checkpoint
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# mc = ModelCheckpoint('best_model.h7', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# # Train and save the model
# model.fit(x_train, y_train, validation_split=0.2, epochs=7, callbacks=[es, mc])


# Load the best model
model = tf.keras.models.load_model('handwritten.model2')

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApplication(master=root, model=model)
    root.mainloop()
