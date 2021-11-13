import tensorflow as tf
import numpy as np

#colocamos arreglos de entradas y salidas de entrenamiento
celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

#keras usa redes neuronales de manera simple
capa = tf.keras.layers.Dense(units=1, input_shape=[1])