from __future__ import absolute_import, division, print_function, unicode_literals

import self as self
import tensorflow as tf
from pathlib import Path
import numpy as np

class Word2Vec:

    def __init__(self, vocab_size=0, embedding_dim=15, optimizer='adam', epochs=10000):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.optimized = optimizer

    # TRAIN METHOD
    def entrenamiento(self, x_train, y_train):
        self.W1 = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_dim]))
        self.b1 = tf.Variable(tf.random.normal([self.embedding_dim]))  # bias
        self.W2 = tf.Variable(tf.random.normal([self.embedding_dim, self.vocab_size]))
        self.b2 = tf.Variable(tf.random.normal([self.vocab_size]))

        for _ in range(self.epochs):
            with tf.GradientTape() as t:
                hidden_layer = tf.add(tf.matmul(x_train, self.W1), self.b1)
                output_layer = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, self.W2), self.b2))
                perdidad_entropia_cruzada = tf.reduce_mean(
                    -tf.math.reduce.sum(y_train * tf.math.log(output_layer), axis=[1]))

            grads = t.gradient(perdidad_entropia_cruzada, [self.W1, self.b1, self.W2, self.b2])
            self.optimizer.apply_gradients(zip(grads, [self.W1, self.b1, self.W2, self.b2]))

            if (_ % 1000 == 0):
                print(perdidad_entropia_cruzada)

    def vectorizacion(self, word_index):
        return (self.W1 + self.b1)[word_index]