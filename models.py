import numpy as np
import tensorflow as tf

tanh=tf.nn.tanh
# kc added sigmoid
sigmoid = tf.nn.sigmoid

class Generator(tf.keras.Model):
  def __init__(self, input_shape, output_shape=2):
    super(Generator, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=32, input_shape=(input_shape,), activation=sigmoid, dtype=tf.float64)
    self.dense2 = tf.keras.layers.Dense(units=16, activation=sigmoid, dtype=tf.float64)
    self.dense3 = tf.keras.layers.Dense(units=8, activation=sigmoid, dtype=tf.float64)
    # kc change to sigmoid
    self.dense4 = tf.keras.layers.Dense(units=output_shape, activation=sigmoid, dtype=tf.float64)

  def call(self, inputs):
    """Run the model."""
    result = self.dense1(inputs)
    result = self.dense2(result)
    result = self.dense3(result)
    result = self.dense4(result)
    return result

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=16, activation=sigmoid, dtype=tf.float64)
#     self.dense2 = tf.keras.layers.Dense(units=8, activation=tanh, dtype=tf.float64)
    self.dense3 = tf.keras.layers.Dense(units=1, activation=sigmoid, dtype=tf.float64)

  def call(self, inputs):
    """Run the model."""
    result = self.dense1(inputs)
#     result = self.dense2(result)
    result = self.dense3(result)
    return result