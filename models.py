import numpy as np
import tensorflow as tf

class Generator(tf.keras.Model):
  def __init__(self, input_shape, output_shape=2):
    super(Generator, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=128, input_shape=(input_shape,), activation=tf.nn.leaky_relu, dtype=tf.float64)
    self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu, dtype=tf.float64)
    self.dense3 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu, dtype=tf.float64)
    self.dense4 = tf.keras.layers.Dense(units=output_shape, activation=None, dtype=tf.float64)

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
    self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu, dtype=tf.float64)
    self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu, dtype=tf.float64)
    self.dense3 = tf.keras.layers.Dense(units=1, activation=None, dtype=tf.float64)

  def call(self, inputs):
    """Run the model."""
    result = self.dense1(inputs)
    result = self.dense2(result)
    result = self.dense3(result)
    return result