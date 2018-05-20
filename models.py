import tensorflow as tf
import numpy as np

def generator(z, out_dim, n_units, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        layers_output = [z] + [np.nan for _ in range(len(n_units) - 1)]
        for i in range(1, len(n_units)):
            # Hidden Layers
            layers_output[i] = tf.layers.dense(layers_output[i - 1], n_units[i], activation=None)
            # Leaky ReLU
            layers_output[i] = tf.maximum(alpha * layers_output[i], layers_output[i])
        
        # Logits and sigmoid output
        logits = tf.layers.dense(layers_output[-1], out_dim, activation=None)
        out = tf.sigmoid(logits)
        
        return out, layers_output[1:]

def discriminator(x, n_units, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        layers_output = [x] + [np.nan for _ in range(len(n_units) - 1)]
        for i in range(1, len(n_units)):
            # Hidden Layers
            layers_output[i] = tf.layers.dense(layers_output[i - 1], n_units[i], activation=None)
            # Leaky ReLU
            layers_output[i] = tf.maximum(alpha * layers_output[i], layers_output[i])
        
        # Logits and sigmoid output
        logits = tf.layers.dense(layers_output[-1], 1, activation=None)
        out = tf.sigmoid(logits)
        
        return out, logits, layers_output[1:]

class Generator(tf.keras.Model):
  def __init__(self, input_shape, output_shape=2):
    super(Generator, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=128, input_shape=(input_shape,), activation=tf.nn.leaky_relu, dtype=tf.float64)
    self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu, dtype=tf.float64)
    self.dense3 = tf.keras.layers.Dense(units=output_shape, activation=None, dtype=tf.float64)

  def call(self, inputs):
    """Run the model."""
    result = self.dense1(inputs)
    result = self.dense2(result)
    result = self.dense3(result)
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