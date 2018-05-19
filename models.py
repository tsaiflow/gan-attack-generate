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

class GeneratorModel(tf.keras.Model):
  def __init__(self):
    super(GeneratorModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=128)
    self.dense2 = tf.keras.layers.Dense(units=64)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    return result

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

class DiscriminatorModel(tf.keras.Model):
  def __init__(self):
    super(GeneratorModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=128)
    self.dense2 = tf.keras.layers.Dense(units=64)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    return result