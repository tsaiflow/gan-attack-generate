import numpy as np
import tensorflow as tf

# We need two inputs, one for the discriminator and one for the generator. Here we'll call the discriminator input `inputs_real` and the generator input `inputs_z`. We'll assign them the appropriate sizes for each of the networks.
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    return inputs_real, inputs_z

def get_flow_dataset(filename=None):
    if filename is None:
        return np.random.normal(size=[20000, 40])
    else:
        pass

def train_test_split(dataset, train_ratio=None, train_nums=None):
    if train_ratio is not None:
        train_nums = int(dataset.shape[0] * train_ratio)
        return dataset[:train_nums], dataset[train_nums:]
    else:
        return dataset[:train_nums], dataset[train_nums:]