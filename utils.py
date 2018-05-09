import numpy as np
import pandas as pd
import tensorflow as tf

def model_inputs(benign_dim, z_dim, attack_remains_dim):
    inputs_benign = tf.placeholder(tf.float32, (None, benign_dim), name='input_benign') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    inputs_attack_remains = tf.placeholder(tf.float32, (None, attack_remains_dim), name='input_attack_remains')
    
    return inputs_benign, inputs_z, inputs_attack_remains

def get_flow_dataset(filename=None):
    if filename is None:
        feature = np.random.uniform(size=[20000,40])
        label = np.random.randint(2, size=[20000,1])
        return np.concatenate([feature,label], axis=1)
    else:
        # TODO
        pass
    
def split_benign_attack(dataset):
    benign, attack = [], []
    for row in dataset:
        if row[-1] == 1:
            benign.append(row[:-1])
        else:
            attack.append(row[:-1])
    return np.array(benign), np.array(attack)