import numpy as np
import pandas as pd
import tensorflow as tf

def model_inputs(benign_dim, z_dim, attack_remains_dim):
    inputs_benign = tf.placeholder(tf.float64, (None, benign_dim), name='input_benign') 
    inputs_z = tf.placeholder(tf.float64, (None, z_dim), name='input_z')
    inputs_attack_remains = tf.placeholder(tf.float64, (None, attack_remains_dim), name='input_attack_remains')
    
    return inputs_benign, inputs_z, inputs_attack_remains

def get_flow_dataset(filename=None, benign_label=0, attack_label=1):
    if filename is None:
        feature = np.random.uniform(size=[20000,40])
        label = np.random.randint(2, size=[20000,1])
        benign, attack = split_benign_attack(np.concatenate([feature,label], axis=1))
        return get_same_len_benign_attack(benign, attack)
    else:
        # TODO
        df = pd.read_csv(filename)
        return df
    
def get_same_len_benign_attack(benign, attack, shuffle=True):
    min_len = min(benign.shape[0], attack.shape[0])
    if shuffle and shuffle is True:
        np.random.shuffle(benign)
        np.random.shuffle(attack)
    return benign[:min_len], attack[:min_len]
    
def split_benign_attack(dataset):
    benign, attack = [], []
    for row in dataset:
        if row[-1] == 0:
            benign.append(row[:-1])
        else:
            attack.append(row[:-1])
    return np.array(benign), np.array(attack)