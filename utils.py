import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def model_inputs(benign_dim, z_dim, attack_remains_dim):
    inputs_benign = tf.placeholder(tf.float64, (None, benign_dim), name='input_benign') 
    inputs_z = tf.placeholder(tf.float64, (None, z_dim), name='input_z')
    inputs_attack_remains = tf.placeholder(tf.float64, (None, attack_remains_dim), name='input_attack_remains')
    
    return inputs_benign, inputs_z, inputs_attack_remains

def get_flow_dataset(filename=None, benign_label=0, attack_label=1, train_frac=0.3):
    if filename is None:
        feature = np.random.uniform(size=[20000,40])
        label = np.random.randint(2, size=[20000,1])
        benign, attack = split_benign_attack(np.concatenate([feature,label], axis=1))
        return get_same_len_benign_attack(benign, attack)
    else:
        # TODO
        df = pd.read_csv(filename)
        RELEVANT_FEATURES = [' Source Port', ' Destination Port', ' Flow Duration', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', 'Bwd Packet Length Max', ' Bwd Packet Length Min', 'Flow Bytes/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Packets/s', ' Packet Length Mean', ' ACK Flag Count', ' Down/Up Ratio', ' Avg Fwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Bwd Avg Bytes/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward', ' act_data_pkt_fwd', ' Active Std', ' Active Min', ' Idle Max']
        df = df[RELEVANT_FEATURES + [' Label']]
        
        benign, attack = df[df[' Label'] == benign_label][RELEVANT_FEATURES], df[df[' Label'] == attack_label][RELEVANT_FEATURES]
        
        benign, attack = benign.as_matrix(), attack.as_matrix()
        benign_train, benign_test = train_test_split(benign, train_size=train_frac)
        attack_train, attack_test = train_test_split(attack, train_size=train_frac)
        return (benign_train, benign_test), (attack_train, attack_test) 
    
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

def max_norm(dataset):
    dataset = dataset - dataset.min(axis=0)
    dataset = normalize(dataset, axis=0, norm='max')
    return dataset

def parse_feature_label(row):
    return row[:-1], tf.cast(row[-1], tf.int32)

def loss(model, x, y):
    predict_y = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=predict_y)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

def sample_n_number(upper, n):
    return random.sample({i for i in range(upper)}, n)