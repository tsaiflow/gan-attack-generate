import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
        RELEVANT_FEATURES = [' Source Port', ' Protocol', ' Flow Duration', ' Total Fwd Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', 'Bwd Packet Length Max', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Max', 'Fwd IAT Total', ' Fwd IAT Max', ' Fwd IAT Min', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', 'Fwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', 'FIN Flag Count', ' RST Flag Count', ' ACK Flag Count', ' CWE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', 'Bwd Avg Bulk Rate', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' Active Min', 'Idle Mean', ' Idle Std']
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