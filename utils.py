import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe
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

def discriminator_loss(discriminator_benign_outputs, discriminator_gen_outputs):
    """Original discriminator loss for GANs, with label smoothing.
    See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
    details.
    Args:
    discriminator_benign_outputs: Discriminator output on benign data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
    to be in the range of (-inf, inf).
    Returns:
    A scalar loss Tensor.
    """

    loss_on_benign = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_benign_outputs),
        discriminator_benign_outputs)
    loss_on_generated = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_gen_outputs),
        discriminator_gen_outputs)
    loss = loss_on_benign + loss_on_generated
    tf.contrib.summary.scalar('discriminator_loss', loss)
    return loss

def generator_loss(discriminator_gen_outputs):
    """Original generator loss for GANs.
    L = -log(sigmoid(D(G(z))))
    See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661)
    for more details.
    Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    Returns:
    A scalar loss Tensor.
    """
    loss = tf.losses.sigmoid_cross_entropy(
         tf.zeros_like(discriminator_gen_outputs), discriminator_gen_outputs)
    tf.contrib.summary.scalar('generator_loss', loss)
    return loss

def select_features(dataset, selected_feat):
    sorted_selected_feat = sorted(selected_feat)
    selected_feature_dataset = tf.expand_dims(dataset[:, sorted_selected_feat[0]], 1)
    for feat_num in sorted_selected_feat[1:]:
        selected_feature_dataset = tf.concat([
                selected_feature_dataset,
                tf.expand_dims(dataset[:, feat_num], 1)
            ],
            axis=1)
    return selected_feature_dataset

def concatenate_generated_remained(attack_feat, generated_part_features, selected_feat):
    concat_attack_feat = None
    j = 0
    for i in range(attack_feat.shape[1]):
        if i in selected_feat:
            if concat_attack_feat is None:
                concat_attack_feat = tf.expand_dims(generated_part_features[:, j], 1)
            else:
                concat_attack_feat = tf.concat([
                    concat_attack_feat,
                    tf.expand_dims(generated_part_features[:, j], 1)
                ],
                axis=1)
            j += 1
        else:
            if concat_attack_feat is None:
                concat_attack_feat = tf.expand_dims(attack_feat[:, i], 1)
            else:
                concat_attack_feat = tf.concat([
                    concat_attack_feat,
                    tf.expand_dims(attack_feat[:, i], 1)
                ],
                axis=1)
    return concat_attack_feat

def train_one_epoch(generator, discriminator, generator_optimizer,
                    discriminator_optimizer, benign_dataset, attack_dataset, 
                    step_counter,
                    log_interval,
                    modified_feature_num):
    """Trains `generator` and `discriminator` models on `dataset`.
    Args:
    generator: Generator model.
    discriminator: Discriminator model.
    generator_optimizer: Optimizer to use for generator.
    discriminator_optimizer: Optimizer to use for discriminator.
    dataset: Dataset of images to train on.
    step_counter: An integer variable, used to write summaries regularly.
    log_interval: How many steps to wait between logging and collecting
    summaries.
    """

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    for (batch_index, ((benign_feat, _), (attack_feat, _))) in enumerate(zip(benign_dataset, attack_dataset)):
        with tf.device('/cpu:0'):
            tf.assign_add(step_counter, 1)

        with tf.contrib.summary.record_summaries_every_n_global_steps(
                log_interval, global_step=step_counter):
            current_batch_size = benign_feat.shape[0]
            feat_size = benign_feat.shape[1]
            selected_feat = sample_n_number(feat_size, modified_feature_num)
            features_to_be_modified = select_features(attack_feat, selected_feat)#tfe.Variable(attack_feat.numpy()[:, selected_feat])

            with tfe.GradientTape(persistent=True) as g:
                generated_part_features = generator(features_to_be_modified)
#                 generated_attack_feat = attack_feat.numpy().copy()
#                 generated_attack_feat[:, selected_feat] = generated_part_features.numpy()
                generated_attack_feat = concatenate_generated_remained(attack_feat, generated_part_features, selected_feat)#tfe.Variable(generated_attack_feat)
                tf.contrib.summary.image(
                    'generated_attack_flow_feature',
                    # TODO: image size
                    tf.cast(tf.reshape(generated_attack_feat, [-1, feat_size, 1, 1]), tf.float32),
                    max_images=10)

                discriminator_gen_outputs = discriminator(generated_attack_feat)
                discriminator_benign_outputs = discriminator(benign_feat)
                discriminator_loss_val = discriminator_loss(discriminator_benign_outputs, discriminator_gen_outputs)
                total_discriminator_loss += discriminator_loss_val

                generator_loss_val = generator_loss(discriminator_gen_outputs)
                total_generator_loss += generator_loss_val

                generator_grad = g.gradient(generator_loss_val, generator.variables)
                discriminator_grad = g.gradient(discriminator_loss_val,
                                          discriminator.variables)

            generator_optimizer.apply_gradients(
              zip(generator_grad, generator.variables))
            discriminator_optimizer.apply_gradients(
              zip(discriminator_grad, discriminator.variables))

            if log_interval and batch_index > 0 and batch_index % log_interval == 0:
                print('Batch #%d\tAverage Generator Loss: %.6f\t'
                  'Average Discriminator Loss: %.6f' %
                  (batch_index, total_generator_loss / batch_index,
                   total_discriminator_loss / batch_index))