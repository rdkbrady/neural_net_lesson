import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

# Create some values to supply to our neuron function

def make_inputs(input_size):
    inputs = 2 * random.random(input_size) - 1
    return inputs

def initialize_neuron(prev_layer_size):
    prev_layer = 2 * random.random(prev_layer_size) - 1
    weights = 2 * random.random(prev_layer_size) - 1
    bias = random.uniform()
    return weights, bias



def mnist_data():
    
    (ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )

    
    def normalize_img(image, label):
      """Normalizes images: `uint8` -> `float32`."""
      return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test
