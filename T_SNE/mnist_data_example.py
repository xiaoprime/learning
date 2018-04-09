# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

TENSORBOARD_FOLDER = os.getcwd()+'/__tensorboard__'

metadata = os.path.join(TENSORBOARD_FOLDER, 'metadata.tsv')

mnist = input_data.read_data_sets(TENSORBOARD_FOLDER, one_hot=True)
images = tf.Variable(mnist.test.images, name='images')
#def save_metadata(file):
with open(metadata, 'w') as metadata_file:
    for row in range(10000):
        c = np.nonzero(mnist.test.labels[::1])[1:][0][row]
        metadata_file.write('{}\n'.format(c))



with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(TENSORBOARD_FOLDER, 'images.ckpt'))

    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(TENSORBOARD_FOLDER), config)
    
# for calling the tensorboard you should be in that drive and call the entire path
#tensorboard --logdir=/Technical_works/tensorflow/mnist-tensorboard/log-1 --port=6006
