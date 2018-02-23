import argparse

import numpy as np
import tensorflow as tf
from keras import backend as K


parser = argparse.ArgumentParser(description="Training End-To-End Memory Networks for Question Answering Tasks")
parser.add_argument('-ep', dest="epochs", type=int, default=10, help="Number of epochs")
parser.add_argument('-bs', dest="batch_size", type=int, default=16, help="Batch Size")

if __name__ == '__main__':
    parsed_args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    # Train the model

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()