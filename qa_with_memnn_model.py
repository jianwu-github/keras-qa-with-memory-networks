import glob
import numpy as np

import tensorflow as tf
from keras import backend as K


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    # Testing Memory-To-Memory Network Model with question answering tasks

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()