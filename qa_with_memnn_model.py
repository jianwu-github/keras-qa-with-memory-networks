import pickle
import re

import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from keras_memory_to_memory_net_builder import KerasMemToMemNNBuilder

_DEFAULT_MODEL_DIR = "model"
_DEFAULT_VOCAB_SIZE = 22
_DEFAULT_MAX_STORY_LEN = 68
_DEFAULT_MAX_QUESTION_LEN = 4


# Helper functions
def tokenize(sentence):
    return [x.strip() for x in re.split('(\W+)', sentence) if x.strip()]


def vectorize_stories(stories, word_idx, max_story_len, max_query_len):
    inputs, queries, answers = [], [], []
    for story, query, answer in stories:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=max_story_len), pad_sequences(queries, maxlen=max_query_len), np.array(answers))


def load_memnn_model(model_file, vocab_file):
    vocab = pickle.load(open(vocab_file, "rb"))
    vocab_size = len(vocab) + 1

    memnn_builder = KerasMemToMemNNBuilder(vocab_size, _DEFAULT_MAX_STORY_LEN, _DEFAULT_MAX_QUESTION_LEN)
    model = memnn_builder.build_mem_to_mem_networks(64)
    model.load_weights(model_file)

    return vocab, model


def test_question_answering():
    model_file = _DEFAULT_MODEL_DIR + "/memnn_model.h5"
    vocab_file = _DEFAULT_MODEL_DIR + "/vocab.pkl"

    vocab, model = load_memnn_model(model_file, vocab_file)

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    test_story = "Sandra went to the hallway . John journeyed to the bathroom . Sandra travelled to the office ."
    test_question = "Where is Sandra ?"

    test_qa = (tokenize(test_story), tokenize(test_question), "?")

    sample_story, sample_question, sample_answer = vectorize_stories([test_qa], word_idx, _DEFAULT_MAX_STORY_LEN, _DEFAULT_MAX_QUESTION_LEN)

    pred = model.predict([sample_story, sample_question])
    pred = np.argmax(pred, axis=1)

    print("Story Vocabularies: {} \n".format(vocab))
    print("Test Story: {} \n".format(test_story))
    print("Test Question: {} \n".format(test_question))
    print("Predicted Answer: {}({}) \n".format(vocab[pred[0] - 1], pred))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    # Testing Memory-To-Memory Network Model with question answering tasks
    test_question_answering()

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()