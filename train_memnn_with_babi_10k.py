import argparse
import pickle
import re

from contextlib import ExitStack

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from functools import reduce

from keras_memory_to_memory_net_builder import KerasMemToMemNNBuilder

_DEFAULT_TRAINING_STORY_FILE = "data/qa1_single-supporting-fact_train.txt"
_DEFAULT_TESTING_STORY_FILE = "data/qa1_single-supporting-fact_test.txt"
_DEFAULT_MODEL_DIR = "model"


# Helper functions
def tokenize(sentence):
    return [x.strip() for x in re.split('(\W+)', sentence) if x.strip()]


def parse_stories(lines, only_supporting=False):
    data = []
    story = []

    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)

        if nid == 1:
            story = []

        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)

    return data


def get_stories(f, only_supporting=False, max_length=None):
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]

    return data


def vectorize_stories(stories, word_idx, max_story_len, max_query_len):
    inputs, queries, answers = [], [], []
    for story, query, answer in stories:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=max_story_len), pad_sequences(queries, maxlen=max_query_len), np.array(answers))


def train_memnn_model(training_story_file, testing_story_file, batch_size, epochs):
    with ExitStack() as stack:
        training_data = stack.enter_context(open(training_story_file, encoding='utf-8'))
        testing_data = stack.enter_context(open(testing_story_file, encoding='utf-8'))

        training_stories = get_stories(training_data)
        testing_stories = get_stories(testing_data)

        # Build Vocabularies
        vocab = set()
        for story, q, answer in training_stories + testing_stories:
            vocab |= set(story + q + [answer])

        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1
        max_story_len = max(map(len, (x for x, _, _ in training_stories + testing_stories)))
        max_query_len = max(map(len, (x for _, x, _ in training_stories + testing_stories)))

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

        inputs_train, queries_train, answers_train = vectorize_stories(training_stories,
                                                                       word_idx,
                                                                       max_story_len,
                                                                       max_query_len)

        inputs_test, queries_test, answers_test = vectorize_stories(testing_stories,
                                                                    word_idx,
                                                                    max_story_len,
                                                                    max_query_len)

        memnn_builder = KerasMemToMemNNBuilder(vocab_size, max_story_len, max_query_len)
        model =memnn_builder.build_mem_to_mem_networks(64)

        # train
        model.fit([inputs_train, queries_train], answers_train,
                  batch_size=32,
                  epochs=120,
                  validation_data=([inputs_test, queries_test], answers_test))

        model_file = _DEFAULT_MODEL_DIR + "/memnn_model.h5"
        vocab_file = _DEFAULT_MODEL_DIR + "/vocab.pkl"

        model.save(model_file)
        pickle.dump(vocab, open(vocab_file, "wb"))


parser = argparse.ArgumentParser(description="Training End-To-End Memory Networks for Question Answering Tasks")
parser.add_argument('-ep', dest="epochs", type=int, default=10, help="Number of epochs")
parser.add_argument('-bs', dest="batch_size", type=int, default=16, help="Batch Size")

if __name__ == '__main__':
    parsed_args = parser.parse_args()
    epochs = parsed_args.epochs
    batch_size = parsed_args.batch_size

    training_story_file = _DEFAULT_TRAINING_STORY_FILE
    testing_story_file = _DEFAULT_TESTING_STORY_FILE

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    # Train the model
    train_memnn_model(training_story_file, testing_story_file, batch_size, epochs)

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()