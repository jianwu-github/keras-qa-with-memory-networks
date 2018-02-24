from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM

class KerasMemToMemNNBuilder:

    def __init__(self, num_of_vocabularies, max_story_len, max_question_len):
        self._num_of_vocabularies = num_of_vocabularies
        self._max_story_len = max_story_len
        self._max_question_len = max_question_len

    def build_mem_to_mem_networks(self, output_dim):
        # placeholders
        story = Input((self._max_story_len,))
        question = Input((self._max_question_len,))

        # encoders
        # embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=self._num_of_vocabularies, output_dim=output_dim))
        input_encoder_m.add(Dropout(0.3))
        # output: (samples, _max_story_len, embedding_dim)

        # embed the input into a sequence of vectors of size _max_question_len
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=self._num_of_vocabularies, output_dim=self._max_question_len))
        input_encoder_c.add(Dropout(0.3))
        # output: (samples, _max_story_len, _max_question_len)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=self._num_of_vocabularies, output_dim=output_dim, input_length=self._max_question_len))
        question_encoder.add(Dropout(0.3))
        # output: (samples, _max_question_len, embedding_dim)

        # encode story and questions (which are indices) to sequences of dense vectors
        input_encoded_m = input_encoder_m(story)
        input_encoded_c = input_encoder_c(story)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence and the question vector sequence
        # shape: `(samples, _max_story_len, _max_query_len)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, _max_story_len, _max_query_len)
        response = Permute((2, 1))(response)      # (samples, _max_query_len, _max_story_len)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step, we choose to use a RNN instead.
        answer = LSTM(32)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(0.3)(answer)
        answer = Dense(self._num_of_vocabularies)(answer)  # (samples, _num_of_vocabularies)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # Finally, build the model
        model = Model([story, question], answer)
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

