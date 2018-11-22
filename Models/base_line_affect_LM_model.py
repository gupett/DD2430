from tensorflow.keras import models
from tensorflow.keras.layers import Dense, LSTM, Input, Add, Lambda
from tensorflow.keras.activations import sigmoid
import numpy as np

class base_line_affect_lm_model:

    def __init__(self, vocab_size, batch_size, look_back_steps, embedding_matrix):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.look_back_steps = look_back_steps
        self.embedding = np.transpose(embedding_matrix)

        # Parameter for tuning the affect energy term
        self.model = self.define_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def define_model(self):

        # The base LM model
        # batch_input_shape=(batch_size, timesteps, data_dim)
        word_input = Input((self.look_back_steps, 1))
        # The output vector from the LSTM layer is 200d and return_sequences=True must be set in order to use
        lstm_layer1 = LSTM(200, return_sequences=True)(word_input)
        lstm_layer2 = LSTM(100)(lstm_layer1)
        # Set the weights to the weights corresponding to the embedding
        dense_word_decoder = Dense(self.vocab_size)(lstm_layer2)


        print('Embedding matrix shape: {}'.format(self.embedding.shape))


        # The energy term for affect words
        # The input is the LIWC feature extraction which has 5 categories (5 index input vector)
        affect_input = Input((5,))
        dense_1 = Dense(100, activation=sigmoid)(affect_input)
        dense_2 = Dense(100, activation=sigmoid)(dense_1)
        # The report does not say anything about the activation function for the second layer
        dense_affect_decoder_1 = Dense(self.vocab_size, trainable=False)(dense_2)
        # multiply the affect vector with the beta energy term
        #beta = Input((1,))
        beta = 1
        dense_affect_decoder = Lambda(lambda x: x*beta)(dense_affect_decoder_1)

        # the bias term for unigram occurrences has not been added to the model
        final_layer = Add()([dense_word_decoder, dense_affect_decoder])

        model = models.Model(inputs=[word_input, affect_input], outputs=[final_layer])

        weights = model.get_layer('dense').get_weights()
        weights[0] = self.embedding
        model.get_layer('dense').set_weights(weights)

        model.summary()

        return model