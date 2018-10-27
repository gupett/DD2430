from tensorflow.keras import models
from tensorflow.keras.layers import Dense, LSTM, Input, Add, Lambda
from tensorflow.keras.activations import sigmoid

class base_line_affect_lm_model:

    def __init__(self, vocab_size, batch_size, look_back_steps):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.look_back_steps = look_back_steps
        # Parameter for tuning the affect energy term
        self.model = self.define_model()

    def define_model(self):

        # The base LM model
        word_input = Input((self.vocab_size, self.batch_size))
        # The output vector from the LSTM layer is 200d and return_sequences=True must be set in order to use
        lstm_layer1 = LSTM(200, return_sequences=True)(word_input)
        lstm_layer2 = LSTM(200)(lstm_layer1)
        dense_word_decoder = Dense(self.vocab_size)

        # The energy term for affect words
        # The input is the LIWC feature extraction which has 5 categories
        affect_input = Input((5,self.batch_size))
        dense_1 = Dense(100)(affect_input)
        activation_1 = sigmoid()(dense_1)
        dense_2 = Dense(200)(activation_1)
        # The report does not say anything about the activation function for the second layer
        activation_2 = sigmoid()(dense_2)
        dense_affect_decoder_1 = Dense(self.vocab_size)(activation_2)
        # multiply the affect vector with the beta energy term
        beta = Input((1,))
        dense_affect_decoder_2 = Lambda(lambda x: x*beta)

        # the bias term for unigram occurrences has not been added to the model
        final_layer = Add(dense_word_decoder, dense_affect_decoder_2)

        return models.Model(inputs=[word_input, affect_input, beta], outputs=[final_layer])