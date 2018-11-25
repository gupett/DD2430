from tensorflow.keras import models
from tensorflow.keras.layers import Dense, LSTM, Input, Add, Lambda, CuDNNLSTM, Softmax
from tensorflow.keras.activations import sigmoid
import numpy as np

class base_line_affect_lm_model:

    def __init__(self, vocab_size, batch_size, look_back_steps, embedding_matrix, bias_vector):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.look_back_steps = look_back_steps
        self.embedding = np.transpose(embedding_matrix)
        self.bias_vector = bias_vector

        # Parameter for tuning the affect energy term
        self.start_weights = []
        self.model = self.define_model(look_back_steps, batch_size)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def define_model(self, look_back_steps, batch_size, state_ful=False):

        # The base LM model
        # batch_input_shape=(batch_size, timesteps, data_dim)
        if state_ful:
            word_input = Input((look_back_steps, 1), batch_size=batch_size)
        else:
            word_input = Input((look_back_steps, 1))
        # The output vector from the LSTM layer is 200d and return_sequences=True must be set in order to use
        #lstm_layer1 = CuDNNLSTM(100, return_sequences=True, stateful=state_ful)(word_input)
        #lstm_layer2 = CuDNNLSTM(100, stateful=state_ful)(lstm_layer1)
        lstm_layer1 = LSTM(100, return_sequences=True, stateful=state_ful)(word_input)
        lstm_layer2 = LSTM(100, stateful=state_ful)(lstm_layer1)
        # Set the weights to the weights corresponding to the embedding
        dense_word_decoder = Dense(self.vocab_size, trainable=False, name='embedding_layer')(lstm_layer2)
        final_prediction = Softmax()(dense_word_decoder)


        #print('Embedding matrix shape: {}'.format(self.embedding.shape))


        # The energy term for affect words
        # The input is the LIWC feature extraction which has 5 categories (5 index input vector)
        affect_input = Input((5,))
        dense_1 = Dense(50, activation=sigmoid)(affect_input)
        dense_2 = Dense(100, activation=sigmoid)(dense_1)
        # The report does not say anything about the activation function for the second layer
        dense_affect_decoder_1 = Dense(self.vocab_size, trainable=False, name='embedding_layer')(dense_2)
        # multiply the affect vector with the beta energy term
        #beta = Input((1,))
        beta = 1.75
        dense_affect_decoder = Lambda(lambda x: x*beta)(dense_affect_decoder_1)

        # the bias term for unigram occurrences has not been added to the model
        final_layer = Add()([dense_word_decoder, dense_affect_decoder])

        model = models.Model(inputs=[word_input, affect_input], outputs=[final_layer])

        weights = model.get_layer('embeddin_layer').get_weights()

        #print(weights[1].shape)
        weights[0] = self.embedding
        weights[1] = self.bias_vector
        #print(weights)
        model.get_layer('embedding_layer').set_weights(weights)

        self.start_weights.append(weights[0])
        self.start_weights.append(weights[1])

        model.summary()

        return model

        # function for solving the problem with having to predict with the same batch size as the training batch size
        # the weights of a trained model are copied to a new model with batch size 1

    def redefine_model(self, old_model, look_back=1, state_ful=True):
        batch_size = 1
        model = self.define_model(look_back, batch_size, state_ful=state_ful)
        # copy weights from old to new model
        old_weights = old_model.get_weights()
        model.set_weights(old_weights)

        return model

'''
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Input, Add, Lambda
from tensorflow.keras.activations import sigmoid, softmax
import numpy as np

class base_line_lm_model:

    def __init__(self, vocab_size, batch_size, look_back_steps,embedding=None, bias_vector=None, use_embedding=False):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.look_back_steps = look_back_steps
        # Parameter for tuning the affect energy term
        self.model = self.define_model(look_back_steps, data_dim=1, embedding=embedding, bias_vector=bias_vector, use_embedding=use_embedding)


    def define_model(self, timesteps, data_dim, embedding=None, stateful=False, bias_vector=None, use_embedding=False):
        # The base LM model
        word_inputs = Input((timesteps, data_dim))
        # The output vector from the LSTM layer is lstm_size (which is the nr of features of the output vector) and
        # return_sequences=True must be set in order to use several layers of LSTM
        # batch_input_shape=(batch_size, timesteps, data_dim)
        lstm_size = 200
        if use_embedding:
            lstm_size = embedding.shape[1]
        lstm_layer1 = LSTM(lstm_size, return_sequences=True, stateful=stateful)(word_inputs)
        lstm_layer2 = LSTM(lstm_size, stateful=stateful)(lstm_layer1)
        # the bias term for unigram occurrences has not been added to the model

        # define and freeze weights if embedding should be used
        if use_embedding:
            W = np.transpose(embedding)
            b = bias_vector
            predictions = Dense(self.vocab_size, activation=softmax, weights=[W,b], trainable=True)(lstm_layer2)
            print('embedding size: {}'.format(embedding.shape))
        else:
            predictions = Dense(self.vocab_size, activation=softmax)(lstm_layer2)

        base_model = Model(inputs=word_inputs, outputs=predictions)




        optimizer = Adam()
        base_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        # the output shapes in the summary are (batch_size, timesteps, data_dim) for LSTM
        # and (batch_size, data_dim) for dense layers. When a layer is None it is yet not defined
        # for example batch_size is defined when fitting the model
        print(base_model.summary())

        return base_model
'''