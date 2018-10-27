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