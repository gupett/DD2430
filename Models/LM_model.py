from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding


class LM_Model:

    def __init__(self, vocab_size, look_back, batch_size, embedding=None):
        self.vocab_size = vocab_size
        self.look_back = look_back
        # Pre trained embedding, word2vec, glove...
        self.embedding = embedding

        self.model = self.define_model(batch_size, self.look_back)
        # print(model.summary())
        # Now the model is evaluated based on accuracy/ should it be evaluated based on loss?
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def define_model(self, batch_size, look_back, state_full=False):
        # Create the LM model
        model = Sequential()
        if self.embedding == None:
            # Add the embedding layer, init a uniform matrix with nr_words = vocab_size and each
            # word is encoded as a 10d vector, one word at a time will be inputted to the network
            model.add(Embedding(
                input_dim=self.vocab_size,  # e.g, 10 if you have 10 words in your vocabulary
                output_dim=100,  # size of the embedded vectors
                input_length=look_back,
                batch_input_shape=(batch_size, look_back)
            ))
        else:
            model.add(Embedding(
                input_dim=self.embedding.shape[0],  # vocabulary size, e.g, 10 if you have 10 words in your vocabulary
                output_dim=self.embedding.shape[1],  # size of the embedded vectors
                weights=self.embedding, # the pre trained word embedding
                input_length=look_back,
                trainable=False, # Freezes the layer so that the embedding does not get changed during back prop
                batch_input_shape=(batch_size, look_back)
            ))

        # The output from the lstm is a 50 dim vector, return_sequences=True is needed so that the next layer in the LSTM will get its input
        model.add(LSTM(50, stateful=state_full, return_sequences=True))
        # Another LSTM layer
        model.add(LSTM(50, stateful=state_full))
        # A final dense layer with softmax activation to transform the output from LSTM to one hot rep for each word
        model.add(Dense(self.vocab_size, activation='softmax'))
        return model

    # function for solving the problem with having to predict with the same batch size as the training batch size
    # the weights of a trained model are copied to a new model with batch size 1
    def redefine_model(self, old_model, look_back=1, state_full=True):
        batch_size = 1
        model = self.define_model(batch_size, look_back=look_back, state_full=state_full)
        # copy weights from old to new model
        old_weights = old_model.get_weights()
        model.set_weights(old_weights)

        return model



