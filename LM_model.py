from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding


class LM_Model:

    def __init__(self, vocab_size, look_back):
        self.batch_size = 4
        self.look_back = 1
        self.vocab_size = vocab_size
        self.look_back = look_back

        # Create the LM model
        self.model = Sequential()
        # Add the embedding layer, init a uniform matrix with nr_words = vocab_size and each
        # word is encoded as a 10d vector, one word at a time will be inputted to the network
        self.model.add(Embedding(
            input_dim=self.vocab_size,  # e.g, 10 if you have 10 words in your vocabulary
            output_dim=10,  # size of the embedded vectors
            input_length=self.look_back,
            batch_input_shape=(self.batch_size, self.look_back)
        ))
        # The output from the lstm is a 50 dim vector, return_sequences=True is needed so that the next layer in the LSTM will get its input
        self.model.add(LSTM(50, stateful=True, return_sequences=True))
        # Another LSTM layer
        self.model.add(LSTM(50, stateful=True))
        # Afinal dense layer with softmax activation to transform the output from lstm to one hot rep for each word
        self.model.add(Dense(self.vocab_size, activation='softmax'))
        # print(model.summary())
        # Now the model is evaluated based on accuracy/ should it be evaluated based on loss?
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
