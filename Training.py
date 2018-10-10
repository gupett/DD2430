import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
import numpy as np
import pickle

from LM_model import LM_Model

LOOK_BACK = 1
BACH_SIZE = 2

# Function for finding the largest number less than K+1 divisible by X
def largest(X, K):
    # returning ans
    return (K - (K % X))

class Training:

    # data should be a tuple with trainingX, trainingY, valX and valY
    def __init__(self, data):
        self.model = LM_Model(LOOK_BACK, BACH_SIZE).model
        # Make sure length of training data is devisable, by the batch size, is a must since using stateful LSTM
        self.trainingX = data[0]
        self.trainingX = self.trainingX[0:largest(BACH_SIZE, data.shape[0])]
        self.trainingY = data[1]
        self.trainingY = self.trainingY[0:largest(BACH_SIZE, data.shape[0])]

        # Do not know if we should hav validation data
        self.valX = data[2]
        self.valY = data[3]

    def train(self, epochs):
        # With multiple non cohesive texts, this should be a loop over the texts, so that the model is reset
        # foreach new sequence of text
        val_acc = 0
        model_history = dict()
        model_history['loss'] = []; model_history['val_loss'] = []
        model_history['acc'] = []; model_history['val_acc'] = []
        for i in range(epochs):
            epoch_history = self.model.fit(self.traingX, self.trainingY, epochs=1, batch_size=4, verbose=2, shuffle=False, validation_data=(self.valX, self.valY))

            # Store history
            model_history['loss'].append(epoch_history.history['loss'])
            model_history['val_loss'].append(epoch_history.history['val_loss'])
            model_history['acc'].append(epoch_history.history['acc'])
            model_history['val_acc'].append(epoch_history.history['val_acc'])

            # Reset the memory cell and hidden node for each epoch, a new sequence will be started
            self.model.reset_states()

            # Check if val acc has increased in such case save the model
            if epoch_history.histort['val_acc'] > val_acc:
                # Store model
                self.model.save('./model/lm_model_epoch{}.hdf5'.format(i))
                val_acc = epoch_history.histort['val_acc']
                # save training history for model
                with open("./history/mode_history_{}".format(i), "wb") as file_pi:
                    pickle.dump(model_history, file_pi)


        # Store the model after the last epoch
        self.model.save('./model/lm_model_final.hdf5')
        # save training history for model
        with open("./history/mode_history_final", "wb") as file_pi:
            pickle.dump(model_history, file_pi)