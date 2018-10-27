import numpy as np
import pickle
from generate_training_data import generate_training_data

from Models.LM_model import LM_Model

SLIDING_WINDOW_SIZE = 20
BACH_SIZE = 4

# Function for finding the largest number less than K+1 divisible by X
def largest(X, K):
    # returning ans
    return (K - (K % X))

class Training:

    # data should be a tuple with trainingX, trainingY, valX and valY
    def __init__(self, pre_trained_embedding=False):

        # Extract and set up an embedding matrix from the pre-trained embedding
        if pre_trained_embedding:
            embedding_matrix = self.get_embedding_matrix()


        training_data = generate_training_data(sliding_window_size=SLIDING_WINDOW_SIZE)
        self.vocab_size = training_data.vocab_size
        self.tokenizer = training_data.tokenizer
        self.lm_model = LM_Model(self.vocab_size, SLIDING_WINDOW_SIZE, BACH_SIZE)
        self.model = self.lm_model.model

        X, y = training_data.training_data
        # Make sure length of training data is devisable, by the batch size, is a must since using stateful LSTM
        self.trainingX = X[0:largest(BACH_SIZE, X.shape[0])]
        print(self.trainingX.shape)
        self.trainingY = y[0:largest(BACH_SIZE, y.shape[0])]
        print(self.trainingY.shape)

        # Do not know if we should hav validation data
        self.valX = X[0:largest(BACH_SIZE, X.shape[0])]
        self.valY = y[0:largest(BACH_SIZE, y.shape[0])]

    def get_embedding_matrix(self):
        # load the entire embedding from file into a dictionary
        embeddings_index = dict()
        f = open('./Word_embedding/glove.6B.100d.txt')
        for line in f:
            # splits on spaces
            values = line.split()
            # the word for the vector is the first word on the row
            word = values[0]
            # Extra the vector corresponding to the word
            vector = np.asarray(values[1:], dtype='float32')
            # Add word (key) and vector (value) to dictionary
            embeddings_index[word] = vector
        f.close()

        # Initialize a embedding matrix with shape vocab_size x word_vector_size
        embedding_matrix = np.zeros((self.vocab_size, 100))
        # Go through the tokenizer and for each index add the corresponding word vector to the row
        for word, index in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix

    def train(self, epochs):
        # With multiple non cohesive texts, this should be a loop over the texts, so that the model is reset
        # foreach new sequence of text
        val_acc = 0
        model_history = dict()
        model_history['loss'] = []; model_history['val_loss'] = []
        model_history['acc'] = []; model_history['val_acc'] = []
        for i in range(epochs):
            epoch_history = self.model.fit(self.trainingX, self.trainingY, epochs=1, batch_size=BACH_SIZE, verbose=2, shuffle=False, validation_data=(self.valX, self.valY))

            # Store history
            model_history['loss'].append(epoch_history.history['loss'])
            model_history['val_loss'].append(epoch_history.history['val_loss'])
            model_history['acc'].append(epoch_history.history['acc'])
            model_history['val_acc'].append(epoch_history.history['val_acc'])

            # Reset the memory cell and hidden node for each epoch, a new sequence will be started
            self.model.reset_states()

            print(epoch_history.history['val_acc'])
            # Check if val acc has increased in such case save the model
            if epoch_history.history['val_acc'][0] > val_acc:
                # Store model
                self.model.save('./model/lm_model_epoch{}.hdf5'.format(i))
                val_acc = epoch_history.history['val_acc']
                # save training history for model
                with open("./model/history/mode_history_{}".format(i), "wb") as file_pi:
                    pickle.dump(model_history, file_pi)


        # Store the model after the last epoch
        self.model.save('./model/lm_model_final.hdf5')
        # save training history for model
        with open("./model/history/mode_history_final", "wb") as file_pi:
            pickle.dump(model_history, file_pi)

        # Create new model with same weights but different batch size
        new_model = self.lm_model.redefine_model(self.model)
        new_model.save('./model/lm_inference_model.hdf5')

        # Save the tokenizer to file for use at inference time
        with open('./model/tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    trainer = Training()
    trainer.train(5)