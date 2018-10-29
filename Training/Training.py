from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
from generate_training_data import generate_training_data

from Models.base_line_LM_model import base_line_lm_model
from Models.LM_model import LM_Model

SLIDING_WINDOW_SIZE = 20
BATCH_SIZE = 4
USE_EMBEDDING = True
MODEL = 'base_LM_Model'
MODEL = 'LM_Model'

# Function for finding the largest number less than K+1 divisible by X
def largest(X, K):
    # returning ans
    return (K - (K % X))

# function for preparing data for specific model
# For the LM_model
def prepare_data_for_model1(X, y):
    # Make sure length of training data is devisable, by the batch size, is a must since using stateful LSTM
    trainingY = y[0:largest(BATCH_SIZE, y.shape[0])]
    trainingX = X[0:largest(BATCH_SIZE, X.shape[0])]
    # Do not know if we should hav validation data
    valX = X[0:largest(BATCH_SIZE, X.shape[0])]
    valY = y[0:largest(BATCH_SIZE, y.shape[0])]

    return trainingX, trainingY, valX, valY

# For the base_line models
def prepare_data_for_base_line_model(X, y):
    # Make sure length of training data is devisable, by the batch size, is a must since using stateful LSTM
    trainingY = y[0:largest(BATCH_SIZE, y.shape[0])]
    trainingX = X[0:largest(BATCH_SIZE, X.shape[0])]
    # Do not know if we should hav validation data
    valX = X[0:largest(BATCH_SIZE, X.shape[0])]
    valY = y[0:largest(BATCH_SIZE, y.shape[0])]

    trainingX = trainingX.reshape(trainingX.shape[0], SLIDING_WINDOW_SIZE, 1)
    valX = valX.reshape(valX.reshape[0], SLIDING_WINDOW_SIZE, 1)

    return trainingX, trainingY, valX, valY

class Training:

    # data should be a tuple with trainingX, trainingY, valX and valY
    def __init__(self, pre_trained_embedding=USE_EMBEDDING):

        self.traning_generator = generate_training_data(sliding_window_size=SLIDING_WINDOW_SIZE)
        self.vocab_size = self.traning_generator.vocab_size
        self.tokenizer = self.traning_generator.tokenizer



        # Get training data
        #X, y = self.traning_generator.training_data
        # Make sure length of training data is devisable, by the batch size, is a must since using stateful LSTM
        #self.trainingY = y[0:largest(BATCH_SIZE, y.shape[0])]
        #self.trainingX = X[0:largest(BATCH_SIZE, X.shape[0])]
        # Do not know if we should hav validation data
        #self.valX = X[0:largest(BATCH_SIZE, X.shape[0])]
        #self.valY = y[0:largest(BATCH_SIZE, y.shape[0])]

        # Extract and set up an embedding matrix from the pre-trained embedding
        self.embedding_matrix = None
        if pre_trained_embedding:
            self.embedding_matrix = self.get_embedding_matrix()
        if MODEL == 'LM_Model':
            self.lm_model = LM_Model(self.vocab_size, SLIDING_WINDOW_SIZE, BATCH_SIZE, embedding=self.embedding_matrix, use_embedding=pre_trained_embedding)
            self.model = self.lm_model.model
        elif MODEL == 'base_LM_Model':
            bias_vector = self.get_bias_vector()
            self.base_model = base_line_lm_model(self.vocab_size, BATCH_SIZE, SLIDING_WINDOW_SIZE, embedding=self.embedding_matrix, bias_vector=bias_vector, use_embedding=pre_trained_embedding)
            self.model = self.base_model.model
            #print('shape of training data: {}'.format(self.trainingX.shape))
            # Reshape training data so it fits the base_LM_model
            # The first dimensions are (nr_training_samples, nr_time_steps_for_sample, feature_for_time_steps)
            #self.trainingX = self.trainingX.reshape(self.trainingX.shape[0], SLIDING_WINDOW_SIZE, 1)

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

    def get_bias_vector(self):
        bias = np.zeros((self.vocab_size,))
        return bias


    def train(self, epochs):
        # With multiple non cohesive texts, this should be a loop over the texts, so that the model is reset
        # foreach new sequence of text
        val_acc = 0
        model_history = dict()
        model_history['loss'] = []; model_history['val_loss'] = []
        model_history['acc'] = []; model_history['val_acc'] = []
        
        #file_path = './model/weights-{epoch:02d}-{loss:.4f}.hdf5'
        #file_path = './model/best_weights.hdf5'
        #checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        #callbacks = [checkpoint]



        #self.model.fit(trainingX, trainingY, epochs=epochs, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)
        epochs = self.traning_generator.nr_batch_files * epochs
        for i in range(epochs):

            X, y = self.traning_generator.get_next_batch()
            if MODEL == 'LM_Model':
                trainingX, trainingY, valX, valY = prepare_data_for_model1(X, y)
            elif MODEL == 'base_LM_Model':
                trainingX, trainingY, valX, valY = prepare_data_for_base_line_model(X, y)

            epoch_history = self.model.fit(trainingX, trainingY, epochs=1, batch_size=BATCH_SIZE, verbose=2, shuffle=True, validation_data=(valX, valY))

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
                self.model.save_weights('./model/best_weights.hdf5')
                val_acc = epoch_history.history['val_acc']
                # save training history for model
                with open("./model/history/mode_history", "wb") as file_pi:
                    pickle.dump(model_history, file_pi)

        # Make sure that the model_weights with the best accuracy is stored
        if epoch_history.history['val_acc'][0] < val_acc:
            self.model.load_weights('./model/best_weights.hdf5')

        # Store the model after the last epoch
        # self.model.save('./model/lm_model_final.hdf5')
        # save training history for model
        #with open("./model/history/mode_history_final", "wb") as file_pi:
        #    pickle.dump(model_history, file_pi)

        # Create new model with same weights but different batch size
        new_model = self.lm_model.redefine_model(self.model)
        new_model.save_weights('./model/lm_inference_weights.hdf5')

        # Save the tokenizer to file for use at inference time
        with open('./model/tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    trainer = Training()
    trainer.train(5)
