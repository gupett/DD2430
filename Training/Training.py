from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from time import time
import numpy as np
import pickle

from Models.base_line_LM_model import base_line_lm_model
from Models.LM_model import LM_Model
from Data import DataGenerator

SLIDING_WINDOW_SIZE = 20
BATCH_SIZE = 20
USE_EMBEDDING = True
MODEL = 'base_LM_Model'
MODEL = 'LM_Model'

class Training:

    # data should be a tuple with trainingX, trainingY, valX and valY
    def __init__(self, pre_trained_embedding=USE_EMBEDDING):

        #self.traning_generator = generate_training_data(sliding_window_size=SLIDING_WINDOW_SIZE)
        self.traning_generator = DataGenerator.dataGenerator()
        self.vocab_size = self.traning_generator.vocab_size
        self.tokenizer = self.traning_generator.tokenizer

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

        # define a tensorboard callback
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        # define early stopping callback
        earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')

        #file_path = './model/weights-{epoch:02d}-{loss:.4f}.hdf5'
        file_path = './model/best_weights.hdf5'
        checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks = [checkpoint, earlystop, tensorboard]

        self.model.fit_generator(generator=self.traning_generator.batch_generator(), steps_per_epoch=self.traning_generator.batch_per_epoch, epochs=epochs, verbose=1, callbacks=callbacks)

        # Create new model with same weights but different batch size
        new_model = self.lm_model.redefine_model(self.model)
        new_model.save_weights('./model/lm_inference_weights.hdf5')

        # Save the tokenizer to file for use at inference time
        with open('./model/tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    trainer = Training()
    trainer.train(3)
