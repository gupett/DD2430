from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from time import time
import numpy as np
import pickle

#from Models.base_line_LM_model import base_line_lm_model
from Models.base_line_affect_LM_model import base_line_affect_lm_model
#from Models.LM_model import LM_Model
#from Data import DataGenerator
from Data import LMDataGemerator
from Data import LMVaildationDataGenerator

SLIDING_WINDOW_SIZE = 20
BATCH_SIZE = 20
USE_EMBEDDING = True
#MODEL = 'base_LM_Model'
#MODEL = 'LM_Model'
MODEL = 'base_affect_LM_Model'

TRAIN_EXISTING = False
USE_EXISTING_TOKENIZER = False
if TRAIN_EXISTING:
    USE_EXISTING_TOKENIZER = True

class Training:

    # data should be a tuple with trainingX, trainingY, valX and valY
    def __init__(self, pre_trained_embedding=USE_EMBEDDING):

        if MODEL == 'LM_Model':
            self.training_generator = DataGenerator.dataGenerator(sliding_window_size=SLIDING_WINDOW_SIZE, train_existing=USE_EXISTING_TOKENIZER)
            self.tokenizer = self.training_generator.tokenizer
        else:
            self.training_generator = LMDataGemerator.dataGenerator(sliding_window_size=SLIDING_WINDOW_SIZE,
                                                                    existing_tokenizer=USE_EXISTING_TOKENIZER)

            self.tokenizer = self.training_generator.tokenizer
            self.validation_generator = LMVaildationDataGenerator.validationDataGenerator(tokenizer=self.tokenizer, sliding_window_size=SLIDING_WINDOW_SIZE)

        self.vocab_size = self.training_generator.vocab_size

        #self.training_generator = generate_training_data(sliding_window_size=SLIDING_WINDOW_SIZE)
        # Extract and set up an embedding matrix from the pre-trained embedding
        self.embedding_matrix = None
        if pre_trained_embedding:
            self.embedding_matrix = self.get_embedding_matrix()
        if MODEL == 'LM_Model':
            self.lm_model = LM_Model(self.vocab_size, SLIDING_WINDOW_SIZE, BATCH_SIZE, embedding=self.embedding_matrix, use_embedding=pre_trained_embedding)
            self.model = self.lm_model.model
            if TRAIN_EXISTING:
                self.model.load_weights('./model/best_weights.hdf5')
        elif MODEL == 'base_LM_Model':
            self.bias_vector = self.get_bias_vector()
            self.lm_model = base_line_lm_model(self.vocab_size, BATCH_SIZE, SLIDING_WINDOW_SIZE, embedding=self.embedding_matrix, bias_vector=self.bias_vector, use_embedding=pre_trained_embedding)
            self.model = self.lm_model.model
        elif MODEL == 'base_affect_LM_Model':
            self.bias_vector = self.get_bias_vector()
            self.lm_model = base_line_affect_lm_model(vocab_size=self.vocab_size, batch_size=BATCH_SIZE, look_back_steps=SLIDING_WINDOW_SIZE, embedding_matrix=self.embedding_matrix, bias_vector=self.bias_vector)
            self.model = self.lm_model.model

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

        with open('./Data/Data/unigrams_wo.txt') as file:
            content = file.read()
            content = content.lower()

        content = content.replace('\n', ' ')
        content = content.replace('\t', ' ')
        unigram = []
        words = []

        for index, value in enumerate(content.split(' ')):
            if index%2 == 0:
                words.append(value)
            elif index%2 == 1:
                unigram.append(value)

        words_unigram = zip(words, unigram)

        #print(words_bigram[1:5])
        bias = np.zeros((self.vocab_size,))
        for word, unigram in words_unigram:
            index = self.tokenizer.word_index[word]
            bias[index] = unigram
        return bias


    def train(self, epochs):
        
        # Save the tokenizer to file for use at inference time
        with open('./model/tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # define a tensorboard callback
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        # define early stopping callback
        earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=25, verbose=0, mode='auto')

        #file_path = './model/weights-{epoch:02d}-{loss:.4f}.hdf5'
        file_path = './model/best_weights.hdf5'
        checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks = [checkpoint, earlystop, tensorboard]

        self.model.fit_generator(generator=self.training_generator.batch_generator(), steps_per_epoch=self.training_generator.batch_per_epoch,
                                 epochs=epochs, verbose=1, callbacks=callbacks, validation_data=self.validation_generator.batch_generator(),
                                 validation_steps=self.validation_generator.batch_per_epoch)

        # Create new model with same weights but different batch size
        new_model = self.lm_model.redefine_model(self.model)
        new_model.save_weights('./model/lm_inference_weights.hdf5')

if __name__ == '__main__':
    trainer = Training()
    '''
    tokenizer = trainer.tokenizer
    embedding = trainer.embedding_matrix
    bias = trainer.bias_vector

    test_words = ['the', 'to', 'of', 'and', 'for', 'it', 'by']

    for word in test_words:
        index = tokenizer.word_index[word]
        print('The bias for the word: {}'.format(word))
        print(bias[index])
    '''

    '''
    test_words = ['the', 'to', 'of', 'and', 'for', 'it', 'by']

    for word in test_words:
        index = tokenizer.word_index[word]
        en = embedding[index]
        print('The embedding for the word: {} is'.format(word))
        print(en[0:10])

    '''
    trainer.train(2)
