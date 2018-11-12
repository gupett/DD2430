from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from os import listdir
from os.path import isfile, join
import numpy as np

FILE_EXTENSION = './Data/Data/parts_small/'
FILES = [join(FILE_EXTENSION, file_name) for file_name in listdir(FILE_EXTENSION) if isfile(join(FILE_EXTENSION, file_name)) and file_name != '.DS_Store']
UNIQUE_WORD_FILE = './Data/Data/unique_words.json'

# Function for finding the largest number less than K+1 divisible by X
def largest(X, K):
    # returning ans
    return (K - (K % X))

class dataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size=20, sliding_window_size=20, shuffle=False, Affect_LM=False):
        self.batch_size = batch_size
        self.sliding_window_size = sliding_window_size
        self.shuffle = shuffle

        # Get all the unique words for every file in the training data set
        self.unique_words = self.get_unique_words_from_json_file()
        # Init a tokenizer which will translate words in to integers
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([self.unique_words])
        # print(self.tokenizer.word_index)
        # Vocabulary size
        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.batch_per_epoch = self.nr_batch_per_epoch()

        self.current_file_nr = -1
        self.x_file, self.y_file = self.load_next_sequence()
        #self.sequence_for_file = self.load_next_sequence()
        # To keep track where in the current sequence the last batch was taken from
        self.batch_in_file = 0

        # Tells which model is used
        self.Affect_LM=Affect_LM
        # self.on_epoch_end()

    '''
    def __len__(self):
        # expected function, gives the number of batches in each epoch
        # I do not know the number of batches per epoch and can not figure it out without going through every file
        return 10000

    def __getitem__(self, item):
        # Generate the data for a batch
    '''

    def nr_batch_per_epoch(self):
        batch_per_epoch = 0
        for file_path in FILES:
            print(file_path)
            with open(file_path) as file:
                file_content = file.read()

            file_content = file_content.lower()
            file_content = file_content.replace('\n', ' ')

            # Get an encoding of the text
            file_encoded_sequence = self.tokenizer.texts_to_sequences([file_content])[0]

            # the // operator gives int values
            batch_per_epoch += ((len(file_encoded_sequence)-(self.sliding_window_size+1))//(self.batch_size))
        return batch_per_epoch



    def load_next_sequence(self):
        self.current_file_nr = (self.current_file_nr + 1) % len(FILES)
        file_path = FILES[self.current_file_nr]
        with open(file_path) as file:
            file_content = file.read()

        file_content = file_content.lower()
        file_content = file_content.replace('\n', ' ')

        # Get an encoding of the text
        file_encoded = self.tokenizer.texts_to_sequences([file_content])[0]

        sequences = []
        for i in range(self.sliding_window_size + 1, len(file_encoded)+1):
            sequence = file_encoded[i - (self.sliding_window_size + 1):i]
            sequences.append(sequence)

        sequences = np.asarray(sequences)

        X = np.array(sequences[:, 0:self.sliding_window_size])
        y = np.array(sequences[:, -1])
        y_b = y[0:largest(self.batch_size, y.shape[0])]
        X_b = X[0:largest(self.batch_size, X.shape[0])]

        return X_b, y_b


    def batch_generator(self):
        # properly check might give errors
        # Check if there are enough words left in the sequence to create a batch
        while True:

            if self.batch_in_file == self.x_file.shape[0]/self.batch_size:
                self.sequence_for_file = self.load_next_sequence()
                self.batch_in_file = 0

            start = self.batch_in_file*self.batch_size
            x_batch = np.array(self.x_file[start:start+self.batch_size, :])
            self.batch_in_file += 1

            y_batch = np.array(self.y_file[start:start+self.batch_size])
            y_batch = to_categorical(y_batch, num_classes=self.vocab_size)

            if self.Affect_LM:
                x_batch = x_batch.reshape(x_batch.shape[0], self.sliding_window_size, 1)

            yield x_batch, y_batch

    # Gets the unique words from the json file
    def get_unique_words_from_json_file(self):
        file_path = UNIQUE_WORD_FILE
        string_data = open(file_path).read()
        return string_data
'''
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
'''

'''

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

'''