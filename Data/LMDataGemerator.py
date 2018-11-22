from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from os import listdir
from os.path import isfile, join
import numpy as np
import gc
import pickle
from Data.analyse import affection_context

FILE_EXTENSION = '../Data/Data/training/'
FILES = [join(FILE_EXTENSION, file_name) for file_name in listdir(FILE_EXTENSION) if isfile(join(FILE_EXTENSION, file_name)) and file_name != '.DS_Store']
UNIQUE_WORD_FILE = '../Data/Data/unique_words.json'

# Function for finding the largest number less than K+1 divisible by X
def largest(X, K):
    # returning ans
    return (K - (K % X))

class dataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size=20, sliding_window_size=20, shuffle=False, Affect_LM=False, existing_tokenizer=False):
        self.batch_size = batch_size
        self.sliding_window_size = sliding_window_size
        self.shuffle = shuffle

        # Get all the unique words for every file in the training data set
        self.unique_words = self.get_unique_words_from_json_file()

        if existing_tokenizer:
            # Load the tokenizer from file
            with open('../model/tokenizer/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            self.tokenizer = tokenizer
        else:
            # Init a tokenizer which will translate words in to integers
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts([self.unique_words])
        # Reversed tokenizer for going from index go word
        self.reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))
        # Vocabulary size
        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.batch_per_epoch = self.nr_batch_per_epoch()

        self.current_file_nr = -1
        self.x_file, self.y_file = self.load_next_sequence()
        # To keep track where in the current sequence the last batch was taken from
        self.batch_in_file = 0

        self.affect_context = affection_context()
        self.affect_categories = self.affect_context.affect_categories

    def nr_batch_per_epoch(self):
        batch_per_epoch = 0
        for file_path in FILES:
            #print(file_path)
            with open(file_path) as file:
                file_content = file.read()

            file_content = file_content.lower()
            file_content = file_content.replace('\n', ' ')

            # Get an encoding of the text
            file_encoded_sequence = self.tokenizer.texts_to_sequences([file_content])[0]

            # the // operator gives int values
            batch_per_epoch += ((len(file_encoded_sequence)-(self.sliding_window_size+1))//(self.batch_size))
        return batch_per_epoch

    def affect_for_batch(self, x_batch):
        affect_batch = np.zeros((x_batch.shape[0], self.affect_categories))
        for i, row in enumerate(x_batch):
            context = []
            for index in row:
                context.append(self.reverse_word_map[index])
            affect_batch[i,:] = self.affect_context.binary_affection_vector_for_context(context)
            #print('affect size: {}'.format(affect_batch.shape))

        return affect_batch

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
                gc.collect()

            start = self.batch_in_file*self.batch_size
            x_batch = np.array(self.x_file[start:start+self.batch_size, :])
            #print('x_batch size 1: {}'.format(x_batch.shape))
            affect_batch = self.affect_for_batch(x_batch)
            # Reshaping for LSTM layer input
            x_batch = x_batch.reshape(x_batch.shape[0], self.sliding_window_size, 1)
            #print('x_batch size: {}'.format(x_batch.shape))
            self.batch_in_file += 1

            y_batch = np.array(self.y_file[start:start+self.batch_size])
            y_batch = to_categorical(y_batch, num_classes=self.vocab_size)

            #yield {'word_input': x_batch, 'affect_input': affect_batch}, y_batch
            yield {'input_1': x_batch, 'input_2': affect_batch}, y_batch
            #yield x_batch, y_batch

    # Gets the unique words from the json file
    def get_unique_words_from_json_file(self):
        file_path = UNIQUE_WORD_FILE
        string_data = open(file_path).read()
        return string_data