from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np
import json

from os import listdir
from os.path import isfile, join

FILE_EXTENSION = './Data/parts_small/'
FILES = [join(FILE_EXTENSION, file_name) for file_name in listdir(FILE_EXTENSION) if isfile(join(FILE_EXTENSION, file_name))]




class generate_training_data:

    def __init__(self, sliding_window_size):
        # variables for initializing the training data
        self.sliding_window_size = sliding_window_size

        # Get all the unique words for every file in the training data set
        self.unique_words = self.get_unique_words_from_json_file()
        #self.unique_words = self.get_file_content()
        # Init a tokenizer which will translate words in to integers
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([self.unique_words])
        print(self.tokenizer.word_index)
        # Vocabulary size
        self.vocab_size = len(self.tokenizer.word_index) + 1

        # Stores all the batch data of one file
        #self.training_data = self.read_file_for_training_data()
        self.batch = 0
        self.nr_batch_files = len(FILES)


    '''
    # Gets the unique words for all the files so that a tokenizer encoding can be generated for every word
    def get_unique_words_for_files(self):
        unique_words = set()
        for file_name in FILES:
            with open(file_name) as file:
                file_content = file.read()

            file_content = file_content.lower()
            file_content = file_content.replace('\n', ' ')
            words = [word for word in file_content.split(' ') if len(word) > 0]

            for word in words:

                if word not in unique_words: unique_words.add(word)

            return list(unique_words)
    '''

    '''
    # Gets the unique words from the json file
    def get_file_content(self):
        file_path = './treat-data/BNC/training_files/A.txt'
        string_data = open(file_path).read()
        return string_data

    # Gets the unique words from the json file
    def get_unique_words_from_json_file(self):
        file_path = './treat-data/BNC/small-example/unique_words.json'
        json_data = open(file_path).read()
        data = json.loads(json_data)
        string_data = ''
        for s in data:
            string_data += s + ' '
        #print(data)
        return string_data
    
    '''


    # Gets the unique words from the json file
    def get_unique_words_from_json_file(self):
        file_path = './treat-data/BNC/small-example/unique_words.txt'
        string_data = open(file_path).read()
        return string_data

    # Create training batches from a full file
    def read_file_for_training_data(self, file_path):
        #text_file = './treat-data/BNC/small-example/A00.txt'
        with open(file_path) as file:
            file_content = file.read()

        file_content = file_content.lower()
        file_content = file_content.replace('\n', ' ')

        file_encoded = self.tokenizer.texts_to_sequences([file_content])[0]

        sequences = []
        for i in range(self.sliding_window_size+1, len(file_encoded)):
            sequence = file_encoded[i-(self.sliding_window_size+1):i]
            sequences.append(sequence)

        sequences = np.asarray(sequences)

        X = np.array(sequences[:,0:self.sliding_window_size])
        y = np.array(sequences[:, -1])

        y = to_categorical(y, num_classes=self.vocab_size)

        return X, y

    # Function to be called from the main training function for collecting new training data
    # Fix function so that it reads in a new file of training data once this function is called
    def get_next_batch(self):
        file_path = FILES[self.batch]
        self.batch += 1
        self.batch = self.batch % len(FILES)

        print(file_path)
        next_X_batch, next_Y_batch = self.read_file_for_training_data(file_path)

        return next_X_batch, next_Y_batch

if __name__ == '__main__':
    trining_data = generate_training_data(sliding_window_size=20)
    print(trining_data.get_unique_words_from_json_file())
    #print(trining_data.get_unique_words_for_files())
    #X, y = trining_data.training_data
    #print(X[5])
    #print(y[0:5])

    #print(trining_data.tokenizer.word_index['the'])
    #print(trining_data.tokenizer.word_index['defence'])
    #print(trining_data.tokenizer.word_index['so'])


#[[828, 1519, 1390, 923, 1343, 1156, 1516, 879, 563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326], [1519, 1390, 923, 1343, 1156, 1516, 879, 563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447], [1390, 923, 1343, 1156, 1516, 879, 563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447, 340], [923, 1343, 1156, 1516, 879, 563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447, 340, 122], [1343, 1156, 1516, 879, 563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447, 340, 122, 1046], [1156, 1516, 879, 563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447, 340, 122, 1046, 434], [1516, 879, 563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447, 340, 122, 1046, 434, 1266], [879, 563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447, 340, 122, 1046, 434, 1266, 756], [563, 1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447, 340, 122, 1046, 434, 1266, 756, 513], [1156, 849, 466, 1619, 230, 1577, 923, 849, 829, 849, 326, 447, 340, 122, 1046, 434, 1266, 756, 513, 1083]]
#['the', "body's", 'defence', 'system', 'so', 'that', 'it', 'cannot', 'fight', 'infection', 'it', 'is', 'not', 'transmitted', 'from', 'giving', 'blood/mosquito', 'bites/to
