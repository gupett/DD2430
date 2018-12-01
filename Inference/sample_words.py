from tensorflow.keras import models
from tensorflow.keras.models import load_model
import pickle
from numpy import array
import random
import bisect
import numpy as np

from Models.LM_model import LM_Model
from Models.base_line_affect_LM_model import base_line_affect_lm_model


def cumulative_distribution_function(probabilities):
    # floating point error
    total = sum(probabilities)
    cdf = []
    c_sum = 0.0
    for p in probabilities:
        c_sum += p
        cdf.append((c_sum / total))
    return cdf


def sample_index_from_distribution(probabilities):
    cdf = cumulative_distribution_function(probabilities)
    # Get a random number between 0 and 1
    x = random.random()
    # Get the index of where x can be inserted to still keep the list ordered
    index = bisect.bisect(cdf, x)
    return index


# Must send in a model with batch size 1, otherwise can not sample one word at a time
class sample_word:
    def __init__(self):
        # Load the tokenizer from file
        with open('../model/tokenizer/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer

        vocab_size = len(self.tokenizer.word_index) + 1

        #self.model = LM_Model(vocab_size, look_back=1, batch_size=1, stateful=True).model
        self.model = base_line_affect_lm_model(
            vocab_size, look_back_steps=1, batch_size=1)
        self.model.load_weights('../model/best_weights_2.hdf5')

    def sample_new_sequence(self, text_sample):
        self.model.reset_states()

        # Reset the memory cell and hidden node since a new sequence will be started
        encoded_sequence = self.tokenizer.texts_to_sequences([text_sample])[0]

        # Loop over all the word indexes in the list and predict the next word
        for word in encoded_sequence:
            encoded_word = array([word])
            word_prediction = self.model.predict(encoded_word, verbose=2)

        # Sample a word based on the probabilities of the words
        #sampled_index = sample_index_from_distribution(word_prediction[0])
        sampled_index = np.argmax(word_prediction[0])

        # Loop over the tokenizer and find the word corresponding to the sampled index
        sampled_word = ''
        for word, index in self.tokenizer.word_index.items():
            if sampled_index == index:
                sampled_word = word
                break

        if sampled_word == 'unk':
            ind = np.argpartition(word_prediction[0], -2)[-2:]
            best_ind = ind[0]
            if ind[0] == sampled_index:
                best_ind = ind[1]

            for word, index in self.tokenizer.word_index.items():
                if best_ind == index:
                    sampled_word = word
                    break

        return sampled_word

    def sample_next_word(self, init_word):
        # Encode the init word into a integer
        #file_content = init_word.lower()
        #file_content = init_word.replace('\n', ' ')

        encoded_word = array(self.tokenizer.texts_to_sequences([init_word])[0])

        encoded_word = array([encoded_word])

        # predict the probabilities for each word
        # print(encoded_word.shape)
        prediction = self.model.predict([encoded_word], verbose=0)
        # Sample a word based on the probabilities of the words

        sampled_index = np.argmax(prediction[0])
        #sampled_index = sample_index_from_distribution(prediction[0])

        # Loop over the tokenizer and find the word corresponding to the sampled index
        sampled_word = ''
        for word, index in self.tokenizer.word_index.items():
            if sampled_index == index:
                sampled_word = word
                # print(sampled_word)
                break

        if sampled_word == 'unk':
            #print('is unk')
            ind = np.argpartition(prediction[0], -2)[-2:]
            best_ind = ind[0]
            if ind[0] == sampled_index:
                best_ind = ind[1]

            for word, index in self.tokenizer.word_index.items():
                if best_ind == index:
                    sampled_word = word
                    break

        return sampled_word


if __name__ == '__main__':

    sentences = ['hi, who are you', 'I am the ']

    input_file = 'inference_input.txt'
    output_file = 'inference_output.txt'

    with open(input_file) as file:
        content = file.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    sampeler = sample_word()
    for word_seq in content:
        # print(sampeler.tokenizer.word_index['pre-recorded'])
        #word = 'The president is not in his office'
        # Initialize a sampeling sequence
        next_word = sampeler.sample_new_sequence(word_seq)

        # Sample based on the words returned from the model
        sample_size = 5
        sampled_sentence = word_seq + ' ' + next_word
        for i in range(sample_size):
            #print('Next word: {}'.format(next_word))
            next_word = sampeler.sample_next_word(next_word)
            sampled_sentence += ' ' + next_word
        with open(output_file, "a") as file:
            file.write(sampled_sentence + '\n')
        print(sampled_sentence)

'latter interpretation delivered patients doctor brewery procedure elements rise expect indifference'
