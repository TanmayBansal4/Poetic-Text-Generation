import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shake.txt' , 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
#lower here is used to get higher efficiency in the neural network

text = text[:]

characters = sorted(set(text))
#set is to get all the unique characters and sorted is to sort everything

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))
#these dictionaries here are to get the characters into numerical format so we can use numpy on these characters
#enumerate is used to assign a number to each character in the set

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_characters = []

for i in range (0 , len(text) - SEQ_LENGTH , STEP_SIZE):
    sentences.append(text[i : i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences),SEQ_LENGTH , len(characters)) , dtype = bool)
y = np.zeros((len(sentences) , len(characters)) , dtype = bool)

for i , sentence in enumerate(sentences):
    for t , character in enumerate(sentence):
        x[i,t,char_to_index[character]] = 1
    y[i , char_to_index[next_characters[i]]] = 1

model = Sequential()
model.add(LSTM(128 , input_shape = (SEQ_LENGTH , len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop())
model.fit(x , y , batch_size = 256 , epochs = 4)

# model.save('./model/textgen.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print(generate_text(300 , 0.2))
# the common sort
# with pay and thanks, and the say the carse
# the carse the conder sore to me the carse
# the beart the browntt the are the praines
# the beat the sare the carsely the with and there
# the stare the carnest of the cander and theer have
# the grown the sare to the have to the cand
# the cand the carst the carning to the ere
# the carse th


# print(generate_text(300 , 0.8))
# le clarence, welcome unto warwick;
# and wher your have trauke the erather thy pault
# wirnth you have of to herd tome a beattes
# and year sirses on your, a manathare;
# should what uperidst the nort, and shall carnios?
#
# lergord:
# hord, nor theme encontsted, my; lord filat
# and he men deat to thier searw.
#
# king rincantio:
# i thoull and son'lled and