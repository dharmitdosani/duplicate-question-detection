import numpy as np
import spacy
import h5py
from numpy import array
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential, Model, model_from_json
from keras.layers import Lambda, Merge
from keras.layers import Dense, Input, Lambda, BatchNormalization, Merge
from keras import backend as K


# load spacy module for getting sent vectors

nlp = spacy.load('en_core_web_lg')


# def function to calculate similarities btw outputs of the NN

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

# def loss function to be used by NN.


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# def accuracy function


def create_base_network(input_dim):
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(BatchNormalization())
    seq.add(Dense(128, activation='relu'))
    seq.add(BatchNormalization())
    seq.add(Dense(128, activation='relu'))
    seq.add(BatchNormalization())
    return seq


def acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)


# load dict

word2tfidf = np.load('word2tfidf_dict.npy').item()

# take input from user and create vector
while True:
    question1 = input("Enter a question: ")

    question2 = input("Enter another question: ")

    doc1 = nlp(question1)
    vec1 = np.zeros([len(doc1), 300])
    print('calculating word vector 1')
    for word in doc1:
        # word2vec
        vec = word.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word)]
        except:
            # print word
            idf = 0
        # compute final vec
        vec1 += vec * idf
    vec1 = vec1.mean(axis=0)

    doc2 = nlp(question2)
    vec2 = np.zeros([len(doc2), 300])

    print('calculating word vector 2')
    for word in doc2:
        # word2vec
        vec = word.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word)]
        except:
            # print word
            idf = 0
        # compute final vec
        vec2 += vec * idf
    vec2 = vec2.mean(axis=0)

    # assign optimizer

    rms = RMSprop()

    # load json and create model

    # json_file = open('final_nn_model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    input_dim = 300
    epochs = 1

    # define the inputs and pass the dimesions

    base_network = create_base_network(input_dim)

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # state how the outputs will be compared

    malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([processed_a, processed_b])

    # create the model with inputs and similarity parameter

    model = Model(inputs=[input_a, input_b], outputs=malstm_distance)

    # with open('final_nn_model.json', 'r') as json_file:
    #     loaded_model = model_from_json(json_file.read())

    # load weights into new model

    model.load_weights("final_nn_model.h5")
    print("Loaded model from disk")

    # compile the model with parameters

    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[acc])

    # convert to numpy arrays

    print('coverting to arrays')

    sent_vec1 = array(vec1)
    sent_vec2 = array(vec2)
    sent_vec1 = sent_vec1.reshape(1, 300)
    sent_vec2 = sent_vec2.reshape(1, 300)

    # predict similarity

    print('predict')

    pred = model.predict([sent_vec1, sent_vec2])

    print(pred[0][0], 'Type: ', type(pred))

    # # print prediction

    if pred[0][0] >= 0.35:
        print("The questions are duplicate.")
    else:
        print("The questions are different.")
