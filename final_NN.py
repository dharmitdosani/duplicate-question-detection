import pickle
import numpy as np
from keras.models import load_model
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from numpy import array
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, BatchNormalization, Merge
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import h5py

# run the new_code.py first to train sentence vectors and create the sentence vector object files
# load the saved files containing vectors of the sentences

q1sentenceVectorFile = open('sentence_vectors_Q_1', 'rb')
q2sentenceVectorFile = open('sentence_vectors_Q_2', 'rb')

sentence_vectors_q_1 = pickle.load(q1sentenceVectorFile)
sentence_vectors_q_2 = pickle.load(q2sentenceVectorFile)

# load the saved dataframe

df = pd.read_pickle('data_frame_dump')

# convert the files to numpy arrays

sent_vec_q1 = array(sentence_vectors_q_1)
sent_vec_q2 = array(sentence_vectors_q_2)

# extract is_duplicate column as labels and convert to integers

df['is_duplicate'] = df['is_duplicate'].apply(pd.to_numeric)

df['is_duplicate'].replace(to_replace=1, value=-1, inplace=True)
df['is_duplicate'].replace(to_replace=0, value=1, inplace=True)
df['is_duplicate'].replace(to_replace=-1, value=0, inplace=True)

labels = df['is_duplicate'].values
labels = labels.reshape(len(labels), 1)

# create structure to store 2 vectors for a particular label adjacently

train_data = []

for q1, q2 in zip(sent_vec_q1, sent_vec_q2):
    train_data.append([q1, q2])

# get appropriate train and test data using sklearn function

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.33, random_state=42)
X_train = array(X_train)
X_test = array(X_test)


# Now Starts the Siamese Network

# def function to calculate similarities btw outputs of the NN

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

# def loss function to be used by NN.


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# create NN structure


def create_base_network(input_dim):
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(BatchNormalization())
    seq.add(Dense(128, activation='relu'))
    seq.add(BatchNormalization())
    seq.add(Dense(128, activation='relu'))
    seq.add(BatchNormalization())
    return seq

# def accuracy function


def acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)


def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()


# assign the dimensions of the inputs and no of iterations

input_dim = 300
epochs = 20

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

# assign optimizer

rms = RMSprop()

# compile the model with parameters

model.compile(loss=contrastive_loss, optimizer=rms, metrics=[acc])

# fit the model on relevant data

model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=300, verbose=2, epochs=epochs)
# pred the values

# pred_test = model.predict([X_test[:,0],X_test[:,1]])
# print (X_test[:,0].shape)
# pred_train = model.predict([X_train[:,0],X_train[:,1]])
# print (pred_test[:10])
# compute accuracy

scores_test = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
print("Test: %s: %.2f%%" % (model.metrics_names[1], scores_test[1] * 100))

scores_train = model.evaluate([X_train[:, 0], X_train[:, 1]], y_train)
print("Train: %s: %.2f%%" % (model.metrics_names[1], scores_train[1] * 100))

# acc_test  = compute_accuracy(pred_test,y_test)
# acc_train = compute_accuracy(pred_train,y_train)

# # #print accuracy

# print('* Accuracy on test set: %0.2f%%' % (100 * acc_test))
# print('* Accuracy on train set: %0.2f%%' % (100 * acc_train))

# save model
# serialize model to JSON

# model_json = model.to_json()
# with open("final_nn_model.json", "w") as json_file:
#     json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("final_nn_model.h5")
print("Saved model to disk")
