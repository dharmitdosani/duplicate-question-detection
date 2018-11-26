import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np
import pickle
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# CONSTANTS
DATASET_DIRECTORY = './data/QuoraDataset.csv'
WORD_SCORE_DICTIONARY_FILE = './data/word2tfidf_dict.npy'

question1 = 'question1'
question2 = 'question2'
count1 = 'count1'
count2 = 'count2'
questionId1 = 'qid1'
questionId2 = 'qid2'


def generate_sentence_score_vector(sentence):
    doc = nlp(sentence)
    dimension_count = doc[0].vector.shape[0]  # gets the number of dimensions returned by the nlp model
    weighted_sum = np.zeros([len(doc), dimension_count])  # word_count x dimension_count
    for word in doc:
        word_vector = word.vector
        try:
            # check if the word is present in the dictionary
            idf = word_score_dict[str(word)]
        except:
            # word not present, set the score to 0
            idf = 0
        weighted_sum += word_vector * idf
    sentence_vector = weighted_sum.mean(axis=0)
    return sentence_vector


nlp = spacy.load('en_core_web_lg')

# load data frame of the dataset

df = pd.read_csv(DATASET_DIRECTORY, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')


# delete the columns of qid1 qid2
df.drop(questionId1, axis=1, inplace=True)
df.drop(questionId2, axis=1, inplace=True)

# count characters in each question

df[count1] = df[question1].str.len()
df[count2] = df[question2].str.len()

# remove questions which are too long or too short

df.drop(df[df[count1] < 10].index, inplace=True)
df.drop(df[df[count2] < 10].index, inplace=True)
df.drop(df[df[count1] > 550].index, inplace=True)
df.drop(df[df[count2] > 550].index, inplace=True)

# drop all rows with null values in any of the cells

df = df.dropna()

# generate corpus by Q1 list + Q2 list

question1_list = df[question1].tolist()
question2_list = df[question2].tolist()
questions = question1_list + question2_list

tfidfVectorizer = TfidfVectorizer(lowercase=False, )
tfidfVectorizer.fit_transform(questions)

# dict key:word and value:tf-idf score
word_score_dict = dict(zip(tfidfVectorizer.get_feature_names(), tfidfVectorizer.idf_))

np.save(WORD_SCORE_DICTIONARY_FILE, word_score_dict)


question1_sentence_vectors = []
for question in question1_list:
    sentence_vector = generate_sentence_score_vector(question)
    question1_sentence_vectors.append(sentence_vector)


question2_sentence_vectors = []
for question in question2_list:
    sentence_vector = generate_sentence_score_vector(question)
    question2_sentence_vectors.append(sentence_vector)

print('Saving sentence vectors')
q1sentenceVectorFile = open('sentence_vectors_Q_1', 'wb')
q2sentenceVectorFile = open('sentence_vectors_Q_2', 'wb')

pickle.dump(question1_sentence_vectors, q1sentenceVectorFile, protocol=2)
pickle.dump(question2_sentence_vectors, q2sentenceVectorFile, protocol=2)

q1sentenceVectorFile.close()
q2sentenceVectorFile.close()


df.to_pickle('data_frame_dump')
