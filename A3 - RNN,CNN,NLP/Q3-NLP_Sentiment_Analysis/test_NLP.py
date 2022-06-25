# import required packages
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from pickle import load
from keras.preprocessing.text import Tokenizer

#Reading the reviews from files from the directory and saving them in the test dataframe along with polarity.
def loading_test_data(dir_list2):
    print("Loading the testing dataset:")
    df_test = pd.DataFrame(columns=['Review','Polarity'])
    for folder in dir_list2:
        if folder == 'neg' or folder =='pos':
            file_list = os.listdir(basepath2+'/'+folder)
            for subfile in file_list:
                fdata = open(basepath2+'/'+folder+'/'+subfile, encoding="utf8")                 
                if folder == 'neg':
                    df_test = df_test.append({'Review': fdata.read(), 'Polarity': 0}, ignore_index=True)
                else:
                    df_test = df_test.append({'Review': fdata.read(), 'Polarity': 1}, ignore_index=True)
    return df_test

#Removing special characters and numbers from the test reviews using regex
def review_to_words(review, string = True, remove_stopwords=True):
    print("Removing special characters from testing Data:")
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\*)|(\d+)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    NON_LETTERS = re.compile("[^a-zA-Z]")
    NO_SPACE = ""
    SPACE = " "
    review = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in review]
    review = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in review]
    review = [NON_LETTERS.sub(SPACE, line) for line in review]  
    return review

#Removing unwanted words to convert a document to a sequence of words which are more useful to determine sentiment
def remove_stopwords_test(df_test):
    print("Removing stop words from testing Data:")
    stopword = set(stopwords.words("english"))
    for i in range(0,25000) : 
        review = df_test.iloc[i,0] # review column : 0
        review = review.lower().split()
        words = [r for r in review if not r in stopword]
        clean_review = ' '.join(words)
        df_test.iloc[i,0] = clean_review
    return df_test


if __name__ == "__main__": 
    # 1. Load your saved model
    model=tf.keras.models.load_model('./models/Group44_NLP_model.h5')

    # 2. Load your testing data
    basepath2 = './data/aclImdb/test'
    dir_list2 = os.listdir(basepath2)

    #creating testing dataframe
    df_test = loading_test_data(dir_list2)
    
    #Data Preprocessing
    # 2.1. Removing unwanted words to convert reviews to a sequence of words which are more useful to determine sentiment
    df_test = remove_stopwords_test(df_test)
    
    # 2.2. Get list of clean test reviews
    clean_test_reviews = review_to_words(df_test.Review)
    
    #Replacing these clean test data reviws list into dataframe again
    df_test = df_test.drop(columns='Review')
    df_test.insert(0,'Review', clean_test_reviews)
    
    # 2.3. Shuffling the data so that the neg and pos polarities are not in order and get mixed
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    print("Training data set is: ", df_test.head())
    print("\n\n")
    
    #converting the dataframe into review and polarity list
    Test_reviews_clean = df_test['Review'].tolist()
    y_test = df_test['Polarity'].tolist()
    
    #Tokenizer
    print("Tokenizing words")
    Train_reviews_clean = load(open('data/Train_reviews_clean.pkl', 'rb'))
    tok = keras.preprocessing.text.Tokenizer()
    tok.fit_on_texts(Train_reviews_clean) 
    X_test_clean = tok.texts_to_sequences(Test_reviews_clean)
    
    #Padding
    print("Padding Sequence:")
    X_test = keras.preprocessing.sequence.pad_sequences(X_test_clean,padding='post',maxlen=1000)    
   
    # 3. Run prediction on the test data and print the test accuracy
    scores = model.evaluate(tf.convert_to_tensor(X_test),tf.convert_to_tensor(y_test))
    test_accuracy = scores[1]
    print('accuracy on testing dataset is:',test_accuracy*100)