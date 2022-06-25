# import required packages
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from pickle import dump, load
from keras.preprocessing.text import Tokenizer

#Reading the reviews from files from the directory and saving them in the train dataframe along with polarity.
def loading_train_data(dir_list1):
    print("Loading the training dataset:")
    df_train = pd.DataFrame(columns=['Review','Polarity'])
    for folder in dir_list1:
        if folder == 'neg' or folder =='pos':
            file_list = os.listdir(basepath1+'/'+folder)
            for subfile in file_list:
                fdata = open(basepath1+'/'+folder+'/'+subfile, encoding="utf8")                 
                if folder == 'neg':
                    df_train = df_train.append({'Review': fdata.read(), 'Polarity': 0}, ignore_index=True)
                else:
                    df_train = df_train.append({'Review': fdata.read(), 'Polarity': 1}, ignore_index=True)
    return df_train

#Removing special characters and numbers from the train reviews using regex
def review_to_words(review, string = True, remove_stopwords=True):
    print("Removing special characters from training dataset:")
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
def remove_stopwords_train(df_train):
    print("Removing stop words from training dataset:")
    stopword = set(stopwords.words("english"))
    for i in range(0,25000) : 
        review = df_train.iloc[i,0] # review column : 0
        review = review.lower().split()
        words = [r for r in review if not r in stopword]
        clean_review = ' '.join(words)
        df_train.iloc[i,0] = clean_review
    return df_train

if __name__ == "__main__": 
    # 1. load your training data
    basepath1 = './data/aclImdb/train'
    dir_list1 = os.listdir(basepath1)

    #creating training dataframe
    df_train = loading_train_data(dir_list1)

    # Data Preprocessing
    # 1. Removing unwanted words to convert reviews to a sequence of words which are more useful to determine sentiment
    df_train = remove_stopwords_train(df_train)
    
    # 2. Get list of clean train reviews
    clean_train_reviews = review_to_words(df_train.Review)
    
    #Replacing these clean train data reviws list into dataframe again
    df_train = df_train.drop(columns='Review')
    df_train.insert(0,'Review', clean_train_reviews)
    
    # 3. Shuffling the data so that the neg and pos polarities are not in order and get mixed
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    print("Training data set is: ",df_train.head())
    print("\n\n")
        
    #converting the dataframe into review and polarity list
    Train_reviews_clean = df_train['Review'].tolist()
    y_train_polarity = df_train['Polarity'].tolist()
    # dumping this local variable so that it can be called in test file
    dump(Train_reviews_clean, open('data/Train_reviews_clean.pkl', 'wb'))
    
    #Tokenizer
    print("Tokenizing words")
    tok = keras.preprocessing.text.Tokenizer()
    tok.fit_on_texts(Train_reviews_clean)
    X_train_clean = tok.texts_to_sequences(Train_reviews_clean)
    
    #Padding
    print("Padding Sequence:")
    X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_clean,padding='post',maxlen=1000)
    
    #Splitting the train dataset into train and validation sets for training
    X_train, X_val, y_train, y_val = train_test_split(X_train_padded, y_train_polarity, test_size=0.3, random_state=42)
    
    
    # 2. Train your network
    # Make sure to print your training loss and accuracy within training to show progress
    # Make sure you print the final training accuracy
    print("Training Model")
    word_size = len(tok.word_index)+1

    model = keras.Sequential()
    model.add(keras.layers.Embedding(word_size, 16))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Conv1D(filters=16,kernel_size=2,padding='valid',activation='relu'))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    
    history = model.fit(tf.convert_to_tensor(X_train),tf.convert_to_tensor(y_train),epochs=20,validation_data=(tf.convert_to_tensor(X_val), tf.convert_to_tensor(y_val)),verbose=1,batch_size=512)
    
    print('The final training Validation accuracy is ',history.history['acc'][-1]*100)

    # 3. Save your model
    model.save("./models/Group44_NLP_model.h5")