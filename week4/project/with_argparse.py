import requests
import re 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sys
import argparse

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB



parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="A program to predict the artist from a piece of text(word). Singers are Ed Sheeran and Taylor Swift. ") # initialization
args = parser.parse_args()
ps = PorterStemmer()

## Function for downloading the webpage of a singer

def download_web(web_page):
    wbpg = requests.get(web_page)
    return wbpg

## Function for downloading the lyrics of a singer

def lyric_down(path,class_list, n = 101):
    
    """This function download the lyrics of a singer from the given class 
    and saves it into the given path"""
    
    for index , element in enumerate(class_list):
        if index < n:
            line = element.find_all('a')[0]
            h_link = re.findall(pattern ='<a href="(.+)">(.+)</a>', string = str(line))
            link = 'https://www.lyrics.com' + h_link[0][0]
            song = requests.get(link)
            song_html = song.text

            lyric_soup = BeautifulSoup(song_html, 'html.parser')
            lyric_text = lyric_soup.find_all('pre', attrs = {'class': "lyric-body"})

            file_name = path + 'song_num_' + str(index) + '.txt'
            file = open(file_name, 'w')
            file.write(lyric_text[0].text)
            file.close()
            
## Function for creating corpus for each singer            

def corpus(path, n= 101):
    """This function reads each text file (in total 101 files), cleans it by removing 
    numbers, new lines and punctuations except apostrophe. Then it will stemm the words 
    and finally creats a corpus of cleaned stemmed lyrics. """
    
    total_stemmed = []
    for i in range (n):
        stemmed_text = []
        file_name = path + 'song_num_' + str(i) + '.txt'
        file = open(file_name, "r")
        text = file.read()

        text_clean1 = re.sub(r"[^\w\s']", '', text)      #rmv punctuations exc apostrophe 
        text_clean2= re.sub(r'[\n]', ' ', text_clean1)   #rmv new lines
        text_clean3 = re.sub(r'[0-9]+', '', text_clean2)  #rmv numbers

        words = word_tokenize(text_clean3)                #spliting each song into words
        for word in words:                                # stemming words
            new_word = ps.stem(word)                 
            stemmed_text.append(new_word)           #appending stemmed words to a list
#             print(stemmed_text)

        total_stemmed.append(stemmed_text)       #appending stemmed songs to a list
#         print(total_stemmed)
        file.close()


    corpus=[0]*n                                #combining stemmed words of each song
    for index, song in enumerate(total_stemmed):   #creating corpus of songs (cleaned words)
        corpus[index] = ' '.join(total_stemmed[index])    

    return corpus


            
#positional arguments:
parser.add_argument('web_page1', help= "The webpage address of a singer that has hyperlinks of her/his songs")
parser.add_argument('web_page2', help= "The webpage address of the second singer that has hyperlinks of her/his songs")


#downloading the page for first singer

user_web_page1 = args.web_page1
user_web1 = download_web(user_web_page1)

##BeautifulSoup object

web_soup = BeautifulSoup(user_web1.text, 'html.parser')
class_list_ed = web_soup.find_all('td', attrs = {'class': "tal qx"})

#downloading the page for second singer

user_web_page2 = args.web_page2
user_web2 = download_web(user_web_page2)

#BeautifulSoup object for second singer

ts_html = user_web2.text
ts_soup = BeautifulSoup(ts_html, 'html.parser')
class_list_ts = ts_soup.find_all('td', attrs = {'class': "tal qx"})

# Creating corpus for first singer

corpus_1 = corpus(user_web_page1)

# Creating corpus for second singer

corpus_2 = corpus(user_web_page2)

# Creating the matrix (vectorizing corpus) and then Normalizing using TF-IDF and finally creating a data frame for each artist

#first singer

vectorizer = CountVectorizer(stop_words="english")
matrix_1 = vectorizer.fit_transform(corpus_1)

tf = TfidfTransformer()
transformed_1 = tf.fit_transform(matrix_1)
tdf_1 = pd.DataFrame(transformed_1.todense(), columns=vectorizer.get_feature_names_out())
tdf_1['artist'] = 'first singer'

#second singer

vectorizer = CountVectorizer(stop_words="english")
matrix_2 = vectorizer.fit_transform(corpus_2)

tf = TfidfTransformer()
transformed_2 = tf.fit_transform(matrix_2)
tdf_2 = pd.DataFrame(transformed_2.todense(), columns=vectorizer.get_feature_names_out())
tdf_2['artist'] = 'second singer'

## Stacking the data frames of two artists and filling NaNs with zero

df = pd.concat([tdf_1,tdf_2])
df = df.fillna(0)

### Fitting a  classification model (LogisticRegression)

X = df.drop(['artist'], axis = 1)
y = df['artist']

df.reset_index()

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=40)

m_lr = LogisticRegression()
m_lr.fit(x_train, y_train)


lr_tr = m_lr.score(x_train, y_train)
lr_ts = m_lr.score(x_test, y_test)

print(f""" LogisticRegression score for train data: {lr_tr}\n LogisticRegression score for test data: {lr_ts}""")

