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


#downloading the page for Ed-Sheeran

wbpg = requests.get('https://www.lyrics.com/artist/Ed-Sheeran/2342870')


# Downloading and saving the lyrics using BeautifulSoup

##BeautifulSoup object

web_soup = BeautifulSoup(wbpg.text, 'html.parser')
class_list_ed = web_soup.find_all('td', attrs = {'class': "tal qx"})

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

##downloading the lyrics of Ed Sheeran

#path_ed = './lyrics/Ed_Sheeran/beautifulsoup/'
#lyric_down(path_ed, class_list_ed)

# Adding another singer

#downloading the page for Taylor-Swift

wbpg_ts = requests.get('https://www.lyrics.com/artist/Taylor-Swift/816977')

#Using the BeautifulSoup object (Taylor-Swift)

ts_html = wbpg_ts.text
ts_soup = BeautifulSoup(ts_html, 'html.parser')

#Finding the class including the hyperlinks of lyrics (Taylor-Swift)

class_list_ts = ts_soup.find_all('td', attrs = {'class': "tal qx"})

#Downloading the lyrics Taylor

#path_ts = './lyrics/Taylor_Swift/beautifulsoup/'
#lyric_down(path_ad, class_list_ts)


# Text Preprocessing for each lyric and creating the corpus
 
# * Removing numbers, punctuations and new lines 
# * Normalisation: Stemming/Lemmatisation ---> I choose stemming (reasons: having two languages, faster and less computationally expensiv)

# Function for creating corpus

ps = PorterStemmer()

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


#Ed Sheeran corpus

path_ed = './lyrics/Ed_Sheeran/beautifulsoup/'
corpus_ed = corpus(path_ed)

# Taylor Swift corpus
path_ts = './lyrics/Taylor_Swift/beautifulsoup/'
corpus_ts = corpus(path_ts)

# Creating the matrix (vectorizing corpus) and then Normalizing using TF-IDF and finally creating a data frame for each artist

#Ed Sheeran 

vectorizer = CountVectorizer(stop_words="english")
matrix_ed = vectorizer.fit_transform(corpus_ed)

tf = TfidfTransformer()
transformed_ed = tf.fit_transform(matrix_ed)
tdf_ed = pd.DataFrame(transformed_ed.todense(), columns=vectorizer.get_feature_names_out())
tdf_ed['artist'] = 'Ed Sheeran'

# Taylor Swift

matrix_ts = vectorizer.fit_transform(corpus_ts)

tf = TfidfTransformer()
transformed_ts = tf.fit_transform(matrix_ts)
tdf_ts = pd.DataFrame(transformed_ts.todense(), columns=vectorizer.get_feature_names_out())
tdf_ts['artist'] = 'Taylor Swift'


## Stacking the data frames of two artists and filling NaNs with zero

df = pd.concat([tdf_ed,tdf_ts])
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

### Fitting the Naive Bayes model

m_nb = MultinomialNB(alpha=0.01)
m_nb.fit(x_train, y_train)


nb_tr = m_nb.score(x_train, y_train)
nb_ts = m_nb.score(x_test, y_test)

print(f""" Naive Bayes score for train data: {nb_tr}\n Naive Bayes score for test data: {nb_ts}""")

#### Probability of a song with the word 'yesterday' for each singer:


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

test_word = zerolistmaker(2698)
test_word = pd.DataFrame([test_word])

#print(m_nb.predict_proba(test_word).round(2))

#print(m_nb.predict(test_word))


### Fitting the Randomforest model

m_rf = RandomForestClassifier(n_estimators=60, 
                            max_depth=4, 
                            random_state=30)
m_rf.fit(x_train, y_train)


rf_tr = m_rf.score(x_train, y_train) 
rf_ts = m_rf.score(x_test, y_test)

print(f""" Randomforest score for train data: {rf_tr}\n Randomforest score for test data: {rf_ts}""")

###Visualization of results

data = [[lr_tr, nb_tr, rf_tr], [lr_ts, nb_ts, rf_ts]]
plt.rcParams['figure.figsize'] = (8, 6)
p = sns.heatmap(data, 
            cmap='BuPu', 
            annot=True, fmt='g',
            linewidths=0, linecolor='white',
            xticklabels= ['Logistic Regression', 'Naive Bayes', 'Random Forest'], 
            yticklabels=['Train acuracy', 'Test accuracy'], 
            )
p.set_title("Different models on 3 Features")

