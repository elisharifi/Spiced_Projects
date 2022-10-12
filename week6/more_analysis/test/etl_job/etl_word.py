from pymongo import MongoClient
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import inspect
import re
import logging
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer

#accessing mongodb
client = MongoClient("mongodb")
db_mongo = client.tweet_db_test

#time.sleep(40)  # seconds

#accessing postgres and creating a table
engine = create_engine("postgresql://postgres:postgres@test-postgresdb-1:5432/postgr_db")
#engine.execute("""CREATE TABLE IN NOT EXISTS tweet_table(
#    tweet_text VARCHAR(500), 
#    sentiment NUMERIC);""")

# Extract data: get tweets from mongodb
def extract():
    """This function creates a list of tweets' texts. 
    Then it converts it to a dataframe.
    Finally it returns this dataframe.
    In between, it will analyzes the text of the all of tweets and finds 10 top frequents words in it and prints them."""
    tweets = list(db_mongo.tweets.distinct("tweet"))
    df = pd.DataFrame(tweets, columns=["tweet_text"])
    #Analysis of the extracted data to find 10 top frequent words of the tweets' topic
    tweet_str = ""
    for tweet in tweets:
        tweet_str = tweet_str + tweet
    tweet_str = re.sub(r"@[A-Za-z0-9]+", '', tweet_str)
    tweet_str = re.sub(r"https?:\/\/\S+", '', tweet_str)
    vectorizer = CountVectorizer(stop_words="english")
    matrix_word = vectorizer.fit_transform(tweet_str)
    df_words = pd.DataFrame(matrix_word.todense(), columns=vectorizer.get_feature_names_out())
    df_words = df_words.sum(axis=0)
    df_words.sort_values(ascending=False, inplace=True)
    '''
    word_list = [word for word in tweet_str.split()]
    word_count = {}
    for word in word_list:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    iggnore_words = ["in", "the", "for", "a", "to", "of", "and"]
    clean_words = word_count
    for key in word_count:
        if key in iggnore_words:
            clean_words.pop(key, None)
    df_words = pd.DataFrame(word_count.items(), columns=["words", "counts"])
    df_words.sort_values("counts", ascending=False, inplace=True)
    '''
    print(df_words.head(10))
    logging.critical("extraction step is done")
    return df


#Transform data: sentiment analysis
def transform_data(df):
    """This function gets a dataframe with one column of strings.
    First it performs the sentimental analysis on it.
    Then calculates the scores and returns the compound score"""
    s = SentimentIntensityAnalyzer()
    scores = df["tweet_text"].apply(s.polarity_scores).apply(pd.Series)
    df["compound_score"]=scores["compound"]
    logging.critical("transforming step is done")
    print(f"""The average compound score is {df["compound_score"].mean()}.""")
    return df

#Load data: write to postgres
def load(df):
    """This function gets the dataframe with two columns (tweet_text and compound_score)
    and dumps it to sql"""
    df.name = 'data_table'
    df.to_sql(df.name, engine)
    logging.critical("loading step is done")
tweets_df = extract()
df_with_scores = transform_data(tweets_df)
load(df_with_scores)
