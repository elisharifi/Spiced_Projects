from pymongo import MongoClient
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import inspect
#import re
import logging
import pandas as pd
import time


#accessing mongodb
client = MongoClient("mongodb")
db_mongo = client.tweet_db

time.sleep(20)  # seconds

#accessing postgres and creating a table
engine = create_engine("postgresql://postgres:postgres@project-postgresdb-1:5432/postgr_db")
#engine.execute("""CREATE TABLE IN NOT EXISTS tweet_table(
#    tweet_text VARCHAR(500), 
#    sentiment NUMERIC);""")

# Extract data: get tweets from mongodb
def extract():
    """This function creates a list of twwets' texts. 
    Then it converts it to a dataframe.
    Finally it returns this dataframe"""
    tweets = list(db_mongo.tweets.distinct("tweet"))
    df = pd.DataFrame(tweets, columns=["tweet_text"])
    return df


#Transform data: sentiment analysis
def transform_data(df):
    """This function gets a dataframe with one column of strings.
    First it performs the sentimental analysis on it.
    Then calculates the scores and returns the compound score"""
    s = SentimentIntensityAnalyzer()
    scores = df["tweet_text"].apply(s.polarity_scores).apply(pd.Series)
    df["compound_score"]=scores["compound"]
    return df

#Load data: write to postgres
def load(df):
    """This function gets the dataframe with two columns (tweet_text and compound_score)
    and dumps it to sql"""
    df.name = 'data_table'
    df.to_sql(df.name, engine)


tweets_df = extract()
df_with_scores = transform_data(tweets_df)
load(df_with_scores)

#logging.critical("extraction step is done")

# Testing to see if we have the dataframe in sql
#inspect(engine).get_table_names()
#inspect(engine).get_columns("compound_score")