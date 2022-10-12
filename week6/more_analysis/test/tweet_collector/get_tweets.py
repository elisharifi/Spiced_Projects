import tweepy
from credentials import BEARER_TOKEN
import logging
from pymongo import MongoClient

### Authentication
client = tweepy.Client( bearer_token = BEARER_TOKEN)
if client:
	logging.critical('\n Athentication is ok')
else:
	logging.critical('\n Athentication went wrong')

#### tweets about a specific trending topic from different users:
search_query = "women rights lang:en -is:retweet"

cursor=tweepy.Paginator(method=client.search_recent_tweets, query=search_query, tweet_fields=['id', 'text', 'author_id', 'public_metrics']).flatten(limit=500)
#for tweet in cursor:
    #print(tweet.text+"\n")
'''   
file = open('womenrights_tweets_paginator.txt', mode = 'a')
for tweet in cursor:
    #print(f"At{tweet.created_at} this tweet was written: {tweet.text}")
    file.write(f"\n\n{tweet.text}")
    print("unicorn") 
file.close() 
'''
###################################################################################    
client_m = MongoClient("mongodb")
tweet_db_test = client_m.tweet_db_test
for tweet in cursor:
    #print("we are inside the loop")
    tweet_db_test.tweets.insert_one({"tweet":tweet.text, "tweet_time":tweet.created_at})