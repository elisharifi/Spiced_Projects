#####################################################################################
#1. Loading Data and Converting to Wide Format
#####################################################################################

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import pickle

df = pd.read_csv('./data/ml-latest-small/ratings.csv', index_col=0)
# print(df.head())
# print(df.info())

df=pd.pivot_table(df,
                  index='userId',
                  columns='movieId',
                  values='rating')
############################################################################################
#2. Simple Recommender¶
############################################################################################

#2.1  Calculate the average rating for each movie in the dataset
average_rating = df.mean()
# print(average_rating)

#2.2  Filter out movies that have been watched by less than 20 users
df_filter = df.loc[:, df.count()>20]
# print(df_filter.head())

#2.3  Recommend the top ten movies that a user has not seen yet
def top_ten(df):
    '''
    This function first creates a dataframe named 'top' with movie ids and their average rates in a descending
    order.
    
    Then it creates a dictionary named 'recom' with users as keys and each user has a list of 10 unseen movies 
    which have the highest rate as the value.
    
    Then it converts this dictionary to a dataframe and returns it.
    '''
    #creating top dataframe
    top = df.mean().sort_values(ascending = False)
    top = top.reset_index()
    top.columns = ['movieId','av_rate']
    
    #creating recom dictionary
    recom = {}
    for user in df.index:
        rec_lst = []
        for movie in top['movieId']:
            if pd.isnull(df.loc[user, movie]):
                rec_lst.append(movie)
            if len(rec_lst)== 10:
                break
        recom[user]= rec_lst
#     #converting recom to dataframe
    recom = pd.DataFrame.from_dict(recom, orient='index')
    return recom

#Testing
recommend = top_ten(df_filter)
# print(recommend.head())

#2.4  Write a function recommend_popular(query, ratings, k=10) that gets a user query of rated movie-ids and the ratings table as input. It returns a list of k movie-ids.

#query: min number of people who have watched the movies. For example recommend_popular(100, df) will return 
#       10 movieids which at least 100 people have watched them.     
def recommend_frequent(query, df, k=10):
    '''
    This function first creates a dataframe named 'freq_watched' with movie ids and how many times they have been 
    wathched in a descending order.
    Then it creates a list of k number of movieIds which have been watched for at least query times and returns it.
    '''
    #creating freq_watched dataframe
    freq_watched = df.count().sort_values(ascending = False)
    freq_watched = freq_watched.reset_index()
    freq_watched.columns = ['movieId','freq']
    
    if query > freq_watched['freq'].max():
        print(f'There is no movie that minimum {query} people have watched it')
        return
    recom = []
    for i, movie in enumerate(freq_watched['movieId']):
        if freq_watched.loc[i, 'freq'] >= query:
            recom.append(movie)
        if len(recom) == k:
            break
    return recom

#Testing
test1 = recommend_frequent(200, df_filter)
# print(test1)
test2 = recommend_frequent(100, df_filter, 5)
# print(test2)
test3 = recommend_frequent(400, df_filter)
# print(test3)

#2.5  The user query is a python dictionary that looks like this: {12: 5, 234: 1, 235: 4.5}.
# query: Here it is a dictionary of number of people with specific rates. For example: 
#        query = {12:5, 234:1, 235:4.5} means that the user wants a list of movies that minimum 12 people gave
#        rate 5 to it, and  minimum 234 people gave rate 1 and minimum 235 people gave rate 4.5

    
def recommend_popular(query, df, k=10):
    '''
    This function first creates a dataframe named 'freq_rates' with movie ids and how many times they have been 
    rated(count number for each possible rate).
    Then it creates a list of k number of movieIds which have been watched for at least query times and returns it.
    '''
    #creating freq_rates dataframe
    ratings_long=pd.melt(df.reset_index(),
                     id_vars='userId',
                     var_name='movieId',
                     value_name='ratings')
    freq_rates = ratings_long.groupby(["movieId", "ratings"]).count()
    freq_rates = freq_rates.rename(columns={'userId': 'count'})
#     print(freq_rates.head())

    #creating the list of recommendations that matchs the query
    recom = []
    for key in query:
        rate = query[key]
        for movieId in df.columns:
            if (movieId, rate) in freq_rates.index:
                if freq_rates.loc[(movieId, rate),'count']>= key:
                    recom.append(movieId)

    recom = list(set([movieId for movieId in recom if recom.count(movieId) > 2]))
    
    #setting how many recommendation should be returned(k)
    recom.sort()
    recom = recom[:k]
    return recom

#Testing
query = {1:2.5, 5:3, 2:4}
test4 = recommend_popular(query, df_filter, k=10)
# print(test4)

#################################################################################################
#3  Implementation of NMF in sklearn
#################################################################################################

#3.1  Filling NaNs with Mean Rates for Each Movie
imputer = SimpleImputer(strategy="mean")
ratings_imp = pd.DataFrame(imputer.fit_transform(df_filter), columns=df_filter.columns, index=df_filter.index)
# print(ratings_imp)

#3.2  NMF model
# Instantiate the NMF model
n_comps=10
nmf = NMF(n_components=n_comps, random_state=42)
nmf.fit(ratings_imp)

#user tastes:(P-matrix) and movie themes/genres:(Q-matrix)
films = df_filter.columns.tolist()
Q = pd.DataFrame(nmf.components_, columns=films, index=[f'genre {i+1}' for i in range(n_comps)])
# print(Q)
# sns.set(rc={'figure.figsize':(20,8)})
# sns.heatmap(Q, annot=False)

P = pd.DataFrame(nmf.transform(ratings_imp), columns=[f'genre {i+1}' for i in range(n_comps)], index=df_filter.index)
# print(P)
# sns.set(rc={'figure.figsize':(20,8)})
# sns.heatmap(P, annot=False)

#Reconstruct R by dotting P and Q (Though not necessary for prediction)
R_recon = P.dot(Q)
# print(R_recon.head())

#Difference from the original ratings
error = nmf.reconstruction_err_
# print(error)

#3.3  Predict movies for a new user
new_user_ratings = {
    5: 4,
    17: 2,
    186: 5,
    1090: 1,
    122920: 3,    
}
new_dict = {}
for film in films:
    if film in new_user_ratings:
        new_dict[film] = new_user_ratings[film]
    else:
        new_dict[film] = np.nan

new_user_df = pd.DataFrame(new_dict, index=[0])
new_user_df.index = ["new_user"]
#Fill-in missing values as before
new_user_df = pd.DataFrame(imputer.transform(new_user_df), columns=films, index=new_user_df.index)
# print(new_user_df)

#Calculate matrix P
P_new_user = pd.DataFrame(nmf.transform(new_user_df), index=new_user_df.index, columns=[f'genre {i+1}' for i in range(n_comps)])
#Create predictions
R_estimate = P_new_user.dot(Q)
#Remove already rated movies
R_estimate = R_estimate.drop(new_user_ratings.keys(), axis=1)
#Sort R_estimate in desceding order
R_estimate_sorted = R_estimate.T.sort_values("new_user", ascending=False)
print(R_estimate_sorted.head(3))

#3.4  saving the model for later use
filename = "nmf_model.sav"
pickle.dump(nmf, open(filename, "wb"))


#defining a function for making recommendations
def nmf_recommender(user_ratings_dict):
    
    '''This function gets a dictionary of {moviename: rate} from user and recommends n movies. To do that,
    it uses nmf model with 10 components which was trained on MovieLens-dataset.'''
    #reading the movie csv file
    movies = pd.read_csv('./data/ml-latest-small/movies.csv', index_col=0)
    #How many movies to recommend
    n_recom = list(user_ratings_dict.items())[-1]
    n = n_recom[1]
    user_ratings_dict.popitem()
    #finding movieIds from the movienames
    user_dict = {}
    for key in user_ratings_dict:
        movieID = movies.index[movies['title']==key].astype(int)[0]
        user_dict[movieID] = user_ratings_dict[key]
    #completing the dictionary and adding all of unseen movies
    new_dict = {}
    for film in films:
        if film in user_dict:
            new_dict[film] = user_dict[film]
        else:
            new_dict[film] = np.nan
    #creating df
    user_df = pd.DataFrame(new_dict, index=[0])
    user_df.index = ["new_user"]
    #Fill-in missing values as before
    user_df = pd.DataFrame(imputer.transform(user_df), columns=films, index=user_df.index)
    #Calculate matrix P
    P_user = pd.DataFrame(nmf.transform(user_df), index=user_df.index, columns=[f'genre {i+1}' for i in range(10)])
    #Create predictions
    R_estimate = P_user.dot(Q)
    #Remove already rated movies
    R_estimate = R_estimate.drop(user_dict.keys(), axis=1)
    #Sort R_estimate in desceding order
    R_estimate_sorted = R_estimate.T.sort_values("new_user", ascending=False)
    recom = R_estimate_sorted.head(n)
    #create a list of movieIds
    recom_movieid = recom.index.tolist()
    #finding movienames
    recommend = []
    for movieid in recom_movieid:
        recommend.append(movies.loc[movieid, 'title'])
    return recommend

#########################################################################################
#5  Neighbourhood based Collaborative Filtering
#########################################################################################

class NeighbourhoodRecommender:
    
 
    def __init__(self, user_ratings):
        self.user_ratings = user_ratings
    
    def prepare_data(self):
        ratings= pd.read_csv('./data/ml-latest-small/ratings.csv', index_col=0)
        ratings= pd.pivot_table(ratings,
                    index='userId',
                    columns='movieId',
                    values='rating')
        ratings= ratings.loc[:, ratings.count()>20]
        imputer = SimpleImputer(strategy="mean")
        ratings_imp = pd.DataFrame(imputer.fit_transform(ratings), columns=ratings.columns, index=ratings.index)
        movies = pd.read_csv('./data/ml-latest-small/movies.csv', index_col=0)
        return ratings, ratings_imp, movies, imputer
    
    def n_movie(self):
        n_recom = list(self.user_ratings.items())[-1]
        n = n_recom[1]
        # print(type(n))
        return int(n)
    
    def finding_movieIds(self):
        control = list(self.user_ratings.keys())
        # print(control)
        last_key = 'How many movies do you want me to recommend'
        if last_key in control:
            # print("I am here")
            self.user_ratings.popitem()
        ratings, ratings_imp, movies, imputer = self.prepare_data()
        user_dict = {}
        for key in self.user_ratings:
            # print(key)
            movieID = movies.index[movies['title']==key].astype(int)[0]
            user_dict[movieID] = self.user_ratings[key]
        return user_dict


    def get_all_movies(self):
        ratings, ratings_imp, movies, imputer = self.prepare_data()
        return ratings.columns   
    
    def populates_unseen_movies_to_user(self):
        user_dic = self.finding_movieIds()
        films = self.get_all_movies()
        
        new_dict = {}
        for film in films:
            if film in user_dic:
                new_dict[film] = user_dic[film]
            else:
                new_dict[film] = np.nan
        return new_dict
    
    def df_user(self):
        '''This function creates a dataframe from user entries.'''
        user_dict = self.populates_unseen_movies_to_user()
        user_df = pd.DataFrame(user_dict, index=[0])
        user_df.index = ["new_user"]        
        #Fill-in missing values as before(with mean)
        ratings, ratings_imp, movies, imputer = self.prepare_data()
        films = self.get_all_movies()
        user_df = pd.DataFrame(imputer.transform(user_df), columns=films, index=user_df.index)
        return user_df

    
    def recommend_movies(self):
        n = self.n_movie()
        ratings, ratings_imp, movies, imputer = self.prepare_data()
        user_dict = self.finding_movieIds()
        seen_movies = list(user_dict.keys())
        films = self.get_all_movies()
        unseen_movies = set(films).difference(set(seen_movies))
        user_df= self.df_user()
        
        predicted_ratings = []
        for movie in unseen_movies:
            # Capture the users who have watched the movie
            whatched_users = ratings[movie][ratings[movie].isna()==False].index
            # Go through each of these users who have watched the movie and calculate av. rating
            num = 0
            den = 0
            for user in whatched_users:
                # capture rating for this `user'
                user_rating = ratings.loc[user][movie]
                # Calculate the similarity between this user and the customer user
                similarity = cosine_similarity(np.array([ratings_imp.loc[user]]), np.array([user_df.loc["new_user"]]))
                num += user_rating * similarity
#                 print(similarity)
                den += similarity
            if den != 0:
                predicted_rating = num/den
            else:
                predicted_rating = 0
            predicted_ratings.append((movie, predicted_rating))

        predicted_rating_df = pd.DataFrame(predicted_ratings, columns=["movie", "rating"])
        predicted_rating_df.sort_values("rating", ascending=False)
        recom_df = predicted_rating_df.head(n)
        #create a list of movieIds
        recom_movieid = recom_df['movie'].tolist()
        #finding movienames
        recom_moviename = []
        for movieid in recom_movieid:
            recom_moviename.append(movies.loc[movieid, 'title'])        
        
        return recom_moviename
