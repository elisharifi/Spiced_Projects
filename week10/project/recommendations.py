import random
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

# prepration of data, and initialization of imputer
df = pd.read_csv('./data/ml-latest-small/ratings.csv', index_col=0)
df=pd.pivot_table(df,
            index='userId',
            columns='movieId',
            values='rating')
df = df.loc[:, df.count()>20]
movies = pd.read_csv('./data/ml-latest-small/movies.csv', index_col=0)


imputer = SimpleImputer(strategy="mean")
ratings_imp = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

films = df.columns.tolist()


movie_list = ["Dune",
              "Star Wars",
              "Lost",
              "Shawshank Redemption",
              "24",
              "Inception",
              "Shutter Island",
              "All Dogs Go to Heaven",
              "12 Angry Men"]

def random_recommender():
    random.shuffle(movie_list)
    return movie_list[:4]

def nmf_recommender(user_ratings_dict):
    '''This function gets a dictionary of {moviename: rate} from user and recommends n movies. To do that,
    it uses nmf model with 10 components which was trained on MovieLens-dataset.'''
    
    nmf = pickle.load(open("nmf_model.sav", "rb"))
    Q = pd.DataFrame(nmf.components_, columns=films, index=[f'genre {i+1}' for i in range(10)])

    #How many movies to recommend
    n_recom = list(user_ratings_dict.items())[-1]
    n = int(n_recom[1])
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
    print(recommend)
    return recommend


class NeighbourhoodRecommender:
    
 
    def __init__(self, user_ratings):
        self.user_ratings = user_ratings
    
    def prepare_data(self):
        '''This function reads rating data, converts it to wode format and
        filters out movies with less than 20 ratings. It also reads movie data
        to get the movie titles. In addition, for filling the nans in rating data, 
        it uses a simpleimputer.'''
        ratings= pd.read_csv('./data/ml-latest-small/ratings.csv', index_col=0)
        ratings= pd.pivot_table(ratings,
                    index='userId',
                    columns='movieId',
                    values='rating')
        ratings= ratings.loc[:, ratings.count()>20]
        mv = list(ratings.columns)
        random.shuffle(mv)
#         print(mv[:8])
        imputer = SimpleImputer(strategy="mean")
        ratings_imp = pd.DataFrame(imputer.fit_transform(ratings), columns=ratings.columns, index=ratings.index)
        movies = pd.read_csv('./data/ml-latest-small/movies.csv', index_col=0)
        return ratings, ratings_imp, movies, imputer
    
    def n_movie(self):
        '''This function extracts the last user entry which is the number of 
        movies that he/she wants to get as recommendation.'''
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

if __name__ == '__main__':
    print(random_recommender())