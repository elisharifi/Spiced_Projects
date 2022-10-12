import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity


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
        '''This function extracts the last user entry which is the number of movies that he/she wants to get as recommendation.'''
        n_recom = list(self.user_ratings.items())[-1]
        n = n_recom[1]
        return n
    
    def finding_movieIds(self):
        if 'How many movies do you want me to recommend:' in list(self.user_ratings.keys()):
            self.user_ratings.popitem()
        ratings, ratings_imp, movies, imputer = self.prepare_data()
        user_dict = {}
        for key in self.user_ratings:
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

#example to test
new_user_ratings = {
    'Shawshank Redemption, The (1994)':5,
    'Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)': 5,
    'Philadelphia Story, The (1940)': 1,
    'How many movies do you want me to recommend:': 12
}
recommender = NeighbourhoodRecommender(new_user_ratings)
print(recommender.recommend_movies())