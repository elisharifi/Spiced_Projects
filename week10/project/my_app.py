from flask import Flask, render_template, request
from recommendations import random_recommender, nmf_recommender, NeighbourhoodRecommender

app = Flask(__name__, static_url_path = '/static')

@app.route("/")
def homepage():
    return render_template("homepage.html", title = "What movie should I watch?")

# @app.route("/recs")
# def recommendations():
#     form = request.args
#     results = random_recommender()
#     return render_template("recommendations.html", movies = results, votes = form)

@app.route("/recs")
def recommendations():
    user_dict= (request.args)
    user_dict = user_dict.to_dict()
    results = nmf_recommender(user_dict)
    length = len(results)
    return render_template("recommendations.html", movies = results, len=length)

# @app.route("/recs")
# def recommendations():
#     user_dict= (request.args)
#     user_dict = user_dict.to_dict()
#     recommender = NeighbourhoodRecommender(user_dict)
#     results = recommender.recommend_movies()
#     length = len(results)
#     return render_template("recommendations.html", movies = results, len=length)

if __name__ == '__main__':
    app.run(debug = True)