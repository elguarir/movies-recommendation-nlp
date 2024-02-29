import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import requests


class MoviesRecommendation:
    def __init__(self):
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.df = pd.read_csv("Top_10000_Movies.csv", engine="python")
        self.load_model()
        self.clean_data()
        self.generate_embeddings()
        self.fit()

    def clean_data(self):
        self.df = self.df[["id", "original_title", "overview", "genre"]]
        self.df = self.df.dropna()
        self.df = self.df.reset_index()

    def load_model(self):
        model_dir = "universal_sentence_encoder"
        self.model_path = os.path.join(model_dir, "model")

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            self.model = hub.load(self.model_url)
            tf.saved_model.save(self.model, self.model_path)
        else:
            self.model = tf.saved_model.load(self.model_path)
        print("Model Loaded!")

    def generate_embeddings(self):
        titles = list(self.df["overview"])
        self.embeddings = self.model(titles)
        print("The embedding shape is:", self.embeddings.shape)
        self.pca = PCA(n_components=2)
        self.emb_2d = self.pca.fit_transform(self.embeddings)

    def visualize(self):
        plt.figure(figsize=(10, 10))
        plt.scatter(self.emb_2d[:, 0], self.emb_2d[:, 1], alpha=0.5)
        plt.title("Movies Overview Embeddings")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()

    def fit(self):
        self.nn = NearestNeighbors(n_neighbors=10)
        self.nn.fit(self.embeddings)

    def embed(self, texts):
        return self.model(texts)

    def recommend(self, input):
        emb = self.embed([input])
        neighbors = self.nn.kneighbors(emb, return_distance=False)[0]
        recommended_movies = self.df.iloc[neighbors][
            ["id", "original_title", "overview", "genre"]
        ]
        print("Recommended Movies", recommended_movies)
        movies_data = []
        for idx, row in recommended_movies.iterrows():
            movie_data = self.get_movie(row["id"])
            if movie_data:
                movie_data["genre"] = row["genre"]
                movies_data.append(movie_data)
        return movies_data

    def get_movie(self, movie_id):
        movie_id = int(movie_id)
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=7dfcebf9bdfcef936200eab6622f3fba"
        response = requests.get(url)
        data = response.json()

        try:
            tagline = data["tagline"]
            title = data["title"]
            overview = data["overview"]
            release_date = data["release_date"]
            poster_url = f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"
        except KeyError:
            return None
        return {
            "title": title,
            "tagline": tagline,
            "overview": overview,
            "release_date": release_date,
            "poster_url": poster_url,
        }
