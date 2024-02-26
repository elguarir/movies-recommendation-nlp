import streamlit as st
from recommendation import MoviesRecommendation

class MovieRecommendationApp:
    def __init__(self):
        self.movies_recommendation = MoviesRecommendation()

    def main(self):
        st.title("Movies Recommendation App")

        user_input = st.text_input("Enter a movie query:", "Spiderman")
        if st.button("Recommend Movies"):
            recommended_movies = self.movies_recommendation.recommend(user_input)

            st.subheader("Recommended Movies:")

            num_columns = st.columns(3)  # Create responsive columns

            for idx, col in enumerate(num_columns):
                for movie_index in range(idx, len(recommended_movies), len(num_columns)):
                    movie = recommended_movies[movie_index]
                    if movie is not None:
                        col.markdown(
                            f"**{movie['title']}**\n"
                            f"Tagline: {movie['tagline']}\n"
                            f"Release Date: {movie['release_date']}\n"
                            f"Overview: {movie['overview']}"
                        )
                        col.image(movie['poster_url'], caption=movie['title'], use_column_width=True)

if __name__ == "__main__":
    app = MovieRecommendationApp()
    app.main()