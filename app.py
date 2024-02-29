import streamlit as st
from recommendation import MoviesRecommendation

# Initialize the MoviesRecommendation class
rec = MoviesRecommendation()

# Title of the Streamlit app
st.title('Movie Recommender System')

# User input for movie description
user_input = st.text_area("Enter a movie description to get recommendations:")

# Button to get recommendations
if st.button('Recommend'):
    if user_input:
        # Get recommendations
        recommendations = rec.recommend(user_input)
        
        # Display each recommended movie
        for movie in recommendations:
            st.subheader(movie['title'])
            st.write(f"Tagline: {movie['tagline']}")
            st.write(f"Overview: {movie['overview']}")
            st.write(f"Release Date: {movie['release_date']}")
            if(len(movie["genre"]) >0):
                genres = ", ".join(movie["genre"])
            st.image(movie['poster_url'])
            st.write("---")
    else:
        st.write("Please enter a movie description to get recommendations.")

# Run the app with `streamlit run app.py`
