import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('genre_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App Title
st.title("🎬 Movie Genre Predictor")

# Input field
plot_input = st.text_area("Enter the movie plot:", height=200)

# Predict button
if st.button("Predict Genre"):
    if plot_input.strip() == "":
        st.warning("Please enter a movie plot.")
    else:
        # Transform input and predict
        X_new = vectorizer.transform([plot_input])
        predicted_genre = model.predict(X_new)[0]
        st.success(f"🎭 Predicted Genre: **{predicted_genre}**")
