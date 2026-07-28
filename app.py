import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🧠 AI Sentiment Analyzer")

st.write("Enter a sentence or review below:")

user_input = st.text_area("Your Review")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")
    else:
        st.warning("Please enter some text.")