import streamlit as st
import pickle

# Load TF-IDF and Model
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Sentiment Classifier")

st.title("Sentiment Analysis App")

# Input text
text = st.text_area("Enter your sentence below:", height=150)

if st.button("Proceed"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        vec = tfidf.transform([text])
        
        # Predict label
        prediction = model.predict(vec)[0]

        # Format result text
        result = prediction.upper()

        # Display in medium font
        st.markdown(
            f"<h3 style='text-align:center; color:#333;'>{result}</h3>",
            unsafe_allow_html=True
        )
