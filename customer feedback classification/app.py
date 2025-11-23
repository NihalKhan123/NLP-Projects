import streamlit as st
import pickle

# ---------------------------
# Load TF-IDF and Model
# ---------------------------
def load_artifacts():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return tfidf, model

# ---------------------------
# Main UI
# ---------------------------
def main():

    st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")

    st.title("💬 Customer Feedback Sentiment Classifier")

    st.write("Type your sentence and click **Proceed** for sentiment prediction.")

    tfidf, model = load_artifacts()

    text = st.text_area("Enter your sentence:", height=150)

    # LABEL MAPPING
    label_map = {
        0: "NEGATIVE",
        1: "NEUTRAL",
        2: "POSITIVE"
    }

    if st.button("Proceed"):

        if text.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            vec = tfidf.transform([text])
            prediction = model.predict(vec)[0]

            # Convert numeric → string label
            result = label_map[int(prediction)]

            # RED COLOR + BIG FONT
            st.markdown(
                f"""
                <h1 style='text-align:center; color:red; font-size:40px; font-weight:700;'>
                    {result}
                </h1>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
