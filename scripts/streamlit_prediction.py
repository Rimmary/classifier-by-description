import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes the input text into individual words after converting it to lowercase
    and removing stopwords.

    Args:
        text (str): Input text string to be tokenized.

    Returns:
        list[str]: A list of tokens (words) after processing.
    """
    text = text.lower().strip()

    tokens = regexp_tokenize(text, "[A-z]+")

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


def lemmatize_text(tokens: list[str]) -> list[str]:
    """
    Lemmatizes the list of tokens using WordNetLemmatizer. Lemmatization is
    performed without part-of-speech (POS) tags to optimize performance.

    Args:
        tokens (list[str]): A list of tokens to be lemmatized.

    Returns:
        list[str]: A list of lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def load_models():
    """
    Loads pre-trained machine learning models (Word2Vec, Random Forest, Label Encoder)
    from local files.

    Returns:
        tuple: A tuple containing the loaded models:
            - w2v_model: Word2Vec model
            - rf_model: Random Forest model
            - label_encoder: LabelEncoder model
    """
    with open("../data/w2v_model.pkl", "rb") as f:
        w2v_model = pickle.load(f)

    with open("../data/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    with open("../data/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return w2v_model, rf_model, label_encoder


def predict_top3_categories(text_input, w2v_model, rf_model, label_encoder):
    """
    Predicts the top 3 categories for a given text input using pre-trained models.

    Args:
        text_input (str): The input text describing the goods.
        w2v_model (Word2Vec): Pre-trained Word2Vec model to convert words into embeddings.
        rf_model (RandomForestClassifier): Pre-trained Random Forest model for classification.
        label_encoder (LabelEncoder): Pre-trained Label Encoder to decode the predicted labels.

    Returns:
        tuple: A tuple containing:
            - top3_classes: The top 3 predicted categories.
            - predicted_proba: Probabilities for each predicted class.
            - top3_idx: Indices of the top 3 predicted classes.
    """
    input_tokens = lemmatize_text(tokenize_text(text_input))

    # Get the word embeddings for the tokens in the input text
    input_embed = np.mean([w2v_model.wv[word] for word in input_tokens if word in w2v_model.wv]
                          or [np.zeros(300)], axis=0)

    input_df = pd.DataFrame([input_embed])

    predicted_proba = rf_model.predict_proba(input_df)

    top3_idx = np.argsort(predicted_proba, axis=1)[:, -3:]

    top3_classes_encoded = rf_model.classes_[top3_idx]
    top3_classes_encoded_flat = top3_classes_encoded.flatten()

    # Convert the encoded labels back to original class names using label encoder
    top3_classes = label_encoder.inverse_transform(top3_classes_encoded_flat)
    top3_classes = top3_classes.reshape(top3_classes_encoded.shape)

    return top3_classes, predicted_proba, top3_idx


def main():
    """
    Main function to launch the Streamlit web application. The application allows the user
    to input a description of goods and predict the top 3 possible categories.
    """
    # Set up the title of the Streamlit app
    st.title('Goods Category Prediction App')

    # Create an input text area for the user to input a goods description
    st.markdown('Enter Goods Description')
    text_input = st.text_area('', 'Type Here...', label_visibility="collapsed")

    button, clear = st.columns([9, 1])

    # Button click event to predict the top 3 categories
    with button:
        if st.button('Predict'):
            # Load models
            w2v_model, rf_model, label_encoder = load_models()

            # Get top 3 categories and their probabilities
            top3_classes, predicted_proba, top3_idx = predict_top3_categories(text_input, w2v_model, rf_model, label_encoder)

            # Display the top 3 predicted categories in a new container
            st.markdown("Top 3 Predicted Categories:")
            for i, classes in enumerate(top3_classes):
                for category, prob in zip(classes[::-1], predicted_proba[i, top3_idx[i]][::-1]):
                    st.markdown(f"{category} {prob * 100:.0f}%")
    with clear:
        clear_button = st.button('Clear')

        if clear_button:
            st.empty()  # Clear the displayed content


# Ensure the app runs when the script is executed
if __name__ == "__main__":
    main()
