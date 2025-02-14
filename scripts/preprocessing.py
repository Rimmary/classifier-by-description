import pickle
import nltk
import pandas as pd
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Uncomment the following lines to download necessary NLTK resources
# nltk.download("stopwords")
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses text data in a DataFrame. The preprocessing steps include:
    1. Tokenization: splitting the text into words.
    2. Removal of stopwords.
    3. Lemmatization: reducing words to their base forms.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing a 'description' column with text data.

    Returns:
        pd.DataFrame: The original DataFrame with an additional column 'tokens' containing the preprocessed text.
    """

    def tokenize_text(text: str) -> list[str]:
        """
        Tokenizes the input text into words and removes common stopwords.

        Args:
            text (str): A string containing the text to tokenize.

        Returns:
            list[str]: A list of tokens (words) from the input text, excluding stopwords.
        """
        # Remove extraneous text characters and normalize case
        text = (
            text.lower().replace("]", "").replace("[", "")
            .replace("\\n", "").replace("\\r\\n", "")
            .replace("^", "").replace("_", "")
            .replace("`", "").replace("\\", "").strip()
        )

        # Tokenize with regular expression (only alphabetic characters)
        tokens = regexp_tokenize(text, "[A-z]+")

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        return tokens

    def lemmatize_text(tokens: list[str]) -> list[str]:
        """
        Lemmatizes the tokens using WordNetLemmatizer and applies part-of-speech (POS) tagging.

        Args:
            tokens (list[str]): A list of tokenized words.

        Returns:
            list[str]: A list of lemmatized tokens.
        """
        # Instantiate the WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        # Generate POS tags for each token
        tokens = nltk.pos_tag(tokens)

        # Lemmatize based on POS tags
        tokens = [lemmatizer.lemmatize(token, pos=f"{tag_map(tag)}") for token, tag in tokens]

        return tokens

    def tag_map(postag: str) -> str:
        """
        Maps part-of-speech tags to WordNet tags used by the lemmatizer.

        Args:
            postag (str): The part-of-speech tag of a word.

        Returns:
            str: The corresponding WordNet part-of-speech tag.
        """
        if postag.startswith("J"):
            return wordnet.ADJ
        elif postag.startswith("V"):
            return wordnet.VERB
        elif postag.startswith(("R", "D", "P", "X")):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # Apply preprocessing steps to the 'description' column
    df["tokens"] = df["description"].apply(lambda x: lemmatize_text(tokenize_text(x)))

    return df


def main():
    """
    The main function that reads the raw data, processes it, and saves the preprocessed data to a pickle file.
    """
    # Load the data from a CSV file into a pandas DataFrame
    df = pd.read_csv("../data/descriptions.csv")

    # Drop rows with null values
    df = df.dropna()

    # Remove rows with extraneous values that do not provide useful descriptions
    df = df[~(df.description.str.match(r'^[\s]+$|^[#]+$|^[-]+$|^\.$|^[0-9\s\-\.]+$|^\b(?:\w+\s*){1,2}\b$|No\. 30247: 3\/8"-24M'))]

    # Perform text preprocessing
    preprocessed_data = preprocess_data(df)

    # Save the preprocessed DataFrame to a pickle file for future use
    with open("../data/preprocessed_df.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()