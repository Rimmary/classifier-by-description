import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.model_selection import train_test_split

def train_word2vec(train_data: pd.Series(list[str]), test_data: pd.Series(list[str])) -> \
        (KeyedVectors, pd.DataFrame, pd.DataFrame):
    """
    Train a Word2Vec model on the given training data and generate word embeddings for both the
    training and testing datasets.

    Parameters:
    - train_data (pd.Series): Tokenized text data for the training set.
    - test_data (pd.Series): Tokenized text data for the testing set.

    Returns:
    - w2v_model (KeyedVectors): The trained Word2Vec model.
    - train_df (pd.DataFrame): A DataFrame of word embeddings for the training set.
    - test_df (pd.DataFrame): A DataFrame of word embeddings for the testing set.
    """
    # Instantiate the Word2Vec model with specific parameters
    w2v_model = Word2Vec(min_count=5, window=5, sg=0, vector_size=300, sample=6e-5, negative=20)

    # Build the vocabulary for the model using the training data
    w2v_model.build_vocab(train_data)

    # Train the model using the training data for 15 epochs
    w2v_model.train(train_data, total_examples=w2v_model.corpus_count, epochs=15)

    # Generate embeddings for the training set by averaging word embeddings for each tokenized sentence
    train_df = pd.DataFrame([np.mean([w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
                                     or [np.zeros(300)], axis=0) for tokens in train_data])

    # Generate embeddings for the testing set by averaging word embeddings for each tokenized sentence
    test_df = pd.DataFrame([np.mean([w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
                                     or [np.zeros(300)], axis=0) for tokens in test_data])

    return w2v_model, train_df, test_df

def main():
    """
    The main function orchestrates the entire pipeline: loading preprocessed data, splitting into
    training and testing sets, training a Word2Vec model, and saving the results as pickle files.
    """
    # Load the preprocessed DataFrame from the pickle file
    with open("../data/preprocessed_df.pkl", "rb") as f:
        df = pickle.load(f)

    # Split the data into training and testing sets (90% train, 10% test)
    train, test = train_test_split(df, test_size=.1, random_state=10, stratify=df.categories)

    # Extract labels (categories) for training and testing
    train_labels = train.categories
    test_labels = test.categories

    # Extract tokenized text data from training and testing sets
    train_tokens = train.tokens
    test_tokens = test.tokens

    # Train the Word2Vec model on the tokenized training data
    w2v_model, train_df, test_df = train_word2vec(train_tokens, test_tokens)

    # Display the first 50 words in the Word2Vec vocabulary
    # print(sorted(w2v_model.wv.key_to_index.keys())[:50])

    # Save the Word2Vec model and the word embeddings as pickle files
    with open('../data/w2v_model.pkl', 'wb') as f:
        pickle.dump(w2v_model, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/train_df.pkl', 'wb') as f:
        pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/test_df.pkl', 'wb') as f:
        pickle.dump(test_df, f, pickle.HIGHEST_PROTOCOL)

    # Save the labels for the train and test sets
    with open('../data/train_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/test_labels.pkl', 'wb') as f:
        pickle.dump(test_labels, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()