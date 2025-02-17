import pytest
import pickle
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from scripts.embedding import train_word2vec
import os


@pytest.fixture
def mock_data():
    """
    Create a mock dataset with descriptions and categories for testing.

    This fixture prepares a small sample dataset to simulate real-world data
    for Word2Vec training and evaluation. Each description is tokenized into words.

    Returns:
        pd.DataFrame: A DataFrame containing the mock data with 'description',
                      'categories', and 'tokens' columns.
    """
    # Create a small mock dataset for testing
    data = pd.DataFrame({
        'description': [
            "This is a test sentence.",
            "Lalala is fine.",
            "We are testing the Word2Vec model.",
            "This sentence is for training.",
            "Word embeddings are useful for NLP tasks.",
            "Natural language processing is fun.",
            "Machine learning helps in automating tasks.",
            "Deep learning models are a type of machine learning.",
            "Supervised learning requires labeled data.",
            "Unsupervised learning does not use labels.",
            "Reinforcement learning is about agents interacting with environments.",
            "Data science is an interdisciplinary field.",
            "Artificial intelligence is a subfield of computer science.",
            "Neural networks are a part of deep learning.",
            "The internet of things connects devices via the internet.",
            "Big data technologies are essential for modern analytics.",
            "Data visualization helps in understanding data.",
            "AI ethics is an important topic in modern AI research.",
            "Text classification is a common NLP task.",
            "Clustering is an unsupervised learning technique."
        ],
        'categories': [
            'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B',
            'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'
        ]
    })

    # Tokenize the mock data by converting to lowercase and splitting by spaces
    data['tokens'] = data['description'].apply(
        lambda x: x.lower().split()  # Simple tokenization for testing purposes
    )

    return data


@pytest.fixture
def split_data(mock_data):
    """
    Split the mock dataset into train and test sets.

    This fixture takes the mock data and splits it into a training set (75%)
    and a test set (25%) while preserving the category distribution.

    Args:
        mock_data (pd.DataFrame): The mock dataset containing descriptions,
                                  categories, and tokens.

    Returns:
        tuple: A tuple containing the train and test DataFrames.
    """
    # Split the dataset into train and test sets (75% train, 25% test)
    return train_test_split(mock_data, test_size=0.25, random_state=10, stratify=mock_data.categories)


@pytest.fixture(scope="module", autouse=True)
def cleanup_files():
    """
    Automatically clean up files created during the tests.

    This fixture ensures that files created during the tests (e.g., pickled
    models and dataframes) are removed after the tests are finished, avoiding
    cluttering the working directory.

    Scope: "module" ensures that this fixture runs once per test module.
    """
    # Yield allows the tests to run first
    yield

    # Cleanup phase: Remove the generated files after the tests
    files = ['test_w2v_model.pkl', 'test_train_df.pkl', 'test_test_df.pkl']
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")


def test_train_word2vec(split_data):
    """
    Test the Word2Vec training process.

    This test ensures that the Word2Vec model can be trained on the mock dataset,
    the model is not empty, and that the tokenized DataFrames are returned correctly.

    Args:
        split_data (tuple): A tuple containing the train and test DataFrames.
    """
    train, test = split_data

    # Extract tokens for train and test sets
    train_tokens = train['tokens']
    test_tokens = test['tokens']

    # Train the Word2Vec model using the train tokens and evaluate using test tokens
    w2v_model, train_df, test_df = train_word2vec(train_tokens, test_tokens)

    # Check that the trained model is a Word2Vec instance and its vocabulary is not empty
    assert isinstance(w2v_model, Word2Vec), "The trained model should be an instance of Word2Vec"
    assert len(w2v_model.wv.key_to_index) > 0, "Vocabulary should not be empty"

    # Check that the returned DataFrames match the number of tokens in the train and test sets
    assert train_df.shape[0] == len(train_tokens), "Train DataFrame should have the same number of rows as tokens"
    assert test_df.shape[0] == len(test_tokens), "Test DataFrame should have the same number of rows as tokens"


def test_pickle_output(split_data):
    """
    Test if the pickling of the model and dataframes works.

    This test ensures that the model and DataFrames can be successfully pickled
    and then unpickled, preserving their original structure.

    Args:
        split_data (tuple): A tuple containing the train and test DataFrames.
    """
    train, _ = split_data
    # Pickle the DataFrame and check if it can be reloaded correctly
    with open('test_w2v_model.pkl', 'wb') as f:
        pickle.dump(train, f)

    # Check if the pickled file can be loaded correctly
    with open('test_w2v_model.pkl', 'rb') as f:
        loaded_df = pickle.load(f)

    # Verify that the loaded DataFrame has the same shape as the original
    assert train.shape == loaded_df.shape, "The shape of the pickled data should be the same"


def test_train_and_save(split_data):
    """
    Test the entire pipeline: train, save, and load the Word2Vec model and DataFrames.

    This test ensures that the Word2Vec model and the associated DataFrames can be
    saved to disk as pickled files, and then successfully reloaded from disk.

    Args:
        split_data (tuple): A tuple containing the train and test DataFrames.
    """
    train, test = split_data
    train_tokens = train['tokens']
    test_tokens = test['tokens']

    # Train the Word2Vec model
    w2v_model, train_df, test_df = train_word2vec(train_tokens, test_tokens)

    # Save the trained model and DataFrames to pickle files
    with open('test_w2v_model.pkl', 'wb') as f:
        pickle.dump(w2v_model, f, pickle.HIGHEST_PROTOCOL)
    with open('test_train_df.pkl', 'wb') as f:
        pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)
    with open('test_test_df.pkl', 'wb') as f:
        pickle.dump(test_df, f, pickle.HIGHEST_PROTOCOL)

    # Load the model and DataFrames from pickle files
    with open('test_w2v_model.pkl', 'rb') as f:
        loaded_w2v_model = pickle.load(f)
    with open('test_train_df.pkl', 'rb') as f:
        loaded_train_df = pickle.load(f)
    with open('test_test_df.pkl', 'rb') as f:
        loaded_test_df = pickle.load(f)

    # Assertions to ensure the loaded objects are correct
    assert isinstance(loaded_w2v_model, Word2Vec), "Loaded model should be an instance of Word2Vec"
    assert loaded_train_df.shape == train_df.shape, "Loaded train_df should have the same shape as the original one"
    assert loaded_test_df.shape == test_df.shape, "Loaded test_df should have the same shape as the original one"
