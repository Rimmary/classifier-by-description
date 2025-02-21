import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
np.random.seed(42)

def train_rf(train_df, train_labels):
    """
    Trains a Random Forest Classifier using the provided training data and labels.

    Parameters:
    train_df (DataFrame): The feature data used for training.
    train_labels (array-like): The target labels for training.

    Returns:
    rf_model (RandomForestClassifier): The trained RandomForestClassifier model.
    """
    rf_model = RandomForestClassifier()
    rf_model.fit(train_df, train_labels)
    return rf_model

def main():
    """
    Main function to load data, train the Random Forest model, evaluate it,
    and save the trained model and label encoder.

    This function:
    - Loads training and testing data from pickle files.
    - Encodes the labels using LabelEncoder.
    - Trains a Random Forest model.
    - Evaluates the model using accuracy on both training and test sets.
    - Saves the trained model and label encoder to pickle files.
    """
    # Load training and testing datasets and labels from pickle files. They are created and saved in embedding.py
    with open("../data/train_df.pkl", "rb") as f:
        train_df = pickle.load(f)

    with open("../data/test_df.pkl", "rb") as f:
        test_df = pickle.load(f)

    with open("../data/train_labels.pkl", "rb") as f:
        train_labels = pickle.load(f)

    with open("../data/test_labels.pkl", "rb") as f:
        test_labels = pickle.load(f)

    # Initialize LabelEncoder to encode the labels into _numeric_ values
    label_encoder = LabelEncoder()

    label_encoder.fit(train_labels)
    train_labels = label_encoder.transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    rf_model = train_rf(train_df, train_labels)

    predict_train = rf_model.predict(train_df)
    predict_test = rf_model.predict(test_df)

    print(f"Train accuracy: {accuracy_score(train_labels, predict_train)}")
    print(f"Test accuracy: {accuracy_score(test_labels, predict_test)}")

    with open('../data/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f, pickle.HIGHEST_PROTOCOL)

    with open("../data/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()