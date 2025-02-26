# classifier-by-description

This project contains a machine learning pipeline for classifying products into categories based on their descriptions. The pipeline uses a Random Forest Classifier combined with Word2Vec embeddings for text data preprocessing and categorization. The model can predict the top 3 categories for a given product description.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Testing](#testing)

## Overview

The classifier works in several stages:

1. **Preprocessing**: The input text descriptions are tokenized, lemmatized, and stopwords are removed.
2. **Embedding**: A Word2Vec model is used to convert the text into numerical representations (embeddings).
3. **Model Training**: A Random Forest Classifier is trained on the processed and embedded text.
4. **Prediction**: The trained model predicts the top 3 categories for a given description using the trained Random Forest model and Word2Vec embeddings.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rimmary/classifier-by-description.git
   cd classifier_by_description
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary datasets. You can use ../data/descriptions_example.csv as example of a structure.

## File Structure

```
/classifier_by_description
├── /scripts
│   ├── embedding.py                # Contains functions for training Word2Vec model.
│   ├── preprocessing.py            # Contains functions for preprocessing text data.
│   ├── rf_classifier.py            # Contains functions for training and evaluating Random Forest Classifier.
│   ├── streamlit_prediction.py     # Streamlit app to input descriptions and predict categories.
├── /tests
│   ├── test_embedding.py           # Tests for Word2Vec model training.
│   ├── test_preprocessing.py       # Tests for data preprocessing.
├── /data
│   ├── descriptions_example.csv    # Example data.
│   ├── preprocessed_df.pkl         # (too heavy to load) Preprocessed full data from preprocessing.py.
│   ├── train_df.pkl                # (too heavy to load) Preprocessed training data from embedding.py.
│   ├── test_df.pkl                 # (too heavy to load) Preprocessed test data from embedding.py.
│   ├── train_labels.pkl            # (too heavy to load) Training labels from embedding.py.
│   ├── test_labels.pkl             # (too heavy to load) Test labels from embedding.py.
│   ├── w2v_model.pkl               # Trained Word2Vec model from embedding.py.
│   ├── rf_model.pkl                # (too heavy to load) Trained Random Forest Classifier model from rf_classifier.py.
│   ├── label_encoder.pkl           # (too heavy to load) Label encoder for transforming categories from rf_classifier.py.
├── requirements.txt                # Python dependencies.
├── README.md                       # Project documentation.
```

## Usage

0. **Change the Input File**:
    By default, the preprocessing.py script expects a file located at ../data/descriptions_example.csv. If you need to use a different file with product descriptions, you can update the file path in preprocessing.py at line 107:
    ```python
    df = pd.read_csv("../data/descriptions.csv")
    ```

1. **Train the models**:
   To train the Word2Vec and Random Forest models, run the `preprocessing.py`, `embedding.py` and `rf_classifier.py` scripts with your training data.

   Example:
   ```bash
   cd ~/classifier_by_description/scripts
   python preprocessing.py
   python embedding.py
   python rf_classifier.py
   ```

2. **Run the Streamlit app**:
   To interact with the model and predict categories, use the Streamlit app. After running the app, you can input product descriptions and view the predicted top 3 categories.

   Example:
   ```bash
   cd ~/classifier_by_description/scripts
   streamlit run streamlit_prediction.py
   ```

3. **Test the pipeline**:
   The project includes unit tests for preprocessing and model training. You can run the tests using `pytest`.

   Example:
   ```bash
   cd ~/classifier_by_description
   pytest
   ```

## Testing

The project includes test cases for various components:

- **Preprocessing**: The `test_preprocessing.py` file tests the text tokenization, stopword removal, and lemmatization.
- **Embedding**: The `test_embedding.py` file tests the Word2Vec model training and serialization.