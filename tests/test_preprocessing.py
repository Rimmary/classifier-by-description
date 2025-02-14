import pytest
import pandas as pd
from scripts.preprocessing import preprocess_data

# Sample data to test
data = {
    "description": [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! This is a test sentence.",
        "Python is a great programming language.",
        "This is an example with multiple spaces.  ",
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Expected output for comparison
expected_tokens = [
    ['quick', 'brown', 'fox', 'jump', 'lazy', 'dog'],
    ['hello', 'world', 'test', 'sentence'],
    ['python', 'great', 'programming', 'language'],
    ['example', 'multiple', 'space']
]

def test_preprocess_data():
    """
    Test case for the `preprocess_data` function. This test checks whether the function correctly
    processes the 'description' column in the DataFrame by tokenizing the text, removing stopwords,
    and lemmatizing the words.

    It also ensures that the expected tokens are correctly added to the DataFrame in a new 'tokens' column.
    """

    # Call the preprocess_data function to process the DataFrame
    result = preprocess_data(df)

    # Check that the 'tokens' column has been added to the DataFrame
    assert "tokens" in result.columns, "The 'tokens' column was not added to the DataFrame."

    # Check that each row in the 'tokens' column matches the expected output
    for idx, row in enumerate(result["tokens"]):
        assert row == expected_tokens[idx], f"Tokens for row {idx} do not match. Expected: {expected_tokens[idx]}, Got: {row}"

# Run the test if the script is executed directly
if __name__ == "__main__":
    pytest.main()