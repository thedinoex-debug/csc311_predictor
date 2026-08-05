import numpy as np
import pandas as pd


# These values came from the training set, computed in the cleaning notebook
Q6_WORDS = ['Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']
Q6_COLS = ['Q6_' + w.replace(' ', '_') for w in Q6_WORDS]
Q6_MEDIANS = {
    'Q6_Skyscrapers': 4.0,
    'Q6_Sport': 3.0,
    'Q6_Art_and_Music': 4.0,
    'Q6_Carnival': 3.0,
    'Q6_Cuisine': 4.0,
    'Q6_Economic': 4.0,
}
 
COMPANION_TYPES = ['Partner', 'Friends', 'Siblings', 'Co-worker']
Q5_COLS = [f'Q5_{c}' for c in COMPANION_TYPES]
 
KEYWORD_MAP = {
    'Q10_kw_tallest': 'tallest',
    'Q10_kw_carnival': 'carnival',
    'Q10_kw_eiffel': 'eiffel',
    'Q10_kw_skyscraper': 'skyscraper',
}
 
Q7_LO, Q7_HI = -20, 45
Q8_HI = 20.0
Q9_HI = 100.0
 
FEATURE_COLS = (
    ['Q1', 'Q2', 'Q3', 'Q4']
    + Q5_COLS
    + Q6_COLS
    + ['Q7_capped', 'Q8_capped', 'Q9_capped', 'Q10_wordcount']
    + list(KEYWORD_MAP.keys())
)


def prepare_features(filename):
    """
    Load a raw test CSV (same format as cleaned_dataset.csv) and
    transform it into a numeric feature matrix, using the exact same
    steps as the cleaning notebook, with all thresholds/medians fixed
    to values learned from the training set only.
    """
    df = pd.read_csv(filename)
 
    # Q7/Q9: strip commas, convert to float
    df['Q7'] = df['Q7'].astype(str).str.replace(',', '', regex=False)
    df['Q7'] = pd.to_numeric(df['Q7'], errors='coerce')
    df['Q9'] = df['Q9'].astype(str).str.replace(',', '', regex=False)
    df['Q9'] = pd.to_numeric(df['Q9'], errors='coerce')
 
    # Q10: missing -> empty string 
    df['Q10'] = df['Q10'].fillna('')
 
    # Q6: parse "word=>rank,word=>rank,..." into 6 numeric columns
    def parse_q6(s):
        ranks = {}
        for part in str(s).split(','):
            if '=>' not in part:
                continue
            word, rank = part.split('=>')
            ranks[word.strip()] = pd.to_numeric(rank, errors='coerce')
        return pd.Series(ranks)
 
    q6_ranks = df['Q6'].apply(parse_q6)
    q6_ranks.columns = ['Q6_' + c.replace(' ', '_') for c in q6_ranks.columns]
    # make sure all 6 expected columns exist, even if a word never appears
    for col in Q6_COLS:
        if col not in q6_ranks.columns:
            q6_ranks[col] = np.nan
    q6_ranks = q6_ranks[Q6_COLS]
    df = pd.concat([df, q6_ranks], axis=1)
    # fill missing ranks with the train medians 
    for col in Q6_COLS:
        df[col] = df[col].fillna(Q6_MEDIANS[col])
 
    # Q5: multi-select -> one binary indicator column per companion type
    for c in COMPANION_TYPES:
        df[f'Q5_{c}'] = df['Q5'].fillna('').str.contains(c, regex=False).astype(int)
 
    # Q7-9: cap outliers using the train bounds
    df['Q7_capped'] = df['Q7'].clip(Q7_LO, Q7_HI)
    df['Q8_capped'] = df['Q8'].clip(upper=Q8_HI)
    df['Q9_capped'] = df['Q9'].clip(upper=Q9_HI)
 
    # Q10: word count + keyword flags
    df['Q10_wordcount'] = df['Q10'].fillna('').str.split().str.len()
    for col, kw in KEYWORD_MAP.items():
        df[col] = df['Q10'].fillna('').str.lower().str.contains(kw, regex=False).astype(int)
 
    # any remaining missing value get filled with 0 so predict_all never crashes on a single bad row
    features = df[FEATURE_COLS].fillna(0)
 
    return features


# Load the trained weights and standardization values, saved by train.py
_params = np.load('model_params.npz', allow_pickle=True)
W = _params['W']
MEAN = _params['mean']
STD = _params['std']
CLASSES = [str(c) for c in _params['classes']]


def softmax(z):
    """
    Turn a matrix of raw scores z (shape N x K) into probabilities that add up to 1 across each row.
    """
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    features = prepare_features(filename)

    # standardize using the mean/std from the training set
    X = ((features.to_numpy() - MEAN) / STD)

    # add a column of 1s for the bias/intercept
    X = np.hstack([np.ones((len(X), 1)), X])

    # predict probabilities for each city, then pick most likely one
    Y = softmax(np.dot(X, W))
    predicted_city_indices = np.argmax(Y, axis=1)

    predictions = [CLASSES[i] for i in predicted_city_indices]

    return predictions

# TODO REMOVE THIS TESTING CODE BEFORE SUBMISSION
if __name__ == '__main__':
    predictions = predict_all('test_raw.csv')  # 223 rows that were in test_features.csv but in original form in cleaned_dataset.csv
    print('First 10 predictions:', predictions[:10])
    print('Total predictions made:', len(predictions))

    # accuracy check, only works if the file has a Label column
    true_labels = pd.read_csv('test_raw.csv')['Label'].tolist()
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    print(f'Accuracy: {correct}/{len(true_labels)} = {correct / len(true_labels):.4f}')

