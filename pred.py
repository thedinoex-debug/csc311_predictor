import sys
import csv
import random
import numpy as np
import pandas as pd


# Saved constants computed from the training set only (see csc311_predictor_fixed.ipynb, cells for cleaning + split)
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
Q8_HI = 14.680000000000064
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
 
    Parameters:
        `filename` - path to a raw CSV file
 
    Returns: a pandas DataFrame with columns FEATURE_COLS, one row
             per input row (in the same order as the input file).
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


def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the four choices: 'Dubai', 'Rio de Janeiro', 'New York City' and 'Paris'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the cities!!
    y = random.choice(['Dubai', 'Rio de Janeiro', 'New York City' ,'Paris'])

    # return the prediction
    return y

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = csv.DictReader(open(filename))

    predictions = []
    for test_example in data:
        # obtain a prediction for this test example
        pred = predict(test_example)
        predictions.append(pred)

    return predictions


if __name__ == '__main__':
    feats = prepare_features('cleaned_dataset.csv')
    print(feats.shape)
    print(feats.head())
