import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[['Age', 'Income', 'CreditScore', 'LoanAmount']] = imputer.fit_transform(df[['Age', 'Income', 'CreditScore', 'LoanAmount']])

    # Encode categorical features
    le = LabelEncoder()
    df['Education'] = le.fit_transform(df['Education'])

    # Split features and target
    X = df.drop('LoanApproved', axis=1)
    y = df['LoanApproved']

    return X, y
