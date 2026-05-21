import pandas as pd

# Fixed thresholds from training data
CREDIT_THRESHOLD = 5000
DURATION_THRESHOLD = 36

def create_features(df):

    # Credit burden
    df["Credit_to_Duration"] = (
        df["Credit amount"] / df["Duration"]
    )

    # Example mappings
    savings_map = {
        "little": 1,
        "moderate": 2,
        "quite rich": 3,
        "rich": 4,
        "unknown": 0
    }

    checking_map = {
        "little": 1,
        "moderate": 2,
        "rich": 3,
        "unknown": 0
    }

    df["Saving_accounts_encoded"] = (
        df["Saving accounts"]
        .map(savings_map)
    )

    df["Checking_account_encoded"] = (
        df["Checking account"]
        .map(checking_map)
    )

    df["Account_Stability"] = (
        df["Saving_accounts_encoded"]
        + df["Checking_account_encoded"]
    )

    # Binary features
    df["High_Credit"] = (
        df["Credit amount"] > CREDIT_THRESHOLD
    ).astype(int)

    df["Long_Duration"] = (
        df["Duration"] > DURATION_THRESHOLD
    ).astype(int)

    return df