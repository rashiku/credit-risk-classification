import pandas as pd

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
    credit_threshold = (
        df["Credit amount"]
        .quantile(0.75)
    )

    duration_threshold = (
        df["Duration"]
        .quantile(0.75)
    )

    df["High_Credit"] = (
        df["Credit amount"] > credit_threshold
    ).astype(int)

    df["Long_Duration"] = (
        df["Duration"] > duration_threshold
    ).astype(int)

    return df