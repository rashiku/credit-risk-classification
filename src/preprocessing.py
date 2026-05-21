from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline

def build_preprocessor(
    categorical_cols,
    numerical_cols
):

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_transformer,
                categorical_cols
            ),
            (
                "num",
                numerical_transformer,
                numerical_cols
            )
        ]
    )

    return preprocessor