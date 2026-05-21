from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

from src.data_loader import load_data
from src.feature_engineering import create_features
from src.preprocessing import build_preprocessor

# Load data
df = load_data("data/german_credit_data.csv")

# Feature engineering
df = create_features(df)

# Split
X = df.drop("Risk", axis=1)

y = df["Risk"]

# Encode target
y = y.map({
    "bad": 0,
    "good": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

# Columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessor
preprocessor = build_preprocessor(
    categorical_cols,
    numerical_cols
)

# Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X_train, y_train)

# Save
joblib.dump(
    pipeline,
    "models/xgboost_pipeline.pkl"
)