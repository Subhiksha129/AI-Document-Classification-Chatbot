
import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading dataset...")

df = pd.read_csv("dataset/processed_training_data.csv")

texts = df["text"].astype(str).tolist()

detail_labels = [
    json.loads(value) for value in df["detail_labels"]
]

# Use the same train-test split as training
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    detail_labels,
    test_size=0.2,
    random_state=42
)

print("Loading trained models...")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/detail_category_model.pkl")
mlb = joblib.load("models/detail_label_encoder.pkl")

print("Preparing test data...")

X_test_tfidf = vectorizer.transform(X_test)
Y_test = mlb.transform(y_test)

print("Generating predictions...")

predictions = model.predict(X_test_tfidf)

print("\nGenerating classification report...")

report = classification_report(
    Y_test,
    predictions,
    target_names=mlb.classes_,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

# Remove overall summary rows
category_report = report_df.drop(
    ["micro avg", "macro avg", "weighted avg", "samples avg"],
    errors="ignore"
)

# Sort categories by F1 score
best_categories = category_report.sort_values(
    "f1-score", ascending=False
)

worst_categories = category_report.sort_values(
    "f1-score", ascending=True
)

print("\nTOP 10 BEST PERFORMING CATEGORIES:")
print(best_categories.head(10)[
    ["precision", "recall", "f1-score", "support"]
].round(3))

print("\nTOP 10 WORST PERFORMING CATEGORIES:")
print(worst_categories.head(10)[
    ["precision", "recall", "f1-score", "support"]
].round(3))

# Save full evaluation report
category_report.to_csv(
    "models/detailed_category_evaluation.csv"
)

print("\nEvaluation report saved!")
print("File: models/detailed_category_evaluation.csv")

print("\nEVALUATION COMPLETED!")
