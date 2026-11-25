
import pandas as pd
import json
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

print("Loading expanded dataset...", flush=True)

df = pd.read_csv("dataset/expanded_training_data.csv")

texts = df["text"].astype(str).tolist()

main_labels = [
    json.loads(value) for value in df["main_labels"]
]

detail_labels = [
    json.loads(value) for value in df["detail_labels"]
]

# Split the dataset
(
    X_train,
    X_test,
    main_train,
    main_test,
    detail_train,
    detail_test
) = train_test_split(
    texts,
    main_labels,
    detail_labels,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train), flush=True)
print("Testing records:", len(X_test), flush=True)

# Create and train a new TF-IDF vectorizer
print("\nCreating new TF-IDF vectorizer...", flush=True)

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF features:", X_train_tfidf.shape[1], flush=True)

# Prepare main category labels
print("\nPreparing main category labels...", flush=True)

main_mlb = MultiLabelBinarizer()

Y_main_train = main_mlb.fit_transform(main_train)
Y_main_test = main_mlb.transform(main_test)

print("Main categories:", len(main_mlb.classes_), flush=True)

# Prepare detailed category labels
print("\nPreparing detailed category labels...", flush=True)

detail_mlb = MultiLabelBinarizer()

Y_detail_train = detail_mlb.fit_transform(detail_train)
Y_detail_test = detail_mlb.transform(detail_test)

print("Detailed categories:", len(detail_mlb.classes_), flush=True)

# Train main category model
print("\nTraining Main Category Classifier...", flush=True)

main_model = OneVsRestClassifier(
    LogisticRegression(
        solver="saga",
        max_iter=100,
        tol=1e-2,
        random_state=42
    ),
    n_jobs=2
)

main_model.fit(X_train_tfidf, Y_main_train)

# Train detailed category model
print("\nTraining Detailed Category Classifier...", flush=True)

detail_model = OneVsRestClassifier(
    LogisticRegression(
        solver="saga",
        max_iter=100,
        tol=1e-2,
        random_state=42
    ),
    n_jobs=2
)

detail_model.fit(X_train_tfidf, Y_detail_train)

# Evaluate main model
print("\nEvaluating Main Category Model...", flush=True)

main_predictions = main_model.predict(X_test_tfidf)

main_micro = f1_score(
    Y_main_test,
    main_predictions,
    average="micro",
    zero_division=0
)

main_macro = f1_score(
    Y_main_test,
    main_predictions,
    average="macro",
    zero_division=0
)

# Evaluate detailed model
print("\nEvaluating Detailed Category Model...", flush=True)

detail_predictions = detail_model.predict(X_test_tfidf)

detail_micro = f1_score(
    Y_detail_test,
    detail_predictions,
    average="micro",
    zero_division=0
)

detail_macro = f1_score(
    Y_detail_test,
    detail_predictions,
    average="macro",
    zero_division=0
)

# Display results
print("\n========== MODEL RESULTS ==========", flush=True)

print("Main Category Micro F1:", main_micro, flush=True)
print("Main Category Macro F1:", main_macro, flush=True)

print("Detailed Category Micro F1:", detail_micro, flush=True)
print("Detailed Category Macro F1:", detail_macro, flush=True)

# Save models separately
os.makedirs("models", exist_ok=True)

print("\nSaving improved models...", flush=True)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer_improved.pkl"
)

joblib.dump(
    main_model,
    "models/main_category_model_improved.pkl"
)

joblib.dump(
    main_mlb,
    "models/main_label_encoder_improved.pkl"
)

joblib.dump(
    detail_model,
    "models/detail_category_model_improved.pkl"
)

joblib.dump(
    detail_mlb,
    "models/detail_label_encoder_improved.pkl"
)

print("\nTRAINING COMPLETED!", flush=True)
print("All five improved model files saved.", flush=True)
print("Original models have not been overwritten.", flush=True)
