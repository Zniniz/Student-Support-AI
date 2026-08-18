import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


KNOWLEDGE_BASE_FILE = "spam.csv"
LABEL_ORDER = ["spam", "ham"]


def loadKnowledgeBase(file):
    """Load, clean, and validate the spam dataset using pandas."""
    try:
        dataFrame = pd.read_csv(file, encoding="latin-1")
    except FileNotFoundError:
        print(f"File path not found: {file}")
        sys.exit(1)
    except pd.errors.ParserError as error:
        print("CSV format error. Ensure entries with commas are enclosed in quotes.")
        print(f"Details: {error}")
        sys.exit(1)

    if dataFrame.shape[1] < 2:
        print("CSV must contain at least two columns: label and message.")
        sys.exit(1)

    # The provided spam.csv has extra empty columns and a misspelled label header,
    # so we keep only the first two columns and rename them ourselves.
    dataFrame = dataFrame.iloc[:, :2].copy()
    dataFrame.columns = ["label", "message"]
    # Normalize casing/whitespace so "Spam", " spam", "SPAM " all match LABEL_ORDER.
    dataFrame["label"] = dataFrame["label"].astype(str).str.strip().str.lower()
    dataFrame["message"] = dataFrame["message"].fillna("").astype(str).str.strip()

    # Drop empty messages and any row whose label isn't exactly "spam"/"ham" —
    # bad rows would otherwise break TF-IDF fitting or skew the confusion matrix.
    dataFrame = dataFrame[dataFrame["message"] != ""]
    dataFrame = dataFrame[dataFrame["label"].isin(LABEL_ORDER)]

    if dataFrame.empty:
        print("No valid spam or ham messages were found in the dataset.")
        sys.exit(1)

    return dataFrame


def plotDistribution(df):
    """Show the number of spam and ham messages in the dataset."""
    counts = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    labels = ["Spam", "Ham"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts.values, color=["#d94f45", "#3578b8"])
    total = np.sum(counts.values)
    plt.title(f"Spam vs Ham Distribution (Total: {total} messages)")
    plt.xlabel("Class")
    plt.ylabel("Number of messages")

    for bar, count in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def featureExtraction(messages):
    """Convert email text into TF-IDF numerical features."""
    # stop_words="english" drops low-signal filler words (the, is, at...) so the
    # model focuses on words that actually distinguish spam from ham.
    # ngram_range=(1, 2) keeps single words AND two-word phrases (e.g. "free prize",
    # "claim now"), since spam is often identifiable by phrasing, not just one word.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(messages)
    return X, vectorizer


def trainModel(X_train, y_train):
    """Train a Logistic Regression classifier for spam detection."""
    # class_weight="balanced" matters here because the dataset is imbalanced
    # (about 87% ham vs 13% spam). Without it, the model could lean toward
    # predicting "ham" and still post a high accuracy while missing real spam.
    # max_iter is raised from sklearn's default (100) since TF-IDF produces a
    # large, sparse feature space that can need more iterations to converge.
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def evaluateModel(model, X_test, y_test):
    """Print accuracy and a correctly ordered confusion matrix."""
    prediction = model.predict(X_test)
    score = accuracy_score(y_test, prediction)
    # labels=LABEL_ORDER forces the matrix into [spam, ham] row/column order.
    # Without this, confusion_matrix() defaults to alphabetical order (ham, spam),
    # which would silently mislabel the table printed below.
    confusion = confusion_matrix(y_test, prediction, labels=LABEL_ORDER)

    print(f"Accuracy: {score * 100:.2f}%")
    print("Confusion Matrix:")
    print("\t\tPredicted")
    print("\t\tSpam\tHam")
    print(f"Actual Spam\t{confusion[0][0]}\t{confusion[0][1]}")
    print(f"Actual Ham\t{confusion[1][0]}\t{confusion[1][1]}")

    return confusion


def plotConfusionHeatmap(confusion):
    """Show the confusion matrix as a seaborn heatmap (visual companion to the printed table)."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Spam", "Ham"],
        yticklabels=["Spam", "Ham"],
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def predictMessage(text, model, vectorizer):
    """Predict whether one message is spam or ham and return confidence."""
    features = vectorizer.transform([text])
    label = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    return label, confidence


def runLoop(model, vectorizer):
    """Continuously classify user messages until the user types quit."""
    while True:
        email = input("Enter message: ").strip()

        if email.lower() == "quit":
            print("Goodbye!")
            break

        if email == "":
            print("Please enter a message or type 'quit' to exit.")
            continue

        label, confidence = predictMessage(email, model, vectorizer)
        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {confidence * 100:.1f}%")


class SpamDetector:
    """Train, evaluate, visualize, and run the spam detector."""

    def __init__(self, file):
        self.df = loadKnowledgeBase(file)
        self.X, self.vectorizer = featureExtraction(self.df["message"])
        self.y = self.df["label"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            # stratify=self.y keeps the same spam/ham ratio in both the train and
            # test sets. Without it, a random split could leave the test set with
            # too few spam examples to evaluate the model meaningfully.
            stratify=self.y,
        )
        self.model = trainModel(self.X_train, self.y_train)

    def showChart(self):
        plotDistribution(self.df)

    def evaluate(self):
        self.confusion = evaluateModel(self.model, self.X_test, self.y_test)

    def showHeatmap(self):
        plotConfusionHeatmap(self.confusion)

    def run(self):
        runLoop(self.model, self.vectorizer)


def main():
    print("Welcome to Spam Detection AI")
    print("Training model...")
    detector = SpamDetector(KNOWLEDGE_BASE_FILE)
    detector.evaluate()
    detector.showChart()
    detector.showHeatmap()
    print("Type 'quit' to exit.")
    detector.run()


if __name__ == "__main__":
    main()