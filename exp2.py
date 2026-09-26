!pip install -q transformers pandas

import pandas as pd
from transformers import pipeline

# Exp 2: Sentiment analysis using DistilBERT
class SentimentAnalyzer:
    def __init__(self, path="/content/IMDB-Dataset.csv"):
        self.df = pd.read_csv(path, engine="python", on_bad_lines="skip")

        self.model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def run(self, n=5):
        print("Dataset loaded successfully!")
        print("Number of reviews:", len(self.df))

        for i in range(n):
            review = self.df.iloc[i]["review"]

            # Old: prediction = self.model(review[:512])[0]
            prediction = self.model(
                review,
                truncation=True,
                max_length=512
            )[0]

            print(f"\nReview {i + 1}")
            print("Actual:", self.df.iloc[i]["sentiment"].upper())
            print("Predicted:", prediction["label"])
            print("Confidence:", round(prediction["score"], 4))
            print("Review:", review[:200], "...")

SentimentAnalyzer().run()