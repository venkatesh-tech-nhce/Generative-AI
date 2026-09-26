!pip install -q transformers datasets rouge-score sentencepiece

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# Exp 1: Abstractive summarization using BART
class Summarizer:
    def __init__(self):
        self.dataset = load_dataset("S3IC/cnn_dailymail", split="train")
        self.name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def summarize(self, article):
        inputs = self.tokenizer(
            article, max_length=512, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=100,
                min_length=20,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def run(self, n=5):
        results = []

        for i in range(n):
            article = self.dataset[i]["article"]
            reference = self.dataset[i]["highlights"]
            summary = self.summarize(article)

            scores = self.scorer.score(reference, summary)

            row = {
                "Article_Number": i + 1,
                "Generated_Summary": summary,
                "Reference_Summary": reference,
                "ROUGE-1": scores["rouge1"].fmeasure,
                "ROUGE-2": scores["rouge2"].fmeasure,
                "ROUGE-L": scores["rougeL"].fmeasure
            }

            results.append(row)

            print(f"\nARTICLE {i + 1}")
            print("SUMMARY:", summary)
            print("ROUGE-1:", round(row["ROUGE-1"], 4))
            print("ROUGE-2:", round(row["ROUGE-2"], 4))
            print("ROUGE-L:", round(row["ROUGE-L"], 4))

        df = pd.DataFrame(results)
        print("\nAVERAGE ROUGE")
        print(df[["ROUGE-1", "ROUGE-2", "ROUGE-L"]].mean())

        df.to_csv("summarization_results.csv", index=False)

Summarizer().run()