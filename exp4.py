!pip install -q faiss-cpu sentence-transformers transformers sentencepiece

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Exp 4: Legal QA using FAISS
class LegalQA:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Old:
        # self.generator = pipeline(
        #     "text-generation",
        #     model="gpt2"
        # )

        self.name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            self.name
        )

        self.docs = [
            "Case 1: A contract dispute arose between two parties. "
            "The court held that the agreement was valid because both parties "
            "had clearly accepted its terms.",

            "Case 2: A person was accused of negligence after causing an accident. "
            "The court explained that negligence requires a duty of care, "
            "a breach of that duty, and resulting harm.",

            "Case 3: A tenant challenged an eviction notice. "
            "The court held that the landlord must follow the applicable "
            "legal procedure before evicting the tenant.",

            "Case 4: A company used another company's trademark without permission. "
            "The court held that unauthorized use of a protected trademark "
            "could amount to infringement."
        ]

        self.build_index()

    def build_index(self):
        embeddings = self.encoder.encode(
            self.docs
        ).astype("float32")

        self.index = faiss.IndexFlatL2(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

    def retrieve(self, question, k=2):
        embedding = self.encoder.encode(
            [question]
        ).astype("float32")

        _, indices = self.index.search(
            embedding, k
        )

        return [
            self.docs[i]
            for i in indices[0]
        ]

    def answer(self, question, documents, level):
        context = "\n".join(documents)

        if level == "1":
            instruction = "Answer in 1 or 2 sentences."
        elif level == "2":
            instruction = "Give a short paragraph explaining the main facts and decision."
        else:
            instruction = (
                "Give a detailed explanation of the facts, "
                "legal issue, and court decision."
            )

        prompt = f"""
You are a legal document analysis assistant.

Use ONLY the retrieved legal content.

Retrieved Legal Content:
{context}

Question:
{question}

Instructions:
{instruction}

Do not invent facts.
If the information is insufficient, say so.
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            output = self.generator.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

    def run(self):
        question = input("Enter your legal question: ")

        documents = self.retrieve(question)

        print("\nRetrieved Legal Content:")
        print("\n".join(documents))

        print("\nChoose summary level:")
        print("1. Brief")
        print("2. Moderate")
        print("3. Detailed")

        level = input("Enter choice: ")

        print("\nFinal Answer:")
        print(self.answer(question, documents, level))


import torch
LegalQA().run()