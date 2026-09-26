!pip install -q biopython scikit-learn transformers sentencepiece sentence-transformers

import torch
from Bio import Entrez
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Exp 3: Biomedical RAG
class BiomedicalRAG:
    def __init__(self):
        Entrez.email = "ydskjfskjfksdjf@gmail.com"

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        self.name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.name, tie_word_embeddings=False
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def fetch(self, query, number=5):
        search = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=number
        )

        ids = Entrez.read(search)["IdList"]

        if not ids:
            return []

        result = Entrez.efetch(
            db="pubmed",
            id=ids,
            rettype="abstract",
            retmode="text"
        )

        return [
            x.strip()
            for x in result.read().split("\n\n")
            if len(x.strip()) > 100
        ][:number]

    def retrieve(self, question, documents, top_k=3):

        # Old:
        # vectorizer = TfidfVectorizer(stop_words="english")
        # document_vectors = vectorizer.fit_transform(documents)
        # question_vector = vectorizer.transform([question])
        # similarity = cosine_similarity(
        #     question_vector, document_vectors
        # )[0]

        document_vectors = self.encoder.encode(documents)
        question_vector = self.encoder.encode([question])

        similarity = cosine_similarity(
            question_vector, document_vectors
        )[0]

        indices = similarity.argsort()[-top_k:][::-1]

        return [documents[i] for i in indices]

    def generate(self, question, documents):
        context = "\n\n".join(documents)

        prompt = f"""
Use the following PubMed information to answer the biomedical question.

PubMed information:
{context}

Question:
{question}

Give a short and accurate answer based only on the provided information.
If the information is insufficient, say so.
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100
            )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

    def run(self):
        # Original question code retained
        question = input("Enter your biomedical question: ")

        documents = self.fetch(question)

        print("\nNumber of PubMed documents retrieved:", len(documents))

        if not documents:
            print("No PubMed documents found.")
            return

        relevant = self.retrieve(question, documents)

        print("\nRelevant PubMed information:")
        for i, doc in enumerate(relevant, 1):
            print(f"\nDocument {i}:")
            print(doc[:800])

        print("\nFINAL BIOMEDICAL ANSWER")
        print(self.generate(question, relevant))


BiomedicalRAG().run()