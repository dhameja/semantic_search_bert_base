📌 Overview

This project implements a Semantic Search API using the BERT Base model to find meaning-based similarity between user queries and sentences.

Unlike keyword search, semantic search understands context and intent, allowing similar meanings to match even when words differ.

Example:

Query: “AI intelligence”
Match: “Artificial intelligence powers modern applications”
🚀 Features
Semantic similarity using BERT embeddings
REST API built with FastAPI
Vector embedding generation
Cosine similarity scoring
Ranked search results
🧠 Model Used

Model: bert-base-uncased

Why BERT?

Captures contextual meaning of words
Produces dense vector embeddings
Strong baseline for NLP semantic tasks

Library used:

sentence-transformers
🏗️ Project Architecture
project/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── data.py                 # Sample sentences dataset
│   ├── embedding.py            # BERT embedding generation
│   ├── search.py               # Similarity + ranking logic
│   └── semantic_search_bert.py # Core semantic pipeline
│
└── README.md
⚙️ Installation
1️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
2️⃣ Install Dependencies
pip install fastapi uvicorn sentence-transformers scikit-learn
▶️ Running the API

Start the server:

uvicorn app.main:app --reload

Server runs at:

http://127.0.0.1:8000

Interactive API docs:

http://127.0.0.1:8000/docs
🔎 How Semantic Search Works
Step 1 — Input Sentences
sentences = [
    "Machine learning improves recommendation systems",
    "Gold and silver are precious metals",
    "Artificial intelligence powers modern applications",
    "News apps personalize articles for users",
    "Stock markets react to economic signals"
]
Step 2 — Embedding Generation

BERT converts text into vectors:

Text → Tokenization → BERT → Embedding Vector (768 dimensions)
Step 3 — Query Encoding

User query is converted into another embedding vector.

Step 4 — Similarity Calculation

Cosine similarity compares vectors:

Similarity = cosine(query_vector, sentence_vector)

Score Range:

1.0 → identical meaning
0.6–0.8 → strong similarity
0.3–0.5 → weak relation
0 → unrelated
📡 API Example
Request
GET /search?q=AI intelligence
Response
{
  "query": "AI intelligence",
  "results": [
    {
      "sentence": "Artificial intelligence powers modern applications",
      "score": 0.66
    }
  ]
}
🧮 Why Score ≈ 0.66?

BERT measures semantic closeness, not exact word overlap:

"AI" ≈ "Artificial Intelligence"
Related meaning → medium-high similarity score.
🛠️ Tech Stack
Python
FastAPI
Uvicorn
Sentence Transformers
BERT Base
Scikit-learn
📈 Future Improvements
Replace static sentences with database
Add news/article dataset
Store embeddings in vector DB (FAISS/Pinecone)
Add caching layer
Support batch search