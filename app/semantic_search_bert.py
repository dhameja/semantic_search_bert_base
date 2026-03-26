from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load BERT Base model
# -----------------------------
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# -----------------------------
# Sentences in system (database)
# -----------------------------
sentences = [
    "Machine learning improves recommendation systems",
    "Gold and silver are precious metals",
    "Artificial intelligence powers modern applications",
    "News apps personalize articles for users",
    "Stock markets react to economic signals"
]

# -----------------------------
# Function: Create embeddings
# -----------------------------
def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean Pooling (convert tokens → sentence vector)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.numpy()


# Create embeddings for database
sentence_embeddings = np.vstack(
    [get_embedding(s) for s in sentences]
)

# -----------------------------
# Search Query
# -----------------------------
query = "metal"

query_embedding = get_embedding(query)

# -----------------------------
# Compute similarity
# -----------------------------
scores = cosine_similarity(query_embedding, sentence_embeddings)[0]

# -----------------------------
# Show results
# -----------------------------
results = sorted(
    zip(sentences, scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop Matches:\n")

for sentence, score in results:
    print(f"{sentence}  --> score: {score:.4f}")