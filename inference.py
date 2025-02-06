from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the Vyakyarth model from Hugging Face Hub
model = SentenceTransformer("krutrim-ai-labs/vyakyarth")

# Define a set of multilingual sentences for similarity evaluation
sentences = [
    "मैं अपने दोस्त से मिला",  # Hindi: "I met my friend"
    "I met my friend",         # English equivalent
    "I love you"              # Unrelated sentence
]

# Generate embeddings for the given sentences
embeddings = np.array(model.encode(sentences))

# Compute cosine similarity between semantically similar sentences
similarity_1 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity Score (Hindi-English translation): {similarity_1:.6f}")  
# Expected high similarity: ~0.98

# Compute cosine similarity between unrelated sentences
similarity_2 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
print(f"Similarity Score (Unrelated sentences): {similarity_2:.6f}")  
# Expected lower similarity
