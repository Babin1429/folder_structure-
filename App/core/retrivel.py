import json
from App.core.vector_storage import VectorStorage
from App.core.embeddings import get_embeddings
from App.core.config import DATA_PATH
with open(DATA_PATH, 'r') as f:
    faqs = json.load(f)

vector_storage = VectorStorage(dim=384)

def retrieve(query, top_k=5):
    query_vector = get_embeddings(query)
    indices = vector_storage.search(query_vector, top_k)

    results = [faqs[i] for i in indices]
    return results
