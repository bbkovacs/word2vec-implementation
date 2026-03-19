from pathlib import Path
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def nearest_neighbors(query_id, V, id_to_word, k=10, query_vec=None):
    if query_vec is None:
        query_vec = V[query_id]

    norms = np.linalg.norm(V, axis=1) + 1e-12
    sims = (V @ query_vec) / (norms * (np.linalg.norm(query_vec) + 1e-12))

    # exclude the word itself
    sims[query_id] = -np.inf

    top_ids = np.argsort(sims)[-k:][::-1]
    return [(id_to_word[i], sims[i]) for i in top_ids]

def evaluation():
    file_dir = Path(__file__).resolve().parent
    path = file_dir / "output.npz"
    data = np.load(path, allow_pickle=True)

    V = data["V"]
    V_prime = data["V_prime"]
    vocab = data["vocab"].tolist()

    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for i, word in enumerate(vocab)}
    
    print("Nearest neighbor evaluation `king`:")
    query = word_to_id["king"]
    print(nearest_neighbors(query, V, id_to_word))
    
    print("Nearest neighbor for analogy `king`-`man`+`woman`: ")
    query_vec = V[word_to_id["king"]] - V[word_to_id["man"]] + V[word_to_id["woman"]]
    print(nearest_neighbors(0, V, id_to_word, query_vec=query_vec))
    
if __name__ == "__main__":
    evaluation()