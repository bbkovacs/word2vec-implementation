from load_corpus import load_corpus
from training import training
from pathlib import Path
import numpy as np
import json

def main():
    file_dir = Path(__file__).resolve().parent
    text8_path = file_dir / "text8"
    output_dir = file_dir / "output.npz"
    
    vocab, corpus_ids, id_to_word = load_corpus(text8_path)
    print("tokens start: ", vocab[:10])
    print("beginning of text encodings: ", corpus_ids[:10])
    
    V, V_prime = training(vocab, corpus_ids, id_to_word)
    
    np.savez_compressed(
        output_dir,
        V=V,
        V_prime=V_prime,
        vocab=np.array(vocab, dtype=object)
    )
    print("DONE! embeddings saved!")

if __name__ == "__main__":
    main()
