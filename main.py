from load_corpus import load_corpus
from pathlib import Path
import numpy as np

def main():
    file_dir = Path(__file__).resolve().parent
    text8_path = file_dir / "text8"
    
    vocab, corpus_ids = load_corpus(text8_path)
    print("tokens start: ", vocab[:10])
    print("beginning of text encodings: ", corpus_ids[:10])
    
    D = len(vocab)
    d = 100 # hyperparameter for the embedding size. looking at the original Mikolov et al. paper, after 100 we get diminishing returns (especially for smaller datasets like i am using) so to keep computation fast, i will go with 100

    # i initialize the matrices with some noise for symmetry-breaking reasons
    rng = np.random.default_rng(42) 
    V = rng.normal(0, 0.01, size=(D, d))
    V_prime = rng.normal(0, 0.01, size=(d, D))

if __name__ == "__main__":
    main()
