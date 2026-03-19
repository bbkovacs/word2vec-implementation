from load_corpus import load_corpus
from training import training
from pathlib import Path
import numpy as np

def main():
    file_dir = Path(__file__).resolve().parent
    text8_path = file_dir / "text8"
    
    vocab, corpus_ids = load_corpus(text8_path)
    print("tokens start: ", vocab[:10])
    print("beginning of text encodings: ", corpus_ids[:10])
    
    V = training(vocab, corpus_ids)
    
    return V
    

if __name__ == "__main__":
    main()
