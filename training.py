import numpy as np
from tqdm import tqdm

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def training(vocab, corpus_ids, ):
    D = len(vocab)
    d = 100 # hyperparameter for the embedding size. looking at the original Mikolov et al. paper (https://arxiv.org/pdf/1301.3781), after 100 we get diminishing returns (especially for smaller datasets like i am using) so to keep computation fast, i will go with 100
    window = 5 # hyperparameter for context window. they say that 5 usually works best with CBOW: https://code.google.com/archive/p/word2vec/ 
    lr = 0.05 #hyperparameter for learning rate - chosen arbitrarily
    epochs = 3 #hyperparameter for epochs - chosen arbitrarily
    
    # i initialize the matrices with some noise for symmetry-breaking reasons
    rng = np.random.default_rng(42) 
    V = rng.normal(0, 0.01, size=(D, d))
    V_prime = rng.normal(0, 0.01, size=(d, D))
    
    #training loop - for each epoch, we train on each example
    for epoch in range(epochs):
        total_loss = 0
        
        for i in tqdm(range(window, len(corpus_ids) - window)):
            pass
            
        print(f"epoch {epoch+1}, avg loss = {total_loss / (len(corpus_ids) - 2*window):.4f}")
    
    return V, V_prime