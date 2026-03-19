import numpy as np
from tqdm import tqdm

def softmax(x):
    x = x - np.max(x) #to prevent overflow
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

def training(vocab, corpus_ids, id_to_word):
    D = len(vocab)
    d = 20 # hyperparameter for the embedding size. looking at the original Mikolov et al. paper (https://arxiv.org/pdf/1301.3781), after 100 we get diminishing returns (especially for smaller datasets like i am using) so to keep computation fast, i will go with 100
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
            target_id = corpus_ids[i]
            left_context_ids = corpus_ids[i-window:i]
            right_context_ids = corpus_ids[i+1:i+window+1]
            context_ids = left_context_ids + right_context_ids
            
            # forward pass
            h = V[context_ids].mean(axis=0)
            logits = h @ V_prime
            probs = softmax(logits)
            
            # loss
            loss = -np.log(probs[target_id] + 1e-12) #small value added to avoid log(0)
            total_loss += loss
            
            #backward pass
            grad_logits = probs.copy()
            grad_logits[target_id] -= 1
            grad_V_prime = np.outer(h, grad_logits)
            grad_h = V_prime @ grad_logits
            
            grad_context = grad_h / len(context_ids)
            
            #SGD update
            V_prime -= lr * grad_V_prime 
            for ctx_id in context_ids: 
                V[ctx_id] -= lr * grad_context
            
        print(f"epoch {epoch+1}, avg loss = {total_loss / (len(corpus_ids) - 2*window):.4f}")
    
    return V, V_prime