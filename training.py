import numpy as np
from tqdm import tqdm

def softmax(x):
    x = x - np.max(x) #to prevent overflow
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

def training(vocab, corpus_ids):
    D = len(vocab)
    d = 100 # hyperparameter for the embedding size. looking at the original Mikolov et al. paper (https://arxiv.org/pdf/1301.3781), after 100 we get diminishing returns (especially for smaller datasets like i am using) so to keep computation fast, i will go with 100
    window = 5 # hyperparameter for context window. they say that 5 usually works best with CBOW: https://code.google.com/archive/p/word2vec/ 
    lr = 0.05 #hyperparameter for learning rate - chosen arbitrarily
    epochs = 3 #hyperparameter for epochs - chosen arbitrarily
    
    # i initialize the matrices with some noise for symmetry-breaking reasons
    rng = np.random.default_rng(42) 
    V = rng.normal(0, 0.01, size=(D, d)).astype(np.float32)
    V_prime = rng.normal(0, 0.01, size=(d, D)).astype(np.float32)   
    
    num_examples = len(corpus_ids) - 2 * window
    
    print("Starting to load examples.")
    
    examples = []
    for i in tqdm(range(window, len(corpus_ids) - window)):
        target_id = corpus_ids[i]
        context_ids = np.concatenate((corpus_ids[i-window:i], corpus_ids[i+1:i+window+1]))
        examples.append((context_ids, target_id))
        
    print("Done loading examples.")
    
    #training loop - for each epoch, we train on each example
    for epoch in range(epochs):
        total_loss = 0

        for i in tqdm(range(len(examples))):
            context_ids = examples[i][0]
            target_id = examples[i][1]
            
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
            np.add.at(V, context_ids, -lr * grad_context)
            
        print(f"epoch {epoch+1}, avg loss = {total_loss / num_examples:.4f}")
    
    return V, V_prime