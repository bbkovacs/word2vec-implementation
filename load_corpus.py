from collections import Counter
import numpy as np

#we return the list of all different words (vocab), and an "id"-d version of the corpus (essentially one-hot encodings)
def load_corpus(text8_path):
    # i am using text8 from https://mattmahoney.net/dc/textdata.html
    # since this is already cleaned etc, there is not much happening here. 
    # in case this was a more "realistic" corpus, we would need more involved cleaning/tokenization too.
    with open(text8_path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
        
    # optimization: exclude rare words from vocabulary (smaller matrices, faster softmax)
    word_counts = Counter(tokens)
    min_count = 5 # this is a hyperparameter, i set it arbitrarily
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    vocab = sorted(vocab)
    vocab = ["<UNK>"] + vocab
    
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    corpus_ids = np.asarray([word_to_id.get(word, 0) for word in tokens], dtype=np.int32) #rare words become index 0, <UNK>
    
    return vocab, corpus_ids, id_to_word