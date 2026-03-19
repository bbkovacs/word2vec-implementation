To run the whole training loop:
- download the zip from https://mattmahoney.net/dc/text8.zip
- extract it, and put the text8 file into the root of this directory
- `uv run main.py`

TASK:

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

Plan:
- i will use text8 as the dataset from https://mattmahoney.net/dc/textdata.html
- i will implement CBOW variant (it is more straightforward/clean to implement first; but s-g w ns might perform better, especially on rare words)
- DONE: read in dataset, tokenize
- DONE: create matrices V, V'
- step 3: create training process (generic loop)
- step 4: create input * V * V' + softmax structure
- step 5: do SGD based (cross entropy loss)
- step 6: get embeddings
- step 7: evaluate embeddings

Non-trivial decisions I made:
- text8 as dataset
- exclude rare words (<5 occurences) from vocab
- d (embedding size) set to 100
- initialization of V and V' with small random numbers (normal(0, 0.01)) 

Small notes:
- i will use a context window of 5 (https://code.google.com/archive/p/word2vec/ says this performs usually the best, and hyperparam tuning is not the point rn)