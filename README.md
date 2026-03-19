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
- DONE: create training process (generic loop)
- step 4: create input * V * V' + softmax structure
- step 5: do SGD based (cross entropy loss)
- step 6: get embeddings
- step 7: evaluate embeddings

Non-trivial decisions I made:
- text8 as dataset (small and pre-cleaned)
- exclude rare words (<5 occurences) from vocab (chosen arbitrarily)
- d (embedding size) set to 100 (see Mikolov paper)
- initialization of V and V' with small random numbers (normal(0, 0.01)) (symmetry breaking)
- context window of 5 (see Google blogpost)
- lr = 0.05, epochs = 3 (chosen arbitrarily)

Disclaimer: other than numpy and standard libraries, I used tqdm, as this does not help the core ML functionality (just makes the training output nicer). This library can be removed from the code without hurting functionality.