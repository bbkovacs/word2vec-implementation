To run the whole training from the beginning:
- download the zip from https://mattmahoney.net/dc/text8.zip
- extract it, and put the text8 file into the root of this directory
- `uv run main.py`

The main training loop is found in the training.py file. 

Plan:
- i will use text8 as the dataset from https://mattmahoney.net/dc/textdata.html
- i will implement CBOW variant (it is more straightforward/clean to implement first; but s-g w ns might perform better, especially on rare words)
- DONE: read in dataset, tokenize
- DONE: create matrices V, V'
- DONE: create training process (generic loop)
- DONE: forward pass, calculate loss
- DONE: do backwards pass, update with SGD
- DONE: get embeddings
- step 7: evaluate embeddings

Non-trivial decisions I made:
- text8 as dataset (small and pre-cleaned)
- exclude rare words (<5 occurences) from vocab (chosen arbitrarily)
- d (embedding size) set to 100 (see Mikolov paper)
- initialization of V and V' with small random numbers (normal(0, 0.01)) (symmetry breaking)
- context window of 5 (see Google blogpost)
- lr = 0.05, epochs = 3 (chosen arbitrarily)
- i use standard SGD optimization (ease of implementation, light in memory)

Disclaimer: other than numpy and standard libraries, I used tqdm, as this does not help the core ML functionality (just makes the training output nicer). This library can be removed from the code without hurting functionality.