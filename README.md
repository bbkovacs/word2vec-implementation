To run the whole training from the beginning:
- download the zip from https://mattmahoney.net/dc/text8.zip
- extract it, and put the text8 file into the root of this directory
- `uv run main.py`

The embeddings are stored in the output.npz file. To simply use the (more heuristical, explicitly not rigorous) evaluation:
- `uv run evaluation.py`
- the file can be changed to inspect the nearest neighbors of different words, or combinations of words 
Current results are subpar, as the current output.npz has only been trained on the first 10^6 words.

The main training loop is found in the training.py file. 

Non-trivial decisions I made:
- i am using the CBOW objective instead of the skip-gram
- text8 as dataset (small and pre-cleaned)
- exclude rare words (<5 occurences) from vocab (chosen arbitrarily)
- d (embedding size) set to 100 (see Mikolov paper)
- initialization of V and V' with small random numbers (normal(0, 0.01)) (symmetry breaking)
- context window of 5 (see Google blogpost)
- lr = 0.05, epochs = 1 (chosen arbitrarily, epoch numbers chosen to keep training time down)
- i use standard SGD optimization (ease of implementation, light in memory)
- i only use 10^6 words from text8 as the corpus to speed up computation (right now i do not do negative sampling, which should be a speedup)

Disclaimer: other than numpy and standard libraries, I used tqdm, as this does not help the core ML functionality (just makes the training output nicer). This library can be removed from the code without hurting functionality.