def get_tokens():
    from pathlib import Path
    file_dir = Path(__file__).resolve().parent
    text8_path = file_dir / "text8"

    # i am using text8 from https://mattmahoney.net/dc/textdata.html
    # since this is already cleaned etc, there is not much happening here. 
    # in case this was a more "realistic" corpus, we would need more involved cleaning/tokenization too.
    with open(text8_path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    
    return tokens