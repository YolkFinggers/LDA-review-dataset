from gensim import corpora
from utils import vprint

def build_corpus(df, verbose=False):
    vprint("🔹 Building dictionary and corpus...", verbose)
    dictionary = corpora.Dictionary(df["tokens"])
    corpus = [dictionary.doc2bow(text) for text in df["tokens"]]
    vprint(f"✅ Dictionary size: {len(dictionary)}", verbose)
    return dictionary, corpus
