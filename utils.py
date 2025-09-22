import spacy
from spacy.util import is_package
import subprocess
import sys

MODEL_NAME = "en_core_web_sm"

# Check if model is already installed as a package
if not is_package(MODEL_NAME):
    print(f"Downloading spaCy model {MODEL_NAME}...")
    subprocess.run([sys.executable, "-m", "spacy", "download", MODEL_NAME])

# Load the model
nlp = spacy.load(MODEL_NAME)


def vprint(msg, verbose=True):
    """Verbose print helper"""
    if verbose:
        print(msg)
