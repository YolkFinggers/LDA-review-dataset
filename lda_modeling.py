from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from .utils import vprint

def run_lda(df, dictionary, corpus, num_topics=5, passes=10, verbose=False):
    vprint(f"🔹 Training LDA with {num_topics} topics...", verbose)
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha="auto",
        per_word_topics=True,
    )

    topics = lda_model.print_topics(num_words=10)
    vprint("✅ LDA training complete.", verbose)

    if verbose:
        for i, topic in enumerate(topics):
            print(f"Topic {i+1}: {topic[1]}")

    return lda_model, topics

def evaluate_model(lda_model, corpus, dictionary, df, verbose=False):
    vprint("🔹 Evaluating LDA model (Coherence Score)...", verbose)
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=df["tokens"],
        dictionary=dictionary,
        coherence="c_v"
    )
    coherence = coherence_model.get_coherence()
    vprint(f"✅ Coherence Score: {coherence:.4f}", verbose)
    return coherence
