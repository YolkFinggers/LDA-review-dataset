import matplotlib.pyplot as plt
import pyLDAvis.gensim_models
import pyLDAvis

from utils import vprint

def plot_wordcloud(lda_model, num_topics=5, verbose=False):
    """Matplotlib quick word visualization"""
    from wordcloud import WordCloud

    vprint("🔹 Plotting topic word clouds...", verbose)
    fig, axes = plt.subplots(1, num_topics, figsize=(20, 10))
    for i in range(num_topics):
        topic_terms = dict(lda_model.show_topic(i, 30))
        wc = WordCloud(width=800, height=600, background_color="white")
        wc.generate_from_frequencies(topic_terms)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Topic {i+1}")
        axes[i].axis("off")
    plt.show()

def interactive_viz(lda_model, corpus, dictionary, verbose=False):
    """PyLDAvis interactive visualization"""
    vprint("🔹 Preparing interactive visualization...", verbose)
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    return vis
