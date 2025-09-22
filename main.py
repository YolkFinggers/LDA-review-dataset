import pandas as pd
from .data_cleaning import clean_text
from .data_analysis import build_corpus
from .lda_modeling import run_lda, evaluate_model
from .visualization import plot_wordcloud, interactive_viz
from .utils import vprint

def run_pipeline(
    csv_path,
    text_column="review_text",
    num_topics=5,
    passes=10,
    verbose=False,
    visualize=True
):
    vprint("🚀 Starting LDA pipeline...", verbose)

    # 1. Load Data
    df = pd.read_csv(csv_path)
    vprint(f"✅ Loaded dataset with {len(df)} rows.", verbose)

    # 2. Clean
    df = clean_text(df, text_column=text_column, verbose=verbose)

    # 3. Build Corpus
    dictionary, corpus = build_corpus(df, verbose=verbose)

    # 4. Run LDA
    lda_model, topics = run_lda(df, dictionary, corpus, num_topics, passes, verbose=verbose)

    # 5. Evaluate
    coherence = evaluate_model(lda_model, corpus, dictionary, df, verbose=verbose)

    # 6. Visualization
    if visualize:
        plot_wordcloud(lda_model, num_topics, verbose=verbose)
        vis = interactive_viz(lda_model, corpus, dictionary, verbose=verbose)
        pyLDAvis.save_html(vis, "lda_visualization.html")
        vprint("✅ Interactive visualization saved: lda_visualization.html", verbose)

    return {
        "model": lda_model,
        "topics": topics,
        "coherence": coherence
    }
