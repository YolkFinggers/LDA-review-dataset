import gradio as gr
import pandas as pd
from main import run_pipeline

def run_lda_app(file, num_topics=5, passes=10, verbose=False, visualize=True):
    # Save uploaded file to temp path
    file_path = "uploaded_file.csv"
    df = pd.read_csv(file.name)
    df.to_csv(file_path, index=False)

    # Run pipeline
    results = run_pipeline(
        csv_path=file_path,
        text_column="review_text",
        num_topics=num_topics,
        passes=passes,
        verbose=verbose,
        visualize=visualize
    )

    # Prepare topic display
    topics_str = "\n".join([f"Topic {i+1}: {t[1]}" for i, t in enumerate(results["topics"])])
    coherence_score = results["coherence"]

    # Return text summary + path to interactive HTML
    return f"✅ LDA Completed!\n\nCoherence Score: {coherence_score:.4f}\n\nTopics:\n{topics_str}", "lda_visualization.html"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📊 LDA Topic Modeling for Low-Rated Reviews")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV (must contain 'review_text')")
            num_topics_input = gr.Slider(2, 10, value=5, step=1, label="Number of Topics")
            passes_input = gr.Slider(1, 20, value=10, step=1, label="Number of Passes")
            verbose_input = gr.Checkbox(label="Verbose Output", value=True)
            visualize_input = gr.Checkbox(label="Generate Visualizations", value=True)
            run_button = gr.Button("Run LDA")
        with gr.Column():
            output_text = gr.Textbox(label="LDA Results", lines=20)
            vis_link = gr.File(label="Download Interactive Visualization")

    run_button.click(
        fn=run_lda_app,
        inputs=[file_input, num_topics_input, passes_input, verbose_input, visualize_input],
        outputs=[output_text, vis_link]
    )

demo.launch()
