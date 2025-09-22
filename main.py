import gradio as gr
from lda_pipeline.main import run_pipeline, run_advanced_analysis

def run_lda_app(file, num_topics=5, passes=10, verbose=False, visualize=True, advanced=False):
    file_path = "uploaded_file.csv"
    df = pd.read_csv(file.name)
    df.to_csv(file_path, index=False)

    # Run main LDA pipeline
    results = run_pipeline(
        csv_path=file_path,
        text_column="review_text",
        num_topics=num_topics,
        passes=passes,
        verbose=verbose,
        visualize=visualize
    )

    topics_str = "\n".join([f"Topic {i+1}: {t[1]}" for i, t in enumerate(results["topics"])])
    coherence_score = results["coherence"]

    output_text = f"✅ LDA Completed!\n\nCoherence Score: {coherence_score:.4f}\n\nTopics:\n{topics_str}"

    # Run optional advanced analysis
    adv_files = None
    if advanced:
        adv_results = run_advanced_analysis(df)
        output_text += "\n\n📈 Advanced Analysis Completed! Check generated files."
        adv_files = adv_results  # could be a list of CSVs or plots

    return output_text, "lda_visualization.html", adv_files

with gr.Blocks() as demo:
    gr.Markdown("## 📊 LDA Topic Modeling for Low-Rated Reviews")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV (must contain 'review_text')")
            num_topics_input = gr.Slider(2, 10, value=5, step=1, label="Number of Topics")
            passes_input = gr.Slider(1, 20, value=10, step=1, label="Number of Passes")
            verbose_input = gr.Checkbox(label="Verbose Output", value=True)
            visualize_input = gr.Checkbox(label="Generate Visualizations", value=True)
            advanced_input = gr.Checkbox(label="Run Advanced Analysis (Optional)", value=False)
            run_button = gr.Button("Run LDA")
        with gr.Column():
            output_text = gr.Textbox(label="LDA Results", lines=20)
            vis_link = gr.File(label="Download Interactive Visualization")
            adv_files = gr.File(label="Download Advanced Analysis Files", file_types=[".csv", ".html"], interactive=True)

    run_button.click(
        fn=run_lda_app,
        inputs=[file_input, num_topics_input, passes_input, verbose_input, visualize_input, advanced_input],
        outputs=[output_text, vis_link, adv_files]
    )

demo.launch()
