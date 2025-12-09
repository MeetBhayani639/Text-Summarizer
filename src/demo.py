import gradio as gr
from infer import Summarizer

summarizer = Summarizer()

def summarize_api(text, max_tokens):
    if not text.strip():
        return "Please enter some text."
    return summarizer.summarize(text, max_new_tokens=max_tokens)

with gr.Blocks() as demo:
    gr.Markdown("# 📝 Abstractive Text Summarizer")
    gr.Markdown("Enter an article or long text below and get a concise summary.")

    with gr.Row():
        input_text = gr.Textbox(
            lines=15,
            label="Input Text",
            placeholder="Paste article or paragraph here..."
        )
        output_text = gr.Textbox(
            lines=10,
            label="Generated Summary"
        )

    max_tokens = gr.Slider(
        minimum=32,
        maximum=256,
        value=128,
        step=8,
        label="Max Summary Tokens"
    )

    summarize_btn = gr.Button("Summarize")

    summarize_btn.click(
        summarize_api,
        inputs=[input_text, max_tokens],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.launch()
