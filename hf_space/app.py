import gradio as gr
from train_kannada_bpe import BPETokenizer

VOCAB_PATH = "vocab.json"
MERGES_PATH = "merges.json"

# Load tokenizer once at startup
tokenizer = BPETokenizer()
tokenizer.load(VOCAB_PATH, MERGES_PATH)


def encode_text(text: str):
    """
    Encode Kannada text into token IDs, and compute compression ratio.
    """
    text = text.strip()
    if not text:
        return [], 0, 0, 0.0

    token_ids = tokenizer.encode(text)
    num_tokens = len(token_ids)
    num_chars = len(text)
    ratio = num_chars / num_tokens if num_tokens > 0 else 0.0

    return token_ids, num_chars, num_tokens, round(ratio, 4)


def decode_ids(id_string: str):
    """
    Decode a comma-separated list of integers back to text.
    Example input: "12, 45, 78"
    """
    id_string = id_string.strip()
    if not id_string:
        return ""

    # Allow both comma-separated and space-separated lists
    # e.g., "12, 45, 78" or "12 45 78"
    raw = id_string.replace(",", " ")
    parts = [p for p in raw.split() if p]

    try:
        ids = [int(x) for x in parts]
    except ValueError:
        return "Error: please enter integers separated by commas or spaces."

    text = tokenizer.decode(ids)
    return text


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Kannada BPE Tokenizer Demo

        Custom BPE tokenizer trained on four Kannada novels(140k characters).

        - **Encode tab**: Type Kannada text → see token IDs and compression ratio  
        - **Decode tab**: Paste token IDs → get back (approximate) text
        """
    )

    with gr.Tab("Encode"):
        input_text = gr.Textbox(
            label="Input text (Kannada)",
            lines=4,
            placeholder="ನಮಸ್ಕಾರ. ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        )
        encode_button = gr.Button("Encode")

        token_ids_out = gr.JSON(label="Token IDs")
        num_chars_out = gr.Number(label="Number of characters", precision=0)
        num_tokens_out = gr.Number(label="Number of tokens", precision=0)
        ratio_out = gr.Number(
            label="Compression ratio (chars / tokens)",
            precision=4,
        )

        encode_button.click(
            fn=encode_text,
            inputs=[input_text],
            outputs=[token_ids_out, num_chars_out, num_tokens_out, ratio_out],
        )

    with gr.Tab("Decode"):
        gr.Markdown(
            "Enter token IDs (comma-separated or space-separated), "
            "for example: `12, 45, 78` or `12 45 78`."
        )
        id_input = gr.Textbox(
            label="Token IDs",
            lines=3,
            placeholder="e.g. 12, 45, 78",
        )
        decode_button = gr.Button("Decode")
        decoded_text = gr.Textbox(
            label="Decoded text",
            lines=4,
        )

        decode_button.click(
            fn=decode_ids,
            inputs=[id_input],
            outputs=[decoded_text],
        )

if __name__ == "__main__":
    demo.launch()

