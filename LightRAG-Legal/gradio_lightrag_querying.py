import gradio as gr
import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import pdfplumber

# Define constants and initialize
WORKING_DIR = "./Legal_Documents"
OUTPUTS_DIR = os.path.join(WORKING_DIR, "../outputs/")
# PDF_PATH = "../Constituion-of-India.pdf"
output_file = os.path.join(OUTPUTS_DIR, "output_queries_2.txt")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# Initialize LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    chunk_token_size=1200,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.1:latest",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(texts, embed_model="nomic-embed-text", host="http://localhost:11434"),
    ),
)


def query_rag(input_text, mode):
    try:
        result = rag.query(input_text, param=QueryParam(mode=mode))
        logs = f"Query executed successfully in mode '{mode}'"
    except Exception as e:
        # Catch exceptions and log errors
        result = "An error occurred during the query execution."
        logs = f"Error: {e}"
    return result, logs


# Code snippet to display
code_text = """
LightRAG(
    working_dir=WORKING_DIR,
    chunk_token_size=1200,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.1:latest",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(texts, embed_model="nomic-embed-text", host="http://localhost:11434"),
    ),
)
"""

# Gradio layout
# Define Gradio interface layout
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='color: white; text-align: center;'>LightRAG Gradio Demo</h1>")
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="Enter your query")
            mode_dropdown = gr.Dropdown(choices=["naive", "local", "global", "hybrid"], label="Select Query Mode")
            submit_button = gr.Button("Submit")
        with gr.Column(scale=2):
            result_output = gr.Textbox(label="LLM Response", lines=20, interactive=True)
            logs_output = gr.Textbox(label="Terminal Logs", lines=10, interactive=True)
        # Link button click to the query function
        submit_button.click(query_rag, inputs=[query_input, mode_dropdown], outputs=[result_output, logs_output])

# Launch Gradio interface
demo.launch()

