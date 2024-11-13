import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import pdfplumber


WORKING_DIR = "./Legal_Documents"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

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


pdf_path = "../Constituion-of-India.pdf"

pdf_text = ""

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        pdf_text += page.extract_text() + "\n"

# rag.insert(pdf_text)


# Define output file path
os.makedirs(
    os.path.join(WORKING_DIR, "../outputs/"),
    exist_ok=True,
)
output_file = os.path.join(WORKING_DIR, "../outputs/output_queries_2.txt")


# Function to write results to file
def write_to_file(output_text):
    with open(output_file, "a", encoding="utf-8") as file:
        file.write(output_text + "\n")


# Perform searches and save results
write_to_file("------------------------------------------Naive--------------------------------------------------------")
write_to_file(
    rag.query(
        "What does companies act mean?",
        param=QueryParam(mode="naive"),
    )
)

print("\033[92mNaive - Done ✔\033[0m")

write_to_file("------------------------------------------Local------------------------------------------------------")
write_to_file(
    rag.query(
        "What does companies act mean?",
        param=QueryParam(mode="local"),
    )
)

print("\033[92mLocal - Done ✔\033[0m")

write_to_file("------------------------------------------Global----------------------------------------------------")
write_to_file(
    rag.query(
        "What does companies act mean?",
        param=QueryParam(mode="global"),
    )
)

print("\033[92mGlobal - Done ✔\033[0m")

write_to_file("------------------------------------------Hybrid------------------------------------------------")
write_to_file(
    rag.query(
        "What does companies act mean?",
        param=QueryParam(mode="hybrid"),
    )
)

print("\033[92mHybrid - Done ✔\033[0m")

# Output confirmation with a green checkmark
print("\033[92mAll Queries Completed ✔✔✔\033[0m")
