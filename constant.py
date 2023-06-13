from chromadb.config import Settings

# Define the folder for storing the model
MODEL_DIRECTORY = "./models"

# model names
GPT4ALL = "gpt4all/ggml-gpt4all-j-v1.3-groovy.bin"
LLAMA_CPP = "llama_cpp/ggml-model-q4_0.bin"

# Define the folder for storing database
PERSIST_DIRECTORY = "./db"

# Define ChromaDB configuration
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    anonymized_telemetry=False
)

COLLECTION_NAME = "personal_doc_store"

# Supported File List
SUPPORTED_FILE_LIST = ["csv", "pdf", "docx", "txt"]
