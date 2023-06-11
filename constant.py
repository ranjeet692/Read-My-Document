from chromadb.config import Settings

# Define the folder for storing the model
MODEL_DIRECTORY = "./models"

# model names
GPT4ALL = "gpt4all"
LLAMA_CPP = "llama_cpp/llama_cpp.bin"

# Define the folder for storing database
PERSIST_DIRECTORY = "./db"

# Define ChromaDB configuration
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

COLLECTION_NAME = "ipl_2023"
