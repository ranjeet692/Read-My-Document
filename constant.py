from chromadb.config import Settings

# Define the folder for storing database
PERSIST_DIRECTORY = "./test"

# Define ChromaDB configuration
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    anonymized_telemetry=False
)

COLLECTION_NAME = "test_collection"

# Supported File List
SUPPORTED_FILE_LIST = ["csv", "pdf", "docx", "txt"]
