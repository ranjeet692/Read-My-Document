from pypdf import PdfReader
from io import BytesIO
from typing import List
import docx2txt
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, VectorStore
from openai.error import AuthenticationError, OpenAIError
from constant import COLLECTION_NAME, PERSIST_DIRECTORY

# Parse a pdf file and extract the text from it
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    pages = []
    print(pdf.pages)
    for page in pdf.pages:
        text = page.extract_text()
        pages.append(text)
    return pages

# Parse a docx file and extract the text from it
def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    return text

# Parse a csv file and extract the text from it
def parse_csv(file: BytesIO) -> str:
    return file.read().decode("utf-8")

# Parse a txt file and extract the text from it
def parse_txt(file: BytesIO) -> List[Document]:
    """Parse a txt file and extract the text from it"""
    #loader = UnstructuredFileLoader(file)
    #return loader.load()
    return file.read().decode("utf-8")

# convert text to document
def text_to_document(text: str | List[str]) -> List[Document]:
    """Converts text or list of text into document chunks"""
    if isinstance(text, str):
        text = [text]

    doc_chunks = []
    #split document into chunks
    for i, doc in enumerate(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            separators=["\n", "\r\n", "\r"], 
            chunk_overlap=0
        )

        chunks = text_splitter.split_text(doc)
        
        for j, chunk in enumerate(chunks):
            document = Document(
                page_content = chunk,
                metadata = {
                    "page": i, 
                    "chunk": i,
                    "source": f"{i}-{j}"
                }
            )
            doc_chunks.append(document)
    return doc_chunks

#Embed document chucks using OpenAI API
def embed_document_chunks_openai(doc_chunks: List[Document]) -> VectorStore:
    """Embed document chunks using OpenAI API and return ChromaDB index"""
    try:
        embedding_llm = OpenAIEmbeddings()
    except AuthenticationError as e:
        print(e)
        return "Open AI Exception"

    #store into vector store
    vector_store = Chroma.from_documents(
        documents = doc_chunks, 
        embedding = embedding_llm, 
        persist_directory = PERSIST_DIRECTORY,
        collection_name =COLLECTION_NAME
    )
    return vector_store


#Embed document chunk with local model
def embed_document_chunks_local(doc_chunks: List[Document]) -> VectorStore:
    """Embed document chunks using local model and return ChromaDB index"""
    embedding = HuggingFaceEmbeddings()
    vector_store = Chroma.from_documents(
        documents = doc_chunks, 
        embedding = embedding, 
        persist_directory = PERSIST_DIRECTORY,
        collection_name =COLLECTION_NAME
    )
    return vector_store

# Parse a file and extract the text from it
def parse_file(file: BytesIO, file_type: str) -> str | List[str]:
    if file.name.endswith(".pdf"):
        return parse_pdf(file)
    elif file.name.endswith(".docx"):
        return parse_docx(file)
    elif file.name.endswith(".csv"):
        return parse_csv(file)
    elif file.name.endswith(".csv"):
        return parse_txt(file)
    else:
        raise Exception("File type not supported")  # noqa: E501

# Embed document chunks using model selected by user    
def embed_and_store_text(text: str | List[str], model: str) -> VectorStore:
    """Processes text or list of text into document chunks, embeds them, and returns a vector store"""
    doc_chunks = text_to_document(text)

    #embed document chunks
    if model.startswith("Open AI"):
        return embed_document_chunks_openai(doc_chunks)
    elif model.startswith("Local"):
        return embed_document_chunks_local(doc_chunks)
    else:
        raise Exception("Model not supported")  # noqa: E501