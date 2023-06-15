import os
from io import BytesIO
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import OpenAI, GPT4All
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from constant import COLLECTION_NAME, PERSIST_DIRECTORY
from document_processor import parse_file, embed_and_store_text
import streamlit as st

#chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
callbacks = [StreamingStdOutCallbackHandler()]

local_llm = None
@st.cache_resource()
def load_gpt4all():
    local_path = "models/gpt4all/ggml-gpt4all-j-v1.3-groovy.bin"
    print(local_path)
    local_llm = GPT4All(model=local_path, n_ctx=512, verbose=True)

#set the embedding model as per the user selection
def get_embedding(model):
    if model.startswith("Open AI"):
        #embedding using OpenAI
        embedding_llm = OpenAIEmbeddings()
    else:
        #embedding using LlamaCppEmbeddings
        embedding_llm = HuggingFaceEmbeddings()
    return embedding_llm

#llm chain to get the result against the query
def get_result(qa, query: str) -> str:
    response = qa(query)
    #chat_history = [(query, response["result"])]
    print(response["result"])
    return response

# store the given file into vector store
def store_document(file: BytesIO, model: str) -> str:
    """Converts file into text, store into vector store and returns a vector qa chain"""
    #get the text from file
    docs = parse_file(file, file.type)
    
    print("File parsed")
    #store into vector store
    vector_store = embed_and_store_text(docs, model)
    vector_store.persist()
    print("Vector store created")
    return "File stored"

def load_qa(model)-> RetrievalQA:
    #load the model
    embedding_llm = get_embedding(model)
    vector_store = Chroma(
                        persist_directory=PERSIST_DIRECTORY, 
                        embedding_function=embedding_llm, 
                        collection_name=COLLECTION_NAME)
    if (vector_store is None):
        print("Vector store not found")
        return None
    else:
        print("Vector store found, returning chain")
        if model.startswith("Open AI"):
            qa_llm = OpenAI()
        else:
            if (local_llm is None):
                qa_llm = load_gpt4all()
            else:
                qa_llm = local_llm
        #qa_llm = OpenAI()   
        qa = RetrievalQA.from_chain_type(
            llm=qa_llm, 
            chain_type="stuff", 
            retriever=vector_store.as_retriever(), 
            return_source_documents=True,
            verbose=True
        )
        return qa
