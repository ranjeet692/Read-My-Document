import os
from io import BytesIO
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import OpenAI, GPT4All
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from constant import MODEL_DIRECTORY, COLLECTION_NAME, PERSIST_DIRECTORY, GPT4ALLMODEL
from document_processor import parse_file, embed_and_store_text

#chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
callbacks = [StreamingStdOutCallbackHandler()]

#set the embedding model as per the user selection
def get_embedding(model):
    if model.startswith("Open AI"):
        #embedding using OpenAI
        embedding_llm = OpenAIEmbeddings()
    else:
        #embedding using LlamaCppEmbeddings
        embedding_llm = HuggingFaceEmbeddings()
    return embedding_llm

#set the qa model as per the user selection
def get_qa_llm(model):
    if model.startswith("Open AI"):
        qa_llm = OpenAI()
    else:
        local_path = "./models/gpt4all/ggml-gpt4all-j-v1.3-groovy.bin"
        qa_llm = GPT4All(model=local_path, n_ctx=512)
    return qa_llm

#llm chain to get the result against the query
def get_result(qa, query: str) -> str:
    print(qa)
    response = qa(query)
    #response = qa({"question": query, "chat_history": chat_history})
    chat_history = [(query, response["result"])]
    return response["result"]

# store the given file into vector store
def store_document(file: BytesIO, model: str) -> str:
    """Converts file into text, store into vector store and returns a vector qa chain"""
    #get the text from file
    docs = parse_file(file, file.type)
    
    print("File parsed")
    #store into vector store
    vector_store = embed_and_store_text(docs, model)
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
        qa_llm = get_qa_llm(model)
        qa = RetrievalQA.from_chain_type(
            llm=qa_llm, 
            chain_type="stuff", 
            retriever=vector_store.as_retriever(), 
            memory=memory, 
            verbose=True
        )
        #qa = ConversationalRetrievalChain.from_llm(open_llm, vector_store.as_retriever(), memory=memory)
        return qa
