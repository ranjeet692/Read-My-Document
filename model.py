import os
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from constant import CHROMA_SETTINGS, GPT4ALL, LLAMA_CPP, MODEL_DIRECTORY, COLLECTION_NAME

#chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_history = []

def get_result(qa, query: str) -> str:
    try :    
        response = qa({"question": query, "chat_history": chat_history})
        chat_history = [(query, response["answer"])]
        return response["answer"]
    except Exception as e:
        print(e)
    

def store_csv(df):
    loader = DataFrameLoader(df, page_content_column="batter")
    document_data = loader.load()
    print(document_data)

    #embed data to store into vectior store
    text_splitter = CharacterTextSplitter(chub_size=1000, overlap_size=0)
    chunks = text_splitter.split(document_data)

    #embedding using LlamaCppEmbeddings
    llama_embeddings = LlamaCppEmbeddings(model_path= f"{MODEL_DIRECTORY}/{LLAMA_CPP}")

    #store into vector store
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings = llama_embeddings, 
        collection_name=COLLECTION_NAME, 
        client_settings=CHROMA_SETTINGS
    )

    llama_llm = LlamaCpp(model_path= f"{MODEL_DIRECTORY}/{LLAMA_CPP}")
    qa = ConversationalRetrievalChain.from_llm(llama_llm, vector_store.as_retriever(), memory=memory)
    vector_store.persist()
    vector_store = None
    return qa