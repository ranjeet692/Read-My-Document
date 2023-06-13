import os
from io import BytesIO
from langchain.embeddings import LlamaCppEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import LlamaCpp, OpenAI, GPT4All
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.chains import VectorDBQA
from langchain.memory import ConversationBufferMemory
from constant import MODEL_DIRECTORY, COLLECTION_NAME, PERSIST_DIRECTORY, GPT4ALL
from document_processor import parse_file, embed_and_store_text

#chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
        #qa using OpenAI
        qa_llm = OpenAI()
    else:
        #qa using LlamaCppEmbeddings
        #qa_llm = LlamaCpp(model_path= f"{MODEL_DIRECTORY}/{LLAMA_CPP}")
        qa_llm = GPT4All(model_path= f"{MODEL_DIRECTORY}/{GPT4ALL}")
    return qa_llm

#llm chain to get the result against the query
def get_result(qa, query: str) -> str:
    print(qa)
    response = qa(query)
    #response = qa({"question": query, "chat_history": chat_history})
    chat_history = [(query, response["result"])]
    return response["result"]

# store the given file into vector store
def store_document(file: BytesIO, model: str) -> VectorDBQA:
    """Converts file into text, store into vector store and returns a vector qa chain"""
    #get the text from file
    text = parse_file(file, file_type="txt")
   
    #store into vector store
    vector_store = embed_and_store_text(text, model)
    print("Vector store created")

    #get the qa model
    qa_llm = get_qa_llm(model)

    #get the qa chain
    qa = VectorDBQA.from_llm(llm=qa_llm, chain_type="stuff", vectorstore=vector_store)

    return qa

#store the csv file into vector store
def store_csv(df, model):
    #get the name of first column of dataframe
    first_column = df.columns[0]
    loader = DataFrameLoader(df, page_content_column=first_column)
    document_data = loader.load()
    try:
        embedding_llm = get_embedding(model)
    except Exception as e:
        print(e)
        return "Error in loading model"

    #store into vector store
    vector_store = Chroma.from_documents(
        documents=document_data, 
        embedding = embedding_llm, 
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    print("Vector store created")

    #load the model
    qa_llm = get_qa_llm(model)
    qa = VectorDBQA.from_llm(llm=qa_llm, chain_type="stuff", vectorstore=vector_store)
    return qa

def load_qa(model):
    #load the model
    embedding_llm = get_embedding(model)
    qa_llm = get_qa_llm(model)
    vector_store = Chroma(
                        persist_directory=PERSIST_DIRECTORY, 
                        embedding_function=embedding_llm, 
                        collection_name=COLLECTION_NAME)
    if (vector_store is None):
        print("Vector store not found")
        return None
    else:
        print("Vector store found, returning chain")
        qa = VectorDBQA.from_llm(llm=qa_llm, vectorstore=vector_store)
        return qa
