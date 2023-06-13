import os
from io import BytesIO
from langchain.embeddings import LlamaCppEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import LlamaCpp, OpenAI, GPT4All
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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
        callbacks = [StreamingStdOutCallbackHandler()]
        path = os.path.join(MODEL_DIRECTORY, GPT4ALL)
        qa_llm = GPT4ALL(model="models/gpt4all/ggml-gpt4all-j-v1.3-groovy.bin", callbacks=callbacks, verbose=True)
        #qa_llm = GPT4All("ggml-gpt4all-j-v1.3-groovy")
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
    qa = RetrievalQA.from_llm(llm=qa_llm, chain_type="stuff", vectorstore=vector_store)
    return qa

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
