import os
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DataFrameLoader
from langchain.llm import LLMChain, PromptTemplate
from langchain.llm import OpenAI
from langchain.llm import GPT4All
from lanchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import TextIO
os.environ["OPENAI_API_KEY"] = 'sdbhfgesj'

def split_text(text):
    text_splitter = CharacterTextSplitter(chub_size=1000, overlap_size=0)
    chunks = text_splitter.split(text)
    return chunks

def embed_text(text, type="document"):
    chunks = split_text(text)
    llama = LlamaCppEmbeddings(model_path="models/llama_cpp/llama_cpp.bin")
    if (type == "document"):
        embeddings = llama.embed_documents(chunks)
    else:
        embeddings = llama.embed_query(chunks)
    
    return embeddings

def store_embeddings(embeddings):
    vector_store = Chroma.from_documents(embeddings)
    return vector_store


#create a class to handle the AI service like: embeding, store, query and return result from llms
class AIService:
    def __init__(self, db, llm_model, llm_api_key) -> None:
        self.db = db
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key

        #prompt template
        self.prompt_template = """Question: {query}
        Answer: Let's think step by step."""

    # load, split, embed and store csv data
    def store_csv(self, df):
        loader = DataFrameLoader(df, page_content_column="batter")
        document_data = loader.load()
        print(document_data)
        embeddings = embed_text(document_data, type="document")
        store_embeddings(embeddings)

    # search input query and return result from vector store
    def get_query_search_result(self, query):
        vector_store = self.db
        query_embedding = embed_text(query, type="query")
        results = vector_store.search(query_embedding, k=1)
        return results
    
    # get the result from llm model based on the query + context
    def get_llm_result(self, query, context):
        prompt = PromptTemplate(template=self.prompt_template, input_variables=[query])
        callbacks = [StreamingStdOutCallbackHandler()]
        if (self.llm_model.startswith("OpenAI")):
            os.environ["OPENAI_API_KEY"] = self.llm_api_key
            llm = OpenAI(temperature=0.9)
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            response = llm_chain.run(query)
        else:
            llm = GPT4All(model_path="models/gpt4all/gpt4all.bin", callbacks=callbacks, verbose=True)
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            response = llm_chain.run(query)
        return response