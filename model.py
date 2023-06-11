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
from constant import CHROMA_SETTINGS, MODEL_DIRECTORY, GPT4ALL, LLAMA_CPP

os.environ["OPENAI_API_KEY"] = 'sdbhfgesj'
#llama = LlamaCppEmbeddings(model_path="models/llama_cpp/llama_cpp.bin")

from langchain.llms import OpenAI
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
llm = OpenAI(model_name="text-ada-001", temperature=0.9)



def get_result(csv: TextIO, query: str) -> str:
    agent = create_csv_agent(OpenAI(temperature=0), csv, verbose=False)
    response = agent.run(query)
    return response

def store_csv(df):
    loader = DataFrameLoader(df, page_content_column="batter")
    document_data = loader.load()
    print(document_data)

    #embed data to store into vectior store
    text_splitter = CharacterTextSplitter(chub_size=1000, overlap_size=0)
    chunks = text_splitter.split(text)
    return chunks

    #embeddings = OpenAIEmbeddings()
    #vector_store = Chroma.from_documents(chunks, embeddings)

    #return vector_store
