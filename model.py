import os
from typing import TextIO
os.environ["OPENAI_API_KEY"]
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
llm = OpenAI(model_name="text-ada-001", temperature=0.9)

def get_result(csv: TextIO, query: str) -> str:
    agent = create_csv_agent(OpenAI(temperature=0), csv, verbose=False)
    response = agent.run(query)
    return response