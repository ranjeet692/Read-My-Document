import streamlit as st
import pandas as pd

from model import get_result, store_csv

with st.sidebar:
    st.title("LangChain with OpenAI/GPT4All")
    model = st.radio("Select Model", ("GPT4ALL(Local LLM)", "Open AI - "))
    if model.startswith("Open AI"):
        key = st.text_input("Enter API Key")

    st.markdown("## Resources")
    st.markdown("[Github]()")

st.title("Query Document with LLM Models")
file = st.file_uploader("Upload a csv file", type=["csv"])
if file is not None:
    dataframe = pd.read_csv(file)
    vectore_store = store_csv(dataframe)

if file is not None:
    query = st.text_area("Write your query here")
    submit = st.button("Submit")
    if submit:
        docs = vectore_store.similarity_search(query)
        print(docs[0].page_content)
        st.markdown(get_result(file, query, model, key), unsafe_allow_html=True)