import streamlit as st
from streamlit_chat import message
import pandas as pd

from model import get_result, store_csv

# page title and icon
st.set_page_config(page_title="IPL 2023", page_icon=":robot:")
st.header("IPL 2023")

# session
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# sidebar
with st.sidebar:
    st.title("LangChain with OpenAI/GPT4All")
    model = st.radio("Select Model", ("GPT4ALL(Local LLM)", "Open AI - "))
    if model.startswith("Open AI"):
        key = st.text_input("Enter API Key")

    st.markdown("## Resources")
    st.markdown("[Github]()")

file = st.file_uploader("Upload a csv file", type=["csv"])
if file is not None:
    dataframe = pd.read_csv(file)
    vectore_store = store_csv(dataframe)

if file is not None:
    query = st.text_area("Write your query here")
    submit = st.button("Submit")
    if submit:
        qa = vectore_store.similarity_search(query)
        response = get_result(qa, query)
        st.markdown((response), unsafe_allow_html=True)
        
        st.session_state.past.append(query)
        st.session_state.generated.append(response)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")