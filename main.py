import streamlit as st
from streamlit_chat import message
import pandas as pd
import os
from model import get_result, load_qa, store_document
from openai.error import AuthenticationError
from constant import SUPPORTED_FILE_LIST

# page title and icon
st.set_page_config(page_title="My Docs QA", page_icon=":robot:")
st.header("QA with Personal Documents")

# session
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# clear submit
def clear_submit():
    st.session_state["submit"] = False

def setOpenApiKey(key):
    st.session_state["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_KEY"] = key
    print("key set to ", key)

if "OPENAI_API_KEY" in os.environ:
    st.session_state["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

key = None
# sidebar
with st.sidebar:
    st.title("LangChain with OpenAI/GPT4All")
    model = st.radio("Select Model", ("Local LLM", "Open AI"), index=1)
    if model.startswith("Open AI"):
        key = st.text_input(
            "Enter Open API Key", 
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx",
            help="Get your api key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=st.session_state.get("OPENAI_API_KEY", "")
        )

    if key is not None:
        setOpenApiKey(key)
    
    st.markdown("## Resources")
    st.markdown("[Github]()")

qa = None
print("model = ", model)

if model.startswith("Open AI") and (st.session_state["OPENAI_API_KEY"] is None or st.session_state["OPENAI_API_KEY"] == ""):
    raise AuthenticationError("Please enter API key and restart the app")
else:
    with st.spinner("Loading model & vector store..."):
        qa = load_qa(model)
    if (qa is None):
        st.write("No documents found. Please upload a document.")

file = st.file_uploader(
    "Upload a supported document", 
    type=SUPPORTED_FILE_LIST,
    on_change=clear_submit
)

if file is not None:
    if model.startswith("Open AI") and st.session_state["OPENAI_API_KEY"] is None:
        st.error("Please enter API key and restart the app")
        st.stop()

    try:
        with st.spinner("Embedding document..."):
            #qa = store_csv(dataframe, model)
            store_document(file, model)
            st.session_state["api_key_configured"] = True
    except Exception as e:
        st.error(e)
        st.stop()
    
if qa is not None:
    query = st.text_area("Write your query here")
    submit = st.button("Submit")
    if submit:
        response = get_result(qa, query)
        #st.markdown((response), unsafe_allow_html=True)
        st.session_state.past.append(query)
        st.session_state.generated.append(response["result"])
        with st.expander("Show raw response", expanded=False):
            st.json(response)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")