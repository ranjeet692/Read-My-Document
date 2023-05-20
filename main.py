import streamlit as st

from model import get_result

st.header("Read My Document")
file = st.file_uploader("Upload a csv file", type=["csv"])

if file is not None:
    query = st.text_area("Ask your questions, for example - 'List some top rated restaurants in html table format with columns restaurant name, rating and cuisine'")
    submit = st.button("Submit")
    if submit:
        st.markdown(get_result(file, query), unsafe_allow_html=True)